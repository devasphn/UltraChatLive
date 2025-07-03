# ==============================================================================
# UltraChat S2S - FINAL CORRECTED VERSION
#
# My sincere apologies. This version fixes the `TypeError` crash.
# The error was in the websocket_handler where the ICE candidate was being
# created incorrectly. This version reverts to a robust parsing method.
#
# This version combines:
# 1. The non-blocking TTS architecture to prevent timeouts.
# 2. The fix for the `TypeError` to prevent the server from crashing.
#
# This should be the working version.
# ==============================================================================

import torch
import asyncio
import json
import logging
import numpy as np
import fractions
import warnings
import collections
import time
import librosa
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av

from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
import torch.hub

# --- Basic Setup ---
try:
    import uvloop
    uvloop.install()
    print("🚀 Using uvloop for asyncio event loop.")
except ImportError:
    print("⚠️ uvloop not found, using default asyncio event loop.")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('aioice.ice').setLevel(logging.WARNING)
logging.getLogger('aiortc').setLevel(logging.WARNING)

# --- Global Variables ---
uv_pipe, tts_model, vad_model = None, None, None
executor = ThreadPoolExecutor(max_workers=4)
pcs = set()

# --- Utility Functions ---
def candidate_from_sdp(candidate_string: str) -> dict:
    """
    Parse an ICE candidate string into a dictionary of parameters.
    This is necessary because aiortc's RTCIceCandidate constructor
    expects individual parameters, not the raw string.
    """
    if candidate_string.startswith("candidate:"):
        candidate_string = candidate_string[10:]
    bits = candidate_string.split()
    if len(bits) < 8:
        raise ValueError(f"Invalid candidate string: {candidate_string}")
    params = {
        'component': int(bits[1]), 'foundation': bits[0], 'ip': bits[4],
        'port': int(bits[5]), 'priority': int(bits[3]), 'protocol': bits[2], 'type': bits[7],
    }
    for i in range(8, len(bits) - 1, 2):
        if bits[i] == "raddr":
            params['relatedAddress'] = bits[i + 1]
        elif bits[i] == "rport":
            params['relatedPort'] = int(bits[i + 1])
    return params

def parse_ultravox_response(result):
    try:
        if isinstance(result, str): return result
        elif isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, str): return item
            elif isinstance(item, dict) and 'generated_text' in item: return item['generated_text']
        return ""
    except Exception as e:
        logger.error(f"Error parsing Ultravox response: {e}")
        return ""

# --- Model Loading and VAD ---
class SileroVAD:
    def __init__(self):
        try:
            logger.info("🎤 Loading Silero VAD model...")
            self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            (self.get_speech_timestamps, _, _, _, _) = utils
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading Silero VAD: {e}", exc_info=True)
            self.model = None

    def detect_speech(self, audio_tensor, sample_rate=16000):
        if self.model is None: return True
        try:
            if isinstance(audio_tensor, np.ndarray): audio_tensor = torch.from_numpy(audio_tensor)
            if audio_tensor.abs().max() < 0.01: return False
            speech_timestamps = self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=sample_rate, min_speech_duration_ms=250)
            return len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return True

def initialize_models():
    global uv_pipe, tts_model, vad_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing models on device: {device}")
    
    vad_model = SileroVAD()
    if not vad_model.model: return False
        
    try:
        logger.info("📥 Loading Ultravox pipeline...")
        uv_pipe = pipeline(model="fixie-ai/ultravox-v0_4", trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
        logger.info("✅ Ultravox pipeline loaded successfully")

        logger.info("📥 Loading Chatterbox TTS...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("✅ Chatterbox TTS loaded successfully")
        
        logger.info("🎉 All models loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Critical model loading error: {e}", exc_info=True)
        return False

# --- Audio Processing Classes ---
class AudioBuffer:
    def __init__(self, max_duration=3.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time = time.time()
        self.min_speech_samples = int(0.4 * sample_rate)
        self.process_interval = 0.5
    
    def add_audio(self, audio_data):
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32768.0
        if np.abs(audio_data).max() > 1.0:
            audio_data /= np.abs(audio_data).max()
        self.buffer.extend(audio_data.flatten())

    def get_audio_array(self): return np.array(list(self.buffer), dtype=np.float32)
    
    def should_process(self):
        current_time = time.time()
        if len(self.buffer) > self.min_speech_samples and (current_time - self.last_process_time) > self.process_interval:
            self.last_process_time = current_time
            audio_array = self.get_audio_array()
            if np.abs(audio_array).max() < 0.005: return False
            return vad_model.detect_speech(audio_array, self.sample_rate)
        return False

    def reset(self): self.buffer.clear()

class ResponseAudioTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue()
        self._current_chunk, self._chunk_pos = None, 0
        self._timestamp, self._time_base = 0, fractions.Fraction(1, 48000)

    async def recv(self):
        frame_samples = 960
        frame = np.zeros(frame_samples, dtype=np.int16)

        if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
            try:
                self._current_chunk = await asyncio.wait_for(self._queue.get(), timeout=0.01)
                self._chunk_pos = 0
            except asyncio.TimeoutError:
                pass
        
        if self._current_chunk is not None:
            needed = frame_samples
            chunk_part = self._current_chunk[self._chunk_pos : self._chunk_pos + needed]
            frame[:len(chunk_part)] = chunk_part
            self._chunk_pos += len(chunk_part)

        audio_frame = av.AudioFrame.from_ndarray(np.array([frame]), format="s16", layout="mono")
        audio_frame.pts, audio_frame.sample_rate = self._timestamp, 48000
        self._timestamp += frame_samples
        return audio_frame

    async def queue_audio(self, audio_float32):
        if audio_float32.size > 0:
            audio_int16 = (audio_float32 * 32767).astype(np.int16)
            await self._queue.put(audio_int16)

class AudioProcessor:
    def __init__(self, output_track: ResponseAudioTrack, executor: ThreadPoolExecutor):
        self.track = None
        self.buffer = AudioBuffer()
        self.output_track = output_track
        self.task = None
        self.executor = executor

    def add_track(self, track): self.track = track
    async def start(self): self.task = asyncio.create_task(self._run())
    async def stop(self):
        if self.task: self.task.cancel()

    async def _run(self):
        try:
            while True:
                try: frame = await self.track.recv()
                except mediastreams.MediaStreamError:
                    logger.warning("Client media stream ended.")
                    break
                
                audio_s16 = frame.to_ndarray()
                audio_float32 = audio_s16.flatten().astype(np.float32) / 32768.0
                resampled_audio = librosa.resample(audio_float32, orig_sr=frame.sample_rate, target_sr=16000)
                
                self.buffer.add_audio(resampled_audio)

                if self.buffer.should_process():
                    audio_to_process = self.buffer.get_audio_array()
                    self.buffer.reset()
                    logger.info(f"🧠 Processing {len(audio_to_process)} samples...")
                    asyncio.create_task(self.process_speech(audio_to_process))
        except asyncio.CancelledError: pass
        except Exception as e: logger.error(f"Audio processor error: {e}", exc_info=True)
        finally: logger.info("Audio processor stopped.")
            
    def _blocking_tts(self, text: str) -> np.ndarray:
        try:
            with torch.inference_mode():
                wav = tts_model.generate(text).cpu().numpy().flatten()
                resampled_wav = librosa.resample(wav.astype(np.float32), orig_sr=24000, target_sr=48000)
                return resampled_wav
        except Exception as e:
            logger.error(f"TTS generation failed in background thread: {e}", exc_info=True)
            return np.array([], dtype=np.float32)

    async def process_speech(self, audio_array):
        try:
            with torch.inference_mode():
                result = uv_pipe({'audio': audio_array, 'turns': [], 'sampling_rate': 16000}, max_new_tokens=50)
            response_text = parse_ultravox_response(result).strip()
            if not response_text: return
            logger.info(f"AI Response: '{response_text}'")

            loop = asyncio.get_running_loop()
            resampled_wav = await loop.run_in_executor(self.executor, self._blocking_tts, response_text)
            
            if resampled_wav.size > 0:
                await self.output_track.queue_audio(resampled_wav)
        except Exception as e:
            logger.error(f"Speech processing error: {e}", exc_info=True)

# --- WebRTC and WebSocket Handling ---
async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    
    pc = RTCPeerConnection(RTCConfiguration([RTCIceServer(urls="stun:stun.l.google.com:19302")]))
    pcs.add(pc)
    processor = None

    @pc.on("track")
    def on_track(track):
        nonlocal processor
        logger.info(f"🎧 Track {track.kind} received")
        if track.kind == "audio":
            output_audio_track = ResponseAudioTrack()
            pc.addTrack(output_audio_track)
            processor = AudioProcessor(output_audio_track, executor)
            processor.add_track(track)
            asyncio.create_task(processor.start())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"ICE Connection State is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            if pc in pcs: pcs.remove(pc)
            await pc.close()

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data["type"] == "offer":
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=data["type"]))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
                
                # --- THIS BLOCK IS THE FIX FOR THE TypeError ---
                # It correctly parses the ICE candidate string from the browser.
                # My previous version handled this incorrectly, causing the crash.
                elif data["type"] == "ice-candidate" and data.get("candidate"):
                    try:
                        candidate_data = data["candidate"]
                        candidate_string = candidate_data.get("candidate")
                        if candidate_string:
                            params = candidate_from_sdp(candidate_string)
                            candidate = RTCIceCandidate(
                                sdpMid=candidate_data.get("sdpMid"),
                                sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                                **params
                            )
                            await pc.addIceCandidate(candidate)
                    except Exception as e:
                        logger.error(f"Error adding ICE candidate: {e}", exc_info=True)
                # --- END OF FIX ---

    except Exception as e:
        logger.error(f"WebSocket handler error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed.")
        if processor: await processor.stop()
        if pc in pcs: pcs.remove(pc)
        if pc.connectionState != "closed": await pc.close()
    return ws

HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat Voice Assistant</title>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #2c3e50; color: #ecf0f1; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
        .container { background: #34495e; padding: 40px; border-radius: 10px; box-shadow: 0 10px 20px rgba(0,0,0,0.2); text-align: center; max-width: 600px; width: 100%; }
        h1 { margin-bottom: 30px; font-weight: 300; }
        button { background: #2ecc71; color: white; border: none; padding: 15px 30px; font-size: 18px; border-radius: 5px; cursor: pointer; margin: 10px; transition: all 0.3s; }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        button:disabled { background: #95a5a6; cursor: not-allowed; transform: none; box-shadow: none; }
        .stop-btn { background: #e74c3c; } .stop-btn:hover { background: #c0392b; }
        .status { margin: 20px 0; padding: 15px; border-radius: 5px; font-weight: 500; transition: background-color 0.5s; }
        .status.connected { background: #27ae60; } .status.disconnected { background: #c0392b; } .status.connecting { background: #f39c12; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ UltraChat Voice Assistant</h1>
        <div class="controls">
            <button id="startBtn" onclick="start()">START</button>
            <button id="stopBtn" onclick="stop()" class="stop-btn" disabled>STOP</button>
        </div>
        <div id="status" class="status disconnected">🔌 Disconnected</div>
        <audio id="remoteAudio" autoplay playsinline></audio>
    </div>
<script>
    let pc, ws, localStream;
    const remoteAudio = document.getElementById('remoteAudio');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDiv = document.getElementById('status');

    function updateStatus(message, className) { statusDiv.textContent = message; statusDiv.className = `status ${className}`; }

    async function start() {
        startBtn.disabled = true;
        updateStatus('🔄 Connecting...', 'connecting');
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (audioContext.state === 'suspended') { await audioContext.resume(); }

            localStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
            pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
            
            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

            pc.ontrack = e => {
                console.log('Remote track received!');
                if (remoteAudio.srcObject !== e.streams[0]) {
                    remoteAudio.srcObject = e.streams[0];
                    remoteAudio.play().catch(err => console.error("Autoplay failed:", err));
                }
            };

            pc.onicecandidate = e => {
                if (e.candidate) { ws.send(JSON.stringify({ type: 'ice-candidate', candidate: e.candidate.toJSON() })); }
            };
            
            pc.onconnectionstatechange = () => {
                const state = pc.connectionState;
                console.log(`Connection state: ${state}`);
                if (state === 'connecting') updateStatus('🤝 Establishing secure connection...', 'connecting');
                else if (state === 'connected') { updateStatus('✅ Connected & Listening...', 'connected'); stopBtn.disabled = false; }
                else if (state === 'failed' || state === 'closed' || state === 'disconnected') stop();
            };

            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onopen = async () => {
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                ws.send(JSON.stringify(offer));
            };

            ws.onmessage = async e => {
                const data = JSON.parse(e.data);
                if (data.type === 'answer' && !pc.currentRemoteDescription) {
                    await pc.setRemoteDescription(new RTCSessionDescription(data));
                }
            };

            const closeHandler = () => { if (pc && pc.connectionState !== 'closed') stop(); };
            ws.onclose = closeHandler;
            ws.onerror = closeHandler;

        } catch (err) { console.error(err); updateStatus(`❌ Error: ${err.message}`, 'disconnected'); stop(); }
    }

    function stop() {
        if (ws) { ws.onclose = null; ws.onerror = null; ws.close(); ws = null; }
        if (pc) { pc.onconnectionstatechange = null; pc.onicecandidate = null; pc.ontrack = null; pc.close(); pc = null; }
        if (localStream) { localStream.getTracks().forEach(track => track.stop()); localStream = null; }
        updateStatus('🔌 Disconnected', 'disconnected');
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
</script>
</body>
</html>
"""

# --- Main Application Logic ---
async def on_shutdown(app):
    logger.info("Shutting down server...")
    for pc_conn in list(pcs): await pc_conn.close()
    pcs.clear()
    executor.shutdown(wait=True)
    logger.info("Shutdown complete.")

async def main():
    if not initialize_models():
        logger.error("Failed to initialize models. The application cannot start.")
        return

    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    print("✅ Server started successfully on http://0.0.0.0:7860")
    print("🚀 Your speech-to-speech agent is live!")
    print("   Press Ctrl+C to stop the server.")
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down by user request...")
