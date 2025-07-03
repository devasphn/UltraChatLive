# ==============================================================================
# UltraChat S2S - THE NEXT-GENERATION ARCHITECTURE
#
# This is a full architectural upgrade based on your excellent request.
#
# UPGRADE 1: End-to-End ASR+LLM
# - Whisper and Phi-3 are replaced with the new `ultravox-v0.6` model.
# - This is a single, powerful model designed for low-latency voice conversations.
#
# UPGRADE 2: State-of-the-Art Cloud TTS
# - Chatterbox is replaced with API calls to Google's Gemini TTS.
# - This provides incredibly natural, high-quality voices.
#
# UPGRADE 3: Secure API Key Handling
# - The code is designed to securely read your Google API key from an
#   environment variable.
#
# This is the new, state-of-the-art version of your agent.
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
import os
import io
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

# New imports for the Google API
import google.generativeai as genai

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av

from transformers import pipeline
import torch.hub

# --- Basic Setup ---
try:
    import uvloop
    uvloop.install()
    print("🚀 Using uvloop for asyncio event loop.")
except ImportError:
    print("⚠️ uvloop not found, using default asyncio event loop.")

warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('aioice.ice').setLevel(logging.WARNING)
logging.getLogger('aiortc').setLevel(logging.WARNING)

# --- Global Variables & HTML ---
ultravox_pipe, vad_model = None, None
executor = ThreadPoolExecutor(max_workers=4)
pcs = set()
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat Voice Assistant (Next-Gen)</title>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #2c3e50; color: #ecf0f1; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
        .container { background: #34495e; padding: 40px; border-radius: 10px; box-shadow: 0 10px 20px rgba(0,0,0,0.2); text-align: center; max-width: 600px; width: 100%; }
        h1 { margin-bottom: 30px; font-weight: 300; }
        button { background: #2ecc71; color: white; border: none; padding: 15px 30px; font-size: 18px; border-radius: 5px; cursor: pointer; margin: 10px; transition: all 0.3s; }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        button:disabled { background: #95a5a6; cursor: not-allowed; transform: none; box-shadow: none; }
        .stop-btn { background: #e74c3c; } .stop-btn:hover { background: #c0392b; }
        .status { margin: 20px 0; padding: 15px; border-radius: 5px; font-weight: 500; transition: background-color 0.5s, box-shadow 0.3s; }
        .status.connected { background: #27ae60; }
        .status.disconnected { background: #c0392b; }
        .status.connecting { background: #f39c12; }
        .status.speaking { background: #3498db; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(52, 152, 219, 0); } 100% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0); } }
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
        if(ws || pc) { stop(); }
        startBtn.disabled = true;
        updateStatus('🔄 Connecting...', 'connecting');
        try {
            localStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
            pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
            
            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

            pc.ontrack = e => {
                console.log('Remote track received!');
                if (remoteAudio.srcObject !== e.streams[0]) {
                    remoteAudio.srcObject = e.streams[0];
                    remoteAudio.play().catch(err => console.error("Autoplay failed:", err));
                    
                    remoteAudio.onplaying = () => { if(pc && pc.connectionState === 'connected') updateStatus('🤖 AI Speaking...', 'speaking'); };
                    remoteAudio.onended = () => { if(pc && pc.connectionState === 'connected') updateStatus('✅ Listening...', 'connected'); };
                }
            };

            pc.onicecandidate = e => { if (e.candidate && ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify({ type: 'ice-candidate', candidate: e.candidate.toJSON() })); } };
            
            pc.onconnectionstatechange = () => {
                const state = pc.connectionState;
                if (state === 'connected') { updateStatus('✅ Listening...', 'connected'); stopBtn.disabled = false; }
                else if (state === 'failed' || state === 'closed' || state === 'disconnected') { stop(); }
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
                if (data.type === 'answer' && !pc.currentRemoteDescription) { await pc.setRemoteDescription(new RTCSessionDescription(data)); }
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

# --- Utility Functions ---
def candidate_from_sdp(candidate_string: str) -> dict:
    if candidate_string.startswith("candidate:"): candidate_string = candidate_string[10:]
    bits = candidate_string.split()
    if len(bits) < 8: raise ValueError(f"Invalid candidate string: {candidate_string}")
    params = {'component': int(bits[1]), 'foundation': bits[0], 'ip': bits[4], 'port': int(bits[5]), 'priority': int(bits[3]), 'protocol': bits[2], 'type': bits[7]}
    for i in range(8, len(bits) - 1, 2):
        if bits[i] == "raddr": params['relatedAddress'] = bits[i + 1]
        elif bits[i] == "rport": params['relatedPort'] = int(bits[i + 1])
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
            speech_timestamps = self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=sample_rate, min_speech_duration_ms=250, threshold=0.5)
            return len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return True

def initialize_models():
    global ultravox_pipe, vad_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"🚀 Initializing models on device: {device} with dtype: {torch_dtype}")
    
    # 1. Configure Google API
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("❌ GOOGLE_API_KEY environment variable not set. Cannot use Gemini TTS.")
            return False
        genai.configure(api_key=api_key)
        logger.info("✅ Google Gemini API configured successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to configure Google API: {e}")
        return False

    # 2. Load VAD
    vad_model = SileroVAD()
    if not vad_model.model: return False
        
    # 3. Load Ultravox
    try:
        logger.info("📥 Loading Ultravox v0.6 model (`fixie-ai/ultravox-v0_6`)...")
        ultravox_pipe = pipeline("text-generation", model="fixie-ai/ultravox-v0_6", device_map="auto", torch_dtype=torch_dtype, trust_remote_code=True)
        logger.info("✅ Ultravox v0.6 loaded successfully")
        
        logger.info("🎉 All models loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Critical model loading error: {e}", exc_info=True)
        return False

# --- Audio Processing Classes ---
class AudioBuffer:
    def __init__(self, max_duration=5.0, sample_rate=16000):
        self.sample_rate, self.max_samples = sample_rate, int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time, self.min_speech_samples, self.process_interval = time.time(), int(0.4 * sample_rate), 0.5
    
    def add_audio(self, audio_data):
        if audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32) / 32768.0
        if np.abs(audio_data).max() > 1.0: audio_data /= np.abs(audio_data).max()
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
        self._queue, self._current_chunk, self._chunk_pos = asyncio.Queue(), None, 0
        self._timestamp, self._time_base = 0, fractions.Fraction(1, 48000)

    async def clear(self):
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except asyncio.QueueEmpty: continue
        self._current_chunk, self._chunk_pos = None, 0
        
    async def recv(self):
        frame_samples, frame = 960, np.zeros(960, dtype=np.int16)
        if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
            try: self._current_chunk, self._chunk_pos = await asyncio.wait_for(self._queue.get(), timeout=0.01), 0
            except asyncio.TimeoutError: pass
        if self._current_chunk is not None:
            end_pos, chunk_part = self._chunk_pos + frame_samples, self._current_chunk[self._chunk_pos:self._chunk_pos + frame_samples]
            frame[:len(chunk_part)], self._chunk_pos = chunk_part, self._chunk_pos + len(chunk_part)
        audio_frame = av.AudioFrame.from_ndarray(np.array([frame]), format="s16", layout="mono")
        audio_frame.pts, audio_frame.sample_rate = self._timestamp, 48000
        self._timestamp += frame_samples
        return audio_frame

    async def queue_audio(self, audio_float32):
        if audio_float32.size > 0: await self._queue.put((audio_float32 * 32767).astype(np.int16))

class AudioProcessor:
    def __init__(self, output_track: ResponseAudioTrack, executor: ThreadPoolExecutor):
        self.track, self.buffer, self.output_track, self.task, self.executor = None, AudioBuffer(), output_track, None, executor
        self.processing_task = None

    def add_track(self, track): self.track = track
    async def start(self): self.task = asyncio.create_task(self._run())
    async def stop(self):
        if self.processing_task: self.processing_task.cancel()
        if self.task: self.task.cancel()

    async def _run(self):
        try:
            while True:
                frame = await self.track.recv()
                audio_float32 = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
                resampled_audio = librosa.resample(audio_float32, orig_sr=frame.sample_rate, target_sr=16000)
                self.buffer.add_audio(resampled_audio)
                if self.buffer.should_process():
                    if self.processing_task and not self.processing_task.done():
                        logger.info("🎤 Barge-in detected, cancelling previous task...")
                        self.processing_task.cancel()
                    audio_to_process = self.buffer.get_audio_array()
                    self.buffer.reset()
                    logger.info(f"🧠 VAD triggered, processing {len(audio_to_process)} samples...")
                    self.processing_task = asyncio.create_task(self.process_speech(audio_to_process))
        except asyncio.CancelledError: pass
        except mediastreams.MediaStreamError: logger.warning("Client media stream ended.")
        except Exception as e: logger.error(f"Audio processor error: {e}", exc_info=True)
        finally: logger.info("Audio processor stopped.")
            
    def _blocking_ultravox_gemini_tts(self, audio_array) -> np.ndarray:
        try:
            # 1. Ultravox (ASR + LLM)
            turns = [
                {"role": "system", "content": "You are a friendly, conversational voice assistant named UltraChat. Keep your answers concise and natural."}
            ]
            result = ultravox_pipe({'audio': audio_array, 'turns': turns, 'sampling_rate': 16000}, max_new_tokens=150)
            response_text = parse_ultravox_response(result).strip().strip('"')
            if not response_text: return np.array([], dtype=np.float32)
            logger.info(f"🤖 Ultravox Response: '{response_text}'")

            # 2. Google Gemini TTS API Call
            logger.info("🔊 Requesting audio from Google Gemini TTS API...")
            response = genai.text_to_speech(
                model="models/text-to-speech",
                text=response_text,
                voice="gemini-1.5-flash" # High-quality and fast voice
            )
            
            # 3. Convert API audio bytes to numpy array
            audio_bytes = response['audio_content']
            # Use soundfile to read the raw audio bytes (e.g., WAV) into a numpy array
            audio_data, original_sr = sf.read(io.BytesIO(audio_bytes))
            logger.info(f"✅ Received audio from Google. Original SR: {original_sr}Hz")

            # 4. Resample to 48kHz for WebRTC if necessary
            if original_sr != 48000:
                audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=original_sr, target_sr=48000)

            return audio_data
        except Exception as e:
            logger.error(f"Error in background processing thread: {e}", exc_info=True)
            return np.array([], dtype=np.float32)

    async def process_speech(self, audio_array):
        try:
            loop = asyncio.get_running_loop()
            resampled_wav = await loop.run_in_executor(self.executor, self._blocking_ultravox_gemini_tts, audio_array)
            await self.output_track.clear()
            if resampled_wav.size > 0:
                await self.output_track.queue_audio(resampled_wav)
        except asyncio.CancelledError:
            logger.info("Speech processing task was cancelled successfully by barge-in.")
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
        if track.kind == "audio":
            output_audio_track = ResponseAudioTrack()
            pc.addTrack(output_audio_track)
            processor = AudioProcessor(output_audio_track, executor)
            processor.add_track(track)
            asyncio.create_task(processor.start())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed", "disconnected"]: await pc.close()

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data["type"] == "offer":
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=data["type"]))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
                elif data["type"] == "ice-candidate" and data.get("candidate"):
                    try:
                        candidate_data = data["candidate"]
                        candidate_string = candidate_data.get("candidate")
                        if candidate_string:
                            params = candidate_from_sdp(candidate_string)
                            await pc.addIceCandidate(RTCIceCandidate(sdpMid=candidate_data.get("sdpMid"), sdpMLineIndex=candidate_data.get("sdpMLineIndex"), **params))
                    except Exception as e: logger.error(f"Error adding ICE candidate: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}", exc_info=True)
    finally:
        logger.info(f"Closing connection for pc: {id(pc)}")
        if processor: await processor.stop()
        if pc in pcs: pcs.remove(pc)
        if pc.connectionState != "closed": await pc.close()
    return ws

async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type='text/html')

# --- Main Application Logic ---
async def on_shutdown(app):
    logger.info("Shutting down server...")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
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
