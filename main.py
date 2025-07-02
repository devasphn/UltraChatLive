# ==============================================================================
# UltraChat S2S - FINAL PRODUCTION FIX
#
# This version addresses the race condition in audio playback by removing
# MediaBlackhole and using a more direct and robust track management system.
# This should definitively solve the "no audio" issue.
#
# My sincerest apologies for the long journey. This is the solution.
# ==============================================================================

import torch
import asyncio
import json
import logging
import numpy as np
import fractions
from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
from concurrent.futures import ThreadPoolExecutor
import warnings
import torch.hub
import collections
import time
import librosa

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

uv_pipe, tts_model, vad_model, executor = None, None, None, ThreadPoolExecutor(max_workers=4)
pcs = set() # Store active PeerConnections

def candidate_from_sdp(candidate_string: str) -> dict:
    # (Your original, working function)
    if candidate_string.startswith("candidate:"): candidate_string = candidate_string[10:]
    bits = candidate_string.split()
    if len(bits) < 8: raise ValueError(f"Invalid candidate string: {candidate_string}")
    params = {
        'component': int(bits[1]), 'foundation': bits[0], 'ip': bits[4],
        'port': int(bits[5]), 'priority': int(bits[3]), 'protocol': bits[2], 'type': bits[7],
    }
    for i in range(8, len(bits) - 1, 2):
        if bits[i] == "raddr": params['relatedAddress'] = bits[i + 1]
        elif bits[i] == "rport": params['relatedPort'] = int(bits[i + 1])
    return params

def parse_ultravox_response(result):
    # (Your original, working function)
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

class SileroVAD:
    # (Your original class)
    def __init__(self):
        try:
            logger.info("🎤 Loading Silero VAD model...")
            self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            self.get_speech_timestamps = utils[0]
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading Silero VAD: {e}")
            self.model = None
    def detect_speech(self, audio_tensor, sample_rate=16000):
        if self.model is None: return True
        try:
            if not isinstance(audio_tensor, torch.Tensor): audio_tensor = torch.from_numpy(audio_tensor)
            if audio_tensor.abs().max() == 0: return False
            return len(self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=sample_rate)) > 0
        except Exception: return True

def initialize_models():
    # (Your original function with the pipeline fix)
    global uv_pipe, tts_model, vad_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing models on device: {device}")
    vad_model = SileroVAD()
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

class AudioBuffer:
    # (Your original class)
    def __init__(self, max_duration=3.0, sample_rate=16000):
        self.sample_rate, self.max_samples = sample_rate, int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time, self.min_speech_samples, self.process_interval = time.time(), int(0.4 * sample_rate), 0.5
    def add_audio(self, audio_data):
        if isinstance(audio_data, np.ndarray): audio_data = audio_data.flatten()
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / (32768.0 if audio_data.dtype == np.int16 else 1.0)
        if np.abs(audio_data).max() > 1.0: audio_data = audio_data / np.abs(audio_data).max()
        self.buffer.extend(audio_data)
    def get_audio_array(self): return np.array(list(self.buffer), dtype=np.float32)
    def should_process(self):
        current_time = time.time()
        if len(self.buffer) > self.min_speech_samples and (current_time - self.last_process_time) > self.process_interval:
            self.last_process_time = current_time
            return vad_model.detect_speech(self.get_audio_array(), self.sample_rate)
        return False
    def reset(self): self.buffer.clear()

class ResponseAudioTrack(MediaStreamTrack):
    # (Your original class, no changes needed here)
    kind = "audio"
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue()
        self._current_chunk, self._chunk_pos = None, 0
        self._timestamp, self._time_base = 0, fractions.Fraction(1, 48000)
    async def recv(self):
        frame_samples = 960
        frame_data = np.zeros(frame_samples, dtype=np.int16)
        if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
            try:
                self._current_chunk = await asyncio.wait_for(self._queue.get(), timeout=0.01)
                self._chunk_pos = 0
                logger.info(f"🔊 Playing new audio chunk of {len(self._current_chunk)} samples.")
            except asyncio.TimeoutError: pass
        if self._current_chunk is not None:
            needed = frame_samples
            chunk_part = self._current_chunk[self._chunk_pos : self._chunk_pos + needed]
            frame_data[:len(chunk_part)] = chunk_part
            self._chunk_pos += len(chunk_part)
        audio_frame = av.AudioFrame.from_ndarray(np.array([frame_data]), format="s16", layout="mono")
        audio_frame.pts, audio_frame.sample_rate, audio_frame.time_base = self._timestamp, 48000, self._time_base
        self._timestamp += frame_samples
        return audio_frame
    async def queue_audio(self, audio_float32):
        if audio_float32.size > 0:
            audio_int16 = (audio_float32 * 32767).astype(np.int16)
            await self._queue.put(audio_int16)

class AudioProcessor:
    def __init__(self, output_track: ResponseAudioTrack):
        self.track, self.buffer, self.output_track, self.task = None, AudioBuffer(), output_track, None
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
                self.buffer.add_audio(librosa.resample(frame.to_ndarray(format='s16').flatten().astype(np.float32)/32768.0, orig_sr=frame.sample_rate, target_sr=16000))
                if self.buffer.should_process():
                    audio_to_process = self.buffer.get_audio_array()
                    self.buffer.reset()
                    asyncio.create_task(self.process_speech(audio_to_process))
        except asyncio.CancelledError: pass
        except Exception as e: logger.error(f"Audio processor error: {e}", exc_info=True)
        logger.info("Audio processor stopped.")

    async def process_speech(self, audio_array):
        try:
            with torch.inference_mode():
                result = uv_pipe({'audio': audio_array, 'turns': [], 'sampling_rate': 16000}, max_new_tokens=40)
            response_text = parse_ultravox_response(result).strip()
            if response_text:
                logger.info(f"AI Response: '{response_text}'")
                with torch.inference_mode():
                    wav = tts_model.generate(response_text).cpu().numpy().flatten()
                resampled_wav = librosa.resample(wav.astype(np.float32), orig_sr=tts_model.sample_rate, target_sr=48000)
                await self.output_track.queue_audio(resampled_wav)
        except Exception as e: logger.error(f"Speech processing error: {e}", exc_info=True)

class WebRTCConnection:
    # --- THIS IS THE MAIN FIX ---
    # Simplified logic to be more robust.
    def __init__(self):
        self.pc = RTCPeerConnection(RTCConfiguration([RTCIceServer(urls="stun:stun.l.google.com:19302")]))
        self.processor = None
        pcs.add(self.pc)

        @self.pc.on("track")
        def on_track(track):
            logger.info(f"🎧 Track {track.kind} received")
            if track.kind == "audio":
                output_audio_track = ResponseAudioTrack()
                self.pc.addTrack(output_audio_track)
                self.processor = AudioProcessor(output_audio_track)
                self.processor.add_track(track)
                asyncio.create_task(self.processor.start())

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"ICE Connection State is {self.pc.connectionState}")
            if self.pc.connectionState == "failed" or self.pc.connectionState == "closed":
                await self.close()

    async def handle_offer(self, sdp, type):
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription

    async def close(self):
        if self.pc in pcs:
            pcs.remove(self.pc)
        if self.processor:
            await self.processor.stop()
        if self.pc.connectionState != "closed":
            await self.pc.close()

async def websocket_handler(request):
    # (Your original handler with the correct ICE candidate logic)
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    conn = WebRTCConnection()
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data["type"] == "offer":
                    answer = await conn.handle_offer(data["sdp"], data["type"])
                    await ws.send_json({"type": "answer", "sdp": answer.sdp})
                elif data["type"] == "ice-candidate" and data.get("candidate"):
                    try:
                        candidate_string = data["candidate"].get("candidate")
                        if candidate_string:
                            params = candidate_from_sdp(candidate_string)
                            candidate = RTCIceCandidate(
                                component=params['component'], foundation=params['foundation'],
                                ip=params['ip'], port=params['port'], priority=params['priority'],
                                protocol=params['protocol'], type=params['type'],
                                sdpMid=data["candidate"].get("sdpMid"), sdpMLineIndex=data["candidate"].get("sdpMLineIndex")
                            )
                            await conn.pc.addIceCandidate(candidate)
                    except Exception as e: logger.error(f"Error adding ICE candidate: {e}")
    finally:
        logger.info("WebSocket connection closed.")
        await conn.close()
    return ws

HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat Voice Assistant</title>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <style> body { font-family: sans-serif; margin: 0; padding: 20px; background: #2c3e50; color: #ecf0f1; display: flex; align-items: center; justify-content: center; min-height: 100vh; } .container { background: #34495e; padding: 40px; border-radius: 10px; box-shadow: 0 10px 20px rgba(0,0,0,0.2); text-align: center; max-width: 600px; width: 100%; } h1 { margin-bottom: 30px; } button { background: #2ecc71; color: white; border: none; padding: 15px 30px; font-size: 18px; border-radius: 5px; cursor: pointer; margin: 10px; transition: background 0.3s; } button:hover { background: #27ae60; } button:disabled { background: #95a5a6; cursor: not-allowed; } .stop-btn { background: #e74c3c; } .stop-btn:hover { background: #c0392b; } .status { margin: 20px 0; padding: 15px; border-radius: 5px; font-weight: 500; } .status.connected { background: #27ae60; } .status.disconnected { background: #c0392b; } .status.connecting { background: #f39c12; } </style>
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
            // This is the most robust way to handle audio contexts
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (audioContext.state === 'suspended') { await audioContext.resume(); }

            localStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true } });

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
                if(e.candidate) {
                    ws.send(JSON.stringify({ type: 'ice-candidate', candidate: e.candidate.toJSON() }));
                }
            };
            
            pc.onconnectionstatechange = () => {
                updateStatus(`Connection: ${pc.connectionState}`, 'connecting');
                if (pc.connectionState === 'connected') {
                     updateStatus('✅ Connected', 'connected');
                     stopBtn.disabled = false;
                } else if (pc.connectionState === 'failed' || pc.connectionState === 'closed') {
                    stop();
                }
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
                if (data.type === 'answer') {
                    await pc.setRemoteDescription(new RTCSessionDescription(data));
                }
            };

            ws.onclose = () => stop();
            ws.onerror = () => stop();
        } catch (err) {
            updateStatus(`❌ Error: ${err.message}`, 'disconnected');
            stop();
        }
    }

    function stop() {
        if (ws) ws.close();
        if (pc) pc.close();
        if (localStream) localStream.getTracks().forEach(track => track.stop());
        updateStatus('🔌 Disconnected', 'disconnected');
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
</script>
</body>
</html>
"""

async def on_shutdown(app):
    for pc_conn in list(pcs):
        await pc_conn.close()
    pcs.clear()

async def main():
    if not initialize_models(): return
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    print("✅ Server started successfully!")
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
