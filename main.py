
import torch
import asyncio
import json
import logging
import numpy as np
import fractions
from aiohttp import web, WSMsgType
# --- FIX 1: Import the correct MediaStreamError ---
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av
from aiortc.contrib.media import MediaBlackhole
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
from concurrent.futures import ThreadPoolExecutor
import warnings
import torch.hub
import collections
import time
import librosa

# Try to install and use uvloop for better performance
try:
    import uvloop
    uvloop.install()
    print("🚀 Using uvloop for asyncio event loop.")
except ImportError:
    print("⚠️ uvloop not found, using default asyncio event loop.")

warnings.filterwarnings("ignore")

# ENHANCED LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from underlying libraries
logging.getLogger('aioice.ice').setLevel(logging.WARNING)
logging.getLogger('aiortc').setLevel(logging.WARNING)

# Global variables for models
uv_pipe = None
tts_model = None
vad_model = None
executor = ThreadPoolExecutor(max_workers=4)

def candidate_from_sdp(candidate_string: str) -> dict:
    if candidate_string.startswith("candidate:"):
        candidate_string = candidate_string[10:]
    bits = candidate_string.split()
    if len(bits) < 8:
        raise ValueError(f"Invalid candidate string: {candidate_string}")
    candidate_params = {
        'component': int(bits[1]), 'foundation': bits[0], 'ip': bits[4],
        'port': int(bits[5]), 'priority': int(bits[3]), 'protocol': bits[2], 'type': bits[7],
    }
    for i in range(8, len(bits) - 1, 2):
        if bits[i] == "raddr":
            candidate_params['relatedAddress'] = bits[i + 1]
        elif bits[i] == "rport":
            candidate_params['relatedPort'] = int(bits[i + 1])
        elif bits[i] == "tcptype":
            candidate_params['tcpType'] = bits[i + 1]
    return candidate_params

def parse_ultravox_response(result):
    try:
        if isinstance(result, str): return result
        elif isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, str): return item
            elif isinstance(item, dict) and 'generated_text' in item: return item['generated_text']
        logger.warning(f"Unexpected Ultravox response format: {type(result)}")
        return ""
    except Exception as e:
        logger.error(f"Error parsing Ultravox response: {e}")
        return ""

class SileroVAD:
    def __init__(self):
        try:
            logger.info("🎤 Loading Silero VAD model...")
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False
            )
            (self.get_speech_timestamps, _, _, _, _) = utils
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading Silero VAD: {e}")
            self.model = None

    def detect_speech(self, audio_tensor, sample_rate=16000):
        if self.model is None:
            logger.warning("⚠️ VAD model not available, assuming speech present")
            return True
        try:
            if isinstance(audio_tensor, np.ndarray): audio_tensor = torch.from_numpy(audio_tensor)
            if audio_tensor.dtype != torch.float32: audio_tensor = audio_tensor.float()
            if len(audio_tensor.shape) > 1: audio_tensor = audio_tensor.squeeze()
            max_val = audio_tensor.abs().max()
            if max_val > 1.0: audio_tensor = audio_tensor / max_val
            elif max_val == 0: return False
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, self.model, sampling_rate=sample_rate, threshold=0.3,
                min_speech_duration_ms=100, min_silence_duration_ms=250
            )
            has_speech = len(speech_timestamps) > 0
            logger.debug(f"🎯 VAD Result: {has_speech}")
            return has_speech
        except Exception as e:
            logger.error(f"❌ VAD error: {e}")
            return True

def initialize_models():
    global uv_pipe, tts_model, vad_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing models on device: {device}")
    logger.info("📥 Loading Silero VAD...")
    vad_model = SileroVAD()
    if vad_model.model is None: return False
    try:
        logger.info("📥 Loading Ultravox pipeline...")
        uv_pipe = pipeline("fixie-ai/ultravox-v0_4", trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
        logger.info("✅ Ultravox pipeline loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading Ultravox: {e}", exc_info=True)
        return False
    try:
        logger.info("📥 Loading Chatterbox TTS...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("✅ Chatterbox TTS loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading TTS: {e}", exc_info=True)
        return False
    logger.info("🎉 All models loaded successfully!")
    return True

class AudioBuffer:
    def __init__(self, max_duration=3.0, sample_rate=16000):
        self.sample_rate, self.max_samples = sample_rate, int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time, self.min_speech_samples, self.process_interval = time.time(), int(0.4 * sample_rate), 0.5
    def add_audio(self, audio_data):
        if isinstance(audio_data, np.ndarray): audio_data = audio_data.flatten()
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16: audio_data = audio_data.astype(np.float32) / 32768.0
            else: audio_data = audio_data.astype(np.float32)
        if np.abs(audio_data).max() > 1.0: audio_data = audio_data / np.abs(audio_data).max()
        self.buffer.extend(audio_data)
    def get_audio_array(self):
        return np.array(list(self.buffer), dtype=np.float32) if self.buffer else np.array([])
    def should_process(self):
        current_time = time.time()
        if len(self.buffer) >= self.min_speech_samples and (current_time - self.last_process_time) >= self.process_interval:
            audio_array = self.get_audio_array()
            if np.abs(audio_array).max() < 0.005:
                self.last_process_time = current_time
                return False
            if vad_model.detect_speech(audio_array, self.sample_rate):
                logger.info(f"🎯 Processing decision: YES. Audio len: {len(audio_array)}")
                self.last_process_time = current_time
                return True
        return False
    def reset(self):
        self.buffer.clear()
        logger.info("🔄 Audio buffer reset")

class OptimizedTTS:
    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip(): return np.zeros(1600, dtype=np.float32)
        logger.info(f"🎤 Synthesizing: '{text[:100]}...'")
        try:
            with torch.inference_mode():
                wav = tts_model.generate(text)
                if not isinstance(wav, torch.Tensor): wav = torch.from_numpy(wav)
                if wav.dim() == 1: wav = wav.unsqueeze(0)
                result = wav.cpu().numpy().flatten().astype(np.float32)
                if hasattr(tts_model, 'sample_rate') and tts_model.sample_rate != 48000:
                    result = librosa.resample(result, orig_sr=tts_model.sample_rate, target_sr=48000)
                logger.info(f"✅ TTS generated {len(result)} samples at 48kHz")
                return result
        except Exception as e:
            logger.error(f"❌ TTS synthesis error: {e}")
            return np.zeros(4800, dtype=np.float32)

class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self):
        super().__init__()
        self.tts = OptimizedTTS()
        self._response_queue = asyncio.Queue()
        self._current_response, self._response_position = None, 0
        self._timestamp, self._time_base = 0, fractions.Fraction(1, 48000)

    async def recv(self):
        frame_samples = 960
        frame = np.zeros(frame_samples, dtype=np.float32)
        if self._current_response is None or self._response_position >= len(self._current_response):
            try:
                self._current_response = await asyncio.wait_for(self._response_queue.get(), timeout=0.001)
                self._response_position = 0
                logger.info(f"🔊 Starting new response playback: {len(self._current_response)} samples")
            except asyncio.TimeoutError: pass
        if self._current_response is not None:
            end_pos = min(self._response_position + frame_samples, len(self._current_response))
            chunk = self._current_response[self._response_position:end_pos]
            frame[:len(chunk)] = chunk
            self._response_position = end_pos
        
        frame_int16 = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
        audio_frame = av.AudioFrame.from_ndarray(frame_int16.reshape(1, -1), format="s16", layout="mono")
        audio_frame.pts, audio_frame.time_base, audio_frame.sample_rate = self._timestamp, self._time_base, 48000
        self._timestamp += frame_samples
        return audio_frame

    async def queue_response_audio(self, audio_array):
        await self._response_queue.put(audio_array)

class AudioFrameProcessor:
    def __init__(self, audio_track_to_process: AudioStreamTrack):
        self.track, self.audio_buffer, self.output_audio_track, self.task = None, AudioBuffer(), audio_track_to_process, None
    def addTrack(self, track): self.track = track
    async def start(self): self.task = asyncio.create_task(self._run())
    async def stop(self):
        if self.task: self.task.cancel()
        logger.info("🛑 AudioFrameProcessor stopped.")

    async def _run(self):
        try:
            while True:
                try:
                    frame = await self.track.recv()
                # --- FIX 1: Use the correct exception from aiortc.mediastreams ---
                except mediastreams.MediaStreamError:
                    logger.warning("Media stream from client ended. Stopping processor.")
                    break
                
                audio_data = frame.to_ndarray()
                if len(audio_data.shape) > 1: audio_data = audio_data[0]
                if audio_data.dtype == np.int16: audio_data = audio_data.astype(np.float32) / 32768.0
                if frame.sample_rate != 16000: audio_data = librosa.resample(audio_data, orig_sr=frame.sample_rate, target_sr=16000)
                
                self.audio_buffer.add_audio(audio_data)
                if self.audio_buffer.should_process():
                    audio_for_processing = self.audio_buffer.get_audio_array()
                    self.audio_buffer.reset()
                    asyncio.create_task(self._process_speech(audio_for_processing))
        except asyncio.CancelledError: logger.info("Audio processing task cancelled.")
        except Exception as e: logger.error(f"❌ Unhandled error in AudioFrameProcessor: {e}", exc_info=True)

    async def _process_speech(self, audio_array):
        try:
            response_text = await self._generate_response(audio_array)
            if response_text and response_text.strip():
                response_audio = self.output_audio_track.tts.synthesize(response_text)
                await self.output_audio_track.queue_response_audio(response_audio)
                logger.info("🎵 Response audio queued for playback")
        except Exception as e: logger.error(f"❌ Error processing speech: {e}", exc_info=True)

    async def _generate_response(self, audio_array):
        try:
            turns = [{"role": "system", "content": "You are a friendly and helpful voice assistant. Keep responses concise."}]
            with torch.inference_mode():
                result = uv_pipe({'audio': audio_array, 'turns': turns, 'sampling_rate': 16000}, max_new_tokens=40)
            response_text = parse_ultravox_response(result).strip()
            logger.info(f"✅ Ultravox raw response: '{response_text}'")
            return response_text
        except Exception as e:
            logger.error(f"❌ Error in Ultravox generation: {e}", exc_info=True)
            return "I'm having trouble thinking."

class WebRTCConnection:
    def __init__(self):
        self.pc = RTCPeerConnection(RTCConfiguration([RTCIceServer(urls="stun:stun.l.google.com:19302")]))
        self.audio_track = AudioStreamTrack()
        self.frame_processor = None
        # --- FIX 3: Assign event handlers inside __init__ ---
        @self.pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            logger.info(f"🧊 ICE connection state: {self.pc.iceConnectionState}")
            if self.pc.iceConnectionState == "failed": await self.pc.close()
        @self.pc.on("track")
        def on_track(track):
            logger.info(f"🎧 Track received: {track.kind}")
            if track.kind == "audio":
                self.frame_processor = AudioFrameProcessor(self.audio_track)
                self.frame_processor.addTrack(track)
                asyncio.create_task(self.frame_processor.start())

    async def handle_offer(self, sdp, type):
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type))
        self.pc.addTrack(self.audio_track)
        recorder = MediaBlackhole()
        recorder.addTrack(self.audio_track)
        await recorder.start()
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription

    async def close(self):
        if self.frame_processor: await self.frame_processor.stop()
        if self.pc.connectionState != "closed": await self.pc.close()
        logger.info("WebRTC Connection closed.")

async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    connection = WebRTCConnection()
    logger.info(f"🔌 New WebSocket connection")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data["type"] == "offer":
                    local_description = await connection.handle_offer(sdp=data["sdp"], type=data["type"])
                    await ws.send_json({"type": "answer", "sdp": local_description.sdp})
                # --- FIX 2: Revert to your original, correct ICE candidate handling ---
                elif data["type"] == "ice-candidate" and "candidate" in data:
                    try:
                        candidate_string = data.get("candidate")
                        if candidate_string:
                            params = candidate_from_sdp(candidate_string)
                            candidate = RTCIceCandidate(
                                component=params['component'], foundation=params['foundation'],
                                ip=params['ip'], port=params['port'], priority=params['priority'],
                                protocol=params['protocol'], type=params['type'],
                                sdpMid=data.get("sdpMid"), sdpMLineIndex=data.get("sdpMLineIndex")
                            )
                            await connection.pc.addIceCandidate(candidate)
                            logger.info(f"✅ Added ICE candidate: {candidate.type}")
                    except Exception as e:
                        logger.error(f"❌ Error adding ICE candidate: {e}")
    finally:
        await connection.close()
        logger.info(f"🔌 Closed WebSocket connection.")
    return ws

HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat S2S Voice Assistant</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style> body { font-family: sans-serif; margin: 0; padding: 20px; background: #2c3e50; color: #ecf0f1; display: flex; align-items: center; justify-content: center; min-height: 100vh; } .container { background: #34495e; padding: 40px; border-radius: 10px; box-shadow: 0 10px 20px rgba(0,0,0,0.2); text-align: center; max-width: 600px; width: 100%; } h1 { margin-bottom: 30px; } button { background: #2ecc71; color: white; border: none; padding: 15px 30px; font-size: 18px; border-radius: 5px; cursor: pointer; margin: 10px; transition: background 0.3s; } button:hover { background: #27ae60; } button:disabled { background: #95a5a6; cursor: not-allowed; } .stop-btn { background: #e74c3c; } .stop-btn:hover { background: #c0392b; } .status { margin: 20px 0; padding: 15px; border-radius: 5px; font-weight: 500; } .status.connected { background: #27ae60; } .status.disconnected { background: #c0392b; } .status.connecting { background: #f39c12; } .conversation-display { background: #2c3e50; border: 1px solid #7f8c8d; border-radius: 5px; padding: 20px; margin: 20px 0; min-height: 200px; text-align: left; max-height: 400px; overflow-y: auto; } .message { margin: 10px 0; padding: 10px; border-radius: 8px; } .user-message { background: #3498db; } .ai-message { background: #8e44ad; } </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ UltraChat Voice Assistant</h1>
        <div class="controls">
            <button id="startBtn" onclick="startConversation()">START</button>
            <button id="stopBtn" onclick="stopConversation()" class="stop-btn" disabled>STOP</button>
        </div>
        <div id="status" class="status disconnected">🔌 Disconnected</div>
        <div class="conversation-display" id="conversationDisplay"></div>
        <audio id="remoteAudio" autoplay playsinline></audio>
    </div>
<script>
    let pc, ws, localStream, audioContext;
    const remoteAudio = document.getElementById('remoteAudio');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDiv = document.getElementById('status');
    const conversationDisplay = document.getElementById('conversationDisplay');

    function updateStatus(message, className) { statusDiv.textContent = message; statusDiv.className = `status ${className}`; }
    function addMessage(text, type) { conversationDisplay.innerHTML += `<div class="message ${type}-message">${text}</div>`; conversationDisplay.scrollTop = conversationDisplay.scrollHeight; }

    async function startConversation() {
        startBtn.disabled = true;
        updateStatus('🔄 Connecting...', 'connecting');
        try {
            if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (audioContext.state === 'suspended') await audioContext.resume();
            
            localStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true } });
            addMessage("<strong>You:</strong> 🎤 Mic access granted.", 'user');

            pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

            pc.ontrack = e => { if (remoteAudio.srcObject !== e.streams[0]) { remoteAudio.srcObject = e.streams[0]; remoteAudio.play().catch(err => console.error("Autoplay failed:", err));} addMessage("<strong>AI:</strong> 🔊 Audio stream received.", 'ai'); };
            pc.onicecandidate = e => e.candidate && ws && ws.send(JSON.stringify({ type: 'ice-candidate', candidate: e.candidate.candidate, sdpMid: e.candidate.sdpMid, sdpMLineIndex: e.candidate.sdpMLineIndex}));

            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onopen = async () => {
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp }));
            };

            ws.onmessage = async e => {
                const data = JSON.parse(e.data);
                if (data.type === 'answer') {
                    await pc.setRemoteDescription(new RTCSessionDescription(data));
                    updateStatus('✅ Connected', 'connected');
                    stopBtn.disabled = false;
                    addMessage("<strong>System:</strong> 🤝 Connection established.", 'system');
                }
            };

            ws.onclose = () => stopConversation();
            ws.onerror = () => stopConversation();
        } catch (err) {
            updateStatus(`❌ Error: ${err.message}`, 'disconnected');
            stopConversation();
        }
    }

    function stopConversation() {
        if (ws) ws.close();
        if (pc) pc.close();
        if (localStream) localStream.getTracks().forEach(track => track.stop());
        updateStatus('🔌 Disconnected', 'disconnected');
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
    window.addEventListener('beforeunload', stopConversation);
</script>
</body>
</html>
"""

async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def main():
    print("🚀 UltraChat S2S - DEFINITIVE FIX - Starting server...")
    if not initialize_models():
        logger.error("❌ Failed to initialize models. Exiting.")
        return
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    print("✅ Server started successfully!")
    print("🌐 Server running on http://0.0.0.0:7860")
    print("🔒 Next step: In another terminal, run 'cloudflared' to create a secure tunnel.")
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
