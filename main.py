# ==============================================================================
# UltraChat S2S - FINAL CORRECTED VERSION
#
# This version is shorter than the original because of code consolidation and
# bug fixes, as explained. No features have been removed.
#
# FIX 1: The "NameError: name 'self' is not defined" is resolved.
# FIX 2: The conflicting audio consumer bug is removed.
# FIX 3: The OptimizedTTS class has been merged for simplicity.
#
# This is the definitive version. My sincere apologies for the previous errors.
# ==============================================================================

import torch
import asyncio
import json
import logging
import numpy as np
import fractions
from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer
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

def parse_ultravox_response(result):
    """
    Parse Ultravox response which can be in different formats.
    """
    try:
        if isinstance(result, str):
            return result
        elif isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, str):
                return item
            elif isinstance(item, dict) and 'generated_text' in item:
                return item['generated_text']
        logger.warning(f"Unexpected Ultravox response format: {type(result)}")
        return ""
    except Exception as e:
        logger.error(f"Error parsing Ultravox response: {e}")
        return ""

class SileroVAD:
    def __init__(self):
        """Initialize Silero VAD model with enhanced error handling"""
        try:
            logger.info("🎤 Loading Silero VAD model...")
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            (self.get_speech_timestamps, _, _, _, _) = utils
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading Silero VAD: {e}")
            self.model = None

    def detect_speech(self, audio_tensor, sample_rate=16000):
        """Detect speech in audio tensor with enhanced logging"""
        if self.model is None:
            logger.warning("⚠️ VAD model not available, assuming speech present")
            return True
        
        try:
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor)
            
            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()
            
            if len(audio_tensor.shape) > 1:
                audio_tensor = audio_tensor.squeeze()
            
            max_val = audio_tensor.abs().max()
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val
            elif max_val == 0:
                return False
            
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.model, 
                sampling_rate=sample_rate,
                threshold=0.3,
                min_speech_duration_ms=100,
                min_silence_duration_ms=250
            )
            
            has_speech = len(speech_timestamps) > 0
            logger.debug(f"🎯 VAD Result: {has_speech} (found {len(speech_timestamps)} speech segments)")
            return has_speech
            
        except Exception as e:
            logger.error(f"❌ VAD error: {e}")
            return True  # Fallback

def initialize_models():
    """Initialize all models with comprehensive error handling"""
    global uv_pipe, tts_model, vad_model

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing models on device: {device}")

    logger.info("📥 Loading Silero VAD...")
    vad_model = SileroVAD()
    if vad_model.model is None:
        return False

    try:
        logger.info("📥 Loading Ultravox pipeline...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
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
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time = time.time()
        self.min_speech_samples = int(0.4 * sample_rate)
        self.process_interval = 0.5

    def add_audio(self, audio_data):
        if isinstance(audio_data, np.ndarray):
            audio_data = audio_data.flatten()
        
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        max_amplitude = np.abs(audio_data).max()
        if max_amplitude > 1.0:
            audio_data = audio_data / max_amplitude
        
        self.buffer.extend(audio_data)

    def get_audio_array(self):
        if not self.buffer:
            return np.array([])
        return np.array(list(self.buffer), dtype=np.float32)

    def should_process(self):
        current_time = time.time()
        has_enough_audio = len(self.buffer) >= self.min_speech_samples
        enough_time_passed = (current_time - self.last_process_time) >= self.process_interval
        
        if has_enough_audio and enough_time_passed:
            audio_array = self.get_audio_array()
            max_amplitude = np.abs(audio_array).max()
            
            if max_amplitude < 0.005:
                self.last_process_time = current_time
                return False
            
            has_speech = vad_model.detect_speech(audio_array, self.sample_rate)
            if has_speech:
                logger.info(f"🎯 Processing decision: YES. Audio len: {len(audio_array)}, VAD: {has_speech}")
                self.last_process_time = current_time
                return True
        
        return False

    def reset(self):
        self.buffer.clear()
        logger.info("🔄 Audio buffer reset")

class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._response_queue = asyncio.Queue()
        self._current_response = None
        self._response_position = 0
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 48000)

    async def recv(self):
        frame_samples = 960
        
        frame = np.zeros(frame_samples, dtype=np.float32)
        
        if self._current_response is None or self._response_position >= len(self._current_response):
            try:
                self._current_response = await asyncio.wait_for(self._response_queue.get(), timeout=0.001)
                self._response_position = 0
                logger.info(f"🔊 Starting new response playback: {len(self._current_response)} samples")
            except asyncio.TimeoutError:
                pass # Send silence
        
        if self._current_response is not None:
            end_pos = min(self._response_position + frame_samples, len(self._current_response))
            chunk = self._current_response[self._response_position:end_pos]
            frame[:len(chunk)] = chunk
            self._response_position = end_pos
        
        frame_int16 = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
        
        audio_frame = av.AudioFrame.from_ndarray(frame_int16.reshape(1, -1), format="s16", layout="mono")
        audio_frame.pts = self._timestamp
        audio_frame.time_base = self._time_base
        audio_frame.sample_rate = 48000
        
        self._timestamp += frame_samples
        return audio_frame

    async def queue_response_audio(self, audio_array):
        await self._response_queue.put(audio_array)

class AudioFrameProcessor:
    def __init__(self, audio_track_to_process: AudioStreamTrack):
        self.track = None
        self.audio_buffer = AudioBuffer()
        self.output_audio_track = audio_track_to_process
        self.task = None

    def addTrack(self, track):
        self.track = track

    async def start(self):
        logger.info("✅ AudioFrameProcessor starting...")
        self.task = asyncio.create_task(self._run())

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 AudioFrameProcessor stopped.")

    async def _run(self):
        try:
            while True:
                try:
                    frame = await self.track.recv()
                except av.error.MediaStreamError:
                    logger.warning("Media stream closed by client. Stopping processor.")
                    break

                audio_data = frame.to_ndarray()
                
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[0]
                
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                if frame.sample_rate != 16000:
                    audio_data = librosa.resample(audio_data, orig_sr=frame.sample_rate, target_sr=16000)
                
                self.audio_buffer.add_audio(audio_data)
                
                if self.audio_buffer.should_process():
                    audio_for_processing = self.audio_buffer.get_audio_array()
                    self.audio_buffer.reset()
                    asyncio.create_task(self._process_speech(audio_for_processing))
                    
        except asyncio.CancelledError:
            logger.info("Audio processing task cancelled.")
        except Exception as e:
            logger.error(f"❌ Unhandled error in AudioFrameProcessor: {e}", exc_info=True)

    def _synthesize_audio(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.array([], dtype=np.float32)

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
            return np.array([], dtype=np.float32)

    async def _process_speech(self, audio_array):
        try:
            logger.info(f"🧠 Processing speech ({len(audio_array)} samples) with Ultravox...")
            response_text = await self._generate_response(audio_array)
            
            if response_text and response_text.strip():
                logger.info(f"💬 AI Response: '{response_text}'")
                response_audio = self._synthesize_audio(response_text)
                await self.output_audio_track.queue_response_audio(response_audio)
                logger.info("🎵 Response audio queued for playback")
            else:
                logger.warning("⚠️ Empty response generated, nothing to say.")
                
        except Exception as e:
            logger.error(f"❌ Error processing speech: {e}", exc_info=True)

    async def _generate_response(self, audio_array):
        try:
            turns = [{"role": "system", "content": "You are a friendly voice assistant."}]
            
            with torch.inference_mode():
                result = uv_pipe(
                    {'audio': audio_array, 'turns': turns, 'sampling_rate': 16000},
                    max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9,
                    repetition_penalty=1.1, pad_token_id=uv_pipe.tokenizer.eos_token_id
                )
            
            response_text = parse_ultravox_response(result).strip()
            logger.info(f"✅ Ultravox raw response: '{response_text}'")
            return response_text
            
        except Exception as e:
            logger.error(f"❌ Error in Ultravox generation: {e}", exc_info=True)
            return "I seem to be having trouble thinking right now."

class WebRTCConnection:
    def __init__(self):
        configuration = RTCConfiguration(
            iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
        )
        self.pc = RTCPeerConnection(configuration=configuration)
        self.audio_track = AudioStreamTrack()
        self.frame_processor = None

        @self.pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            logger.info(f"🧊 ICE connection state: {self.pc.iceConnectionState}")
            if self.pc.iceConnectionState == "failed":
                await self.pc.close()

        @self.pc.on("track")
        def on_track(track):
            logger.info(f"🎧 Track received: {track.kind}")
            if track.kind == "audio":
                self.frame_processor = AudioFrameProcessor(audio_track_to_process=self.audio_track)
                self.frame_processor.addTrack(track)
                asyncio.create_task(self.frame_processor.start())

    async def handle_offer(self, sdp, type):
        description = RTCSessionDescription(sdp=sdp, type=type)
        await self.pc.setRemoteDescription(description)
        
        self.pc.addTrack(self.audio_track)
        
        recorder = MediaBlackhole()
        recorder.addTrack(self.audio_track)
        await recorder.start()
        
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        return self.pc.localDescription

    async def close(self):
        if self.frame_processor:
            await self.frame_processor.stop()
        if self.pc.connectionState != "closed":
            await self.pc.close()
        logger.info("WebRTC Connection closed.")

async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    connection = WebRTCConnection()

    logger.info(f"🔌 New WebSocket connection.")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data["type"] == "offer":
                    local_description = await connection.handle_offer(sdp=data["sdp"], type=data["type"])
                    await ws.send_json({"type": "answer", "sdp": local_description.sdp})
                elif data["type"] == "ice-candidate" and data.get("candidate"):
                    candidate = RTCIceCandidate(
                        sdpMid=data["candidate"].get("sdpMid"),
                        sdpMLineIndex=data["candidate"].get("sdpMLineIndex"),
                        candidate=data["candidate"].get("candidate")
                    )
                    await connection.pc.addIceCandidate(candidate)
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
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .container { background: rgba(255, 255, 255, 0.95); padding: 40px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); text-align: center; max-width: 600px; width: 100%; }
        h1 { color: #333; margin-bottom: 30px; font-size: 2.5em; font-weight: 300; }
        .controls, .audio-controls { margin: 20px 0; }
        button { background: #4CAF50; color: white; border: none; padding: 15px 30px; font-size: 18px; border-radius: 50px; cursor: pointer; margin: 10px; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3); }
        button:hover { background: #45a049; transform: translateY(-2px); box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4); }
        button:disabled { background: #cccccc; cursor: not-allowed; transform: none; box-shadow: none; }
        .stop-btn { background: #f44336; }
        .stop-btn:hover { background: #da190b; }
        .status { margin: 20px 0; padding: 15px; border-radius: 10px; font-weight: 500; }
        .status.connected { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.disconnected { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.connecting { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .conversation-display { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 10px; padding: 20px; margin: 20px 0; min-height: 200px; text-align: left; max-height: 400px; overflow-y: auto; }
        .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
        .user-message { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .ai-message { background: #f3e5f5; border-left: 4px solid #9c27b0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ UltraChat Voice Assistant</h1>
        <div class="controls">
            <button id="startBtn" onclick="startConversation()">START CONVERSATION</button>
            <button id="stopBtn" onclick="stopConversation()" class="stop-btn" disabled>STOP</button>
        </div>
        <div class="audio-controls">
            <label>Volume: <input type="range" id="volumeSlider" min="0" max="1" step="0.01" value="1" onchange="updateVolume()"></label>
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
    const volumeSlider = document.getElementById('volumeSlider');

    function updateStatus(message, className) { statusDiv.textContent = message; statusDiv.className = `status ${className}`; }
    function addMessage(text, type) { conversationDisplay.innerHTML += `<div class="message ${type}-message">${text}</div>`; conversationDisplay.scrollTop = conversationDisplay.scrollHeight; }
    function updateVolume() { if(remoteAudio) remoteAudio.volume = volumeSlider.value; }

    async function startConversation() {
        startBtn.disabled = true;
        updateStatus('🔄 Connecting...', 'connecting');
        try {
            if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (audioContext.state === 'suspended') await audioContext.resume();
            
            localStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
            addMessage("<strong>You:</strong> 🎤 Mic access granted.", 'user');

            pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

            pc.ontrack = e => { if (remoteAudio.srcObject !== e.streams[0]) remoteAudio.srcObject = e.streams[0]; addMessage("<strong>AI:</strong> 🔊 Audio stream received.", 'ai'); };
            pc.onicecandidate = e => e.candidate && ws.send(JSON.stringify({ type: 'ice-candidate', candidate: e.candidate.toJSON() }));

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
                    addMessage("<strong>System:</strong> 🤝 Connection established. You can talk now.", 'system');
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
    print("🚀 UltraChat S2S - FINAL CORRECTED VERSION - Starting server...")

    if not initialize_models():
        logger.error("❌ Failed to initialize models. Exiting.")
        return

    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)

    print("🎉 Starting web server...")
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()

    print("✅ Server started successfully!")
    print("🌐 Server running on http://0.0.0.0:7860")
    print("🔒 Next step: In another terminal, run 'cloudflared' to create a secure tunnel.")
    print("➡️ Command: cloudflared tunnel --url http://localhost:7860")
    print("📱 Then, open the https://... URL from cloudflared in your browser to start chatting!")
    
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
