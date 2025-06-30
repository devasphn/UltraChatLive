# --- START OF FILE main.py ---

import torch
import asyncio
import json
import logging
import numpy as np
import tempfile
import torchaudio
import websockets
import threading
import ssl
import os
import fractions
from aiohttp import web, web_runner, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
import av  # PyAV for AudioFrame
from aiortc.contrib.media import MediaRecorder, MediaRelay, MediaBlackhole
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
                threshold=0.3, # Slightly higher threshold to reduce noise triggers
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
    if vad_model.model is None: return False
    
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
        logger.error(f"❌ Error loading Ultravox: {e}")
        return False
    
    try:
        logger.info("📥 Loading Chatterbox TTS...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("✅ Chatterbox TTS loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading TTS: {e}")
        return False
    
    logger.info("🎉 All models loaded successfully!")
    return True

class AudioBuffer:
    def __init__(self, max_duration=3.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time = time.time()
        self.min_speech_samples = int(0.4 * sample_rate)  # Reduced minimum samples
        self.process_interval = 0.5  # Process more frequently
        
    def add_audio(self, audio_data):
        if isinstance(audio_data, np.ndarray):
            audio_data = audio_data.flatten()
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16: audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32: audio_data = audio_data.astype(np.float32) / 2147483648.0
            else: audio_data = audio_data.astype(np.float32)
        
        max_amplitude = np.abs(audio_data).max()
        if max_amplitude > 1.0: audio_data = audio_data / max_amplitude
        
        self.buffer.extend(audio_data)
        
    def get_audio_array(self):
        if not self.buffer: return np.array([])
        return np.array(list(self.buffer), dtype=np.float32)

    def should_process(self):
        current_time = time.time()
        has_enough_audio = len(self.buffer) >= self.min_speech_samples
        enough_time_passed = (current_time - self.last_process_time) >= self.process_interval
        
        if has_enough_audio and enough_time_passed:
            audio_array = self.get_audio_array()
            max_amplitude = np.abs(audio_array).max()
            if max_amplitude < 0.005:  # Silence threshold
                logger.debug(f"🔇 Audio too quiet: max amplitude {max_amplitude:.6f}")
                self.last_process_time = current_time # Reset timer even if quiet
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

class OptimizedTTS:
    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(1600, dtype=np.float32)
        logger.info(f"🎤 Synthesizing: '{text[:100]}...'")
        try:
            with torch.inference_mode():
                wav = tts_model.generate(text)
            if not isinstance(wav, torch.Tensor): wav = torch.from_numpy(wav)
            if wav.dim() == 1: wav = wav.unsqueeze(0)
            result = wav.cpu().numpy().flatten().astype(np.float32)
            if hasattr(tts_model, 'sample_rate') and tts_model.sample_rate != 16000:
                result = librosa.resample(result, orig_sr=tts_model.sample_rate, target_sr=16000)
            logger.info(f"✅ TTS generated {len(result)} samples")
            return result
        except Exception as e:
            logger.error(f"❌ TTS synthesis error: {e}")
            return np.zeros(1600, dtype=np.float32)

class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.tts = OptimizedTTS()
        self._response_queue = asyncio.Queue()
        self._current_response = None
        self._response_position = 0
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 16000)
        self._consumer_task = None
        self._is_running = False
        
    def start_consumer(self):
        if self._consumer_task is None:
            self._is_running = True
            self._consumer_task = asyncio.create_task(self._consume_audio())
            logger.info("🎧 Audio output stream consumer started")
        
    async def _consume_audio(self):
        try:
            while self._is_running:
                await self.recv()
                await asyncio.sleep(0.01) # Fine-tune sleep for smoother audio
        except asyncio.CancelledError:
            logger.info("🛑 Audio output stream consumer stopped")
        except Exception as e:
            logger.error(f"❌ Audio output consumer error: {e}")
        
    async def recv(self):
        frame_samples = 320  # 20ms at 16kHz
        if self._current_response is None or self._response_position >= len(self._current_response):
            try:
                self._current_response = await asyncio.wait_for(self._response_queue.get(), timeout=0.001)
                self._response_position = 0
                logger.info(f"🔊 Starting new response playback: {len(self._current_response)} samples")
            except asyncio.TimeoutError:
                frame = np.zeros(frame_samples, dtype=np.float32)
        
        if self._current_response is not None:
            end_pos = min(self._response_position + frame_samples, len(self._current_response))
            frame = self._current_response[self._response_position:end_pos]
            self._response_position = end_pos
            if len(frame) < frame_samples:
                frame = np.pad(frame, (0, frame_samples - len(frame)), 'constant')
        else:
            frame = np.zeros(frame_samples, dtype=np.float32)

        frame_2d = frame.reshape(1, -1)
        audio_frame = av.AudioFrame.from_ndarray(frame_2d, format="flt", layout="mono")
        audio_frame.pts = self._timestamp
        audio_frame.time_base = self._time_base
        audio_frame.sample_rate = 16000
        self._timestamp += frame_samples
        return audio_frame

    async def queue_response_audio(self, audio_array):
        await self._response_queue.put(audio_array)

    def stop_consumer(self):
        self._is_running = False
        if self._consumer_task: self._consumer_task.cancel()


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
                frame = await self.track.recv()
                audio_data = frame.to_ndarray()
                
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[0] # Take the first channel
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                self.audio_buffer.add_audio(audio_data)

                if self.audio_buffer.should_process():
                    audio_for_processing = self.audio_buffer.get_audio_array()
                    self.audio_buffer.reset()
                    asyncio.create_task(self._process_speech(audio_for_processing))
        except asyncio.CancelledError:
            logger.info("Audio processing task cancelled.")
        except Exception as e:
            logger.error(f"❌ Unhandled error in AudioFrameProcessor: {e}", exc_info=True)

    async def _process_speech(self, audio_array):
        try:
            logger.info(f"🧠 Processing speech ({len(audio_array)} samples) with Ultravox...")
            response_text = await self._generate_response(audio_array)
            if response_text and response_text.strip():
                logger.info(f"💬 AI Response: '{response_text}'")
                response_audio = self.output_audio_track.tts.synthesize(response_text)
                await self.output_audio_track.queue_response_audio(response_audio)
                logger.info("🎵 Response audio queued for playback")
            else:
                logger.warning("⚠️ Empty response generated, nothing to say.")
        except Exception as e:
            logger.error(f"❌ Error processing speech: {e}", exc_info=True)

    async def _generate_response(self, audio_array):
        try:
            turns = [{
                "role": "system",
                "content": "You are a friendly and helpful voice assistant. Keep your responses concise, natural, and conversational, ideally under 20 words."
            }]
            with torch.inference_mode():
                result = uv_pipe({
                    'audio': audio_array,
                    'turns': turns,
                    'sampling_rate': 16000
                },
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=uv_pipe.tokenizer.eos_token_id
                )
            
            response_text = result[0]["generated_text"].strip()
            logger.info(f"✅ Ultravox raw response: '{response_text}'")
            return response_text
        except Exception as e:
            logger.error(f"❌ Error in Ultravox generation: {e}", exc_info=True)
            return "I seem to be having trouble thinking right now."


class WebRTCConnection:
    def __init__(self):
        # Updated RTCPeerConnection configuration with a testing TURN server
        self.pc = RTCPeerConnection(configuration={
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"},
                # Using a common testing TURN server. If this doesn't work,
                # you will likely need to set up your own TURN server.
                {"urls": "turn:numbers.webrtc.org:19302"} 
            ]
        })
        self.audio_track = AudioStreamTrack() # This is the track we SEND to the client
        self.frame_processor = None # This will handle the track we RECEIVE from the client
        
        @self.pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            logger.info(f"🧊 ICE connection state: {self.pc.iceConnectionState}")
            if self.pc.iceConnectionState == "failed":
                logger.error("ICE connection failed. Attempting to close PeerConnection.")
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
        self.audio_track.start_consumer()
        recorder = MediaBlackhole()
        recorder.addTrack(self.audio_track)
        await recorder.start()
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription
    
    async def close(self):
        if self.frame_processor:
            await self.frame_processor.stop()
        if self.audio_track:
            self.audio_track.stop_consumer()
        if self.pc.connectionState != "closed":
            await self.pc.close()
        logger.info("WebRTC Connection closed.")


# WebSocket handler
async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    
    connection_id = id(ws)
    connection = WebRTCConnection()
    logger.info(f"🔌 New WebSocket connection: {connection_id}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data["type"] == "offer":
                        logger.info("📨 Received WebRTC offer")
                        local_description = await connection.handle_offer(sdp=data["sdp"], type=data["type"])
                        await ws.send_str(json.dumps({
                            "type": "answer",
                            "sdp": local_description.sdp
                        }))
                        logger.info("📤 Sent WebRTC answer")
                    
                    elif data["type"] == "ice-candidate" and "candidate" in data:
                        try:
                            candidate_string = data.get("candidate")
                            sdp_mid = data.get("sdpMid")
                            sdp_mline_index = data.get("sdpMLineIndex")

                            if candidate_string and sdp_mid is not None and sdp_mline_index is not None:
                                candidate = RTCIceCandidate(
                                    candidate_string,
                                    sdp_mid=sdp_mid,
                                    sdp_mline_index=sdp_mline_index,
                                )
                                await connection.pc.addIceCandidate(candidate)
                                logger.debug(f"✅ Added ICE candidate: {candidate_string[:30]}...")
                            else:
                                logger.warning(f"Received incomplete ICE candidate data: {data}")

                        except Exception as e:
                            logger.error(f"❌ Error adding ICE candidate. Data: {data}. Error: {e}", exc_info=True)
                except json.JSONDecodeError:
                    logger.error("❌ Received invalid JSON message.")
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}", exc_info=True)

            elif msg.type == WSMsgType.ERROR:
                logger.error(f'❌ WebSocket error: {ws.exception()}')
    except Exception as e:
        logger.error(f"❌ WebSocket handler error: {e}", exc_info=True)
    finally:
        await connection.close()
        logger.info(f"🔌 Closed WebSocket connection: {connection_id}")
    return ws


HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat S2S - FINAL FIX</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; }
        .container { background: rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 30px; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); }
        h1 { text-align: center; margin-bottom: 30px; font-size: 2.5em; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); }
        .controls { display: flex; justify-content: center; gap: 20px; margin: 30px 0; flex-wrap: wrap; }
        button { padding: 15px 30px; font-size: 16px; border: none; border-radius: 50px; cursor: pointer; transition: all 0.3s ease; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
        .start-btn { background: linear-gradient(45deg, #4CAF50, #45a049); color: white; }
        .stop-btn { background: linear-gradient(45deg, #f44336, #da190b); color: white; }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .status { text-align: center; font-size: 18px; margin: 20px 0; padding: 15px; border-radius: 10px; background: rgba(255, 255, 255, 0.2); }
        .status.connected { background: rgba(76, 175, 80, 0.3); }
        .status.error { background: rgba(244, 67, 54, 0.3); }
        .audio-visualizer { height: 100px; background: rgba(255, 255, 255, 0.1); border-radius: 10px; margin: 20px 0; display: flex; align-items: center; justify-content: center; font-size: 14px; }
        .debug-info { background: rgba(0, 0, 0, 0.3); border-radius: 10px; padding: 15px; margin: 20px 0; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto; }
        .connection-info { background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 15px; margin: 20px 0; font-family: monospace; font-size: 12px; }
        .fix-note { background: rgba(76, 175, 80, 0.2); border-left: 4px solid #4CAF50; padding: 15px; margin: 20px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Real-time S2S AI Chat - DEFINITIVE FIX</h1>
        <div class="fix-note">
            <strong>✅ Definitive Fix Applied:</strong> This version correctly configures ICE servers (STUN/TURN) to ensure robust WebRTC connection establishment.
        </div>
        <div class="controls">
            <button id="startBtn" class="start-btn">Start Conversation</button>
            <button id="stopBtn" class="stop-btn" disabled>Stop Conversation</button>
        </div>
        <div id="status" class="status">Ready - Click Start to begin!</div>
        <div class="audio-visualizer" id="visualizer">Waiting to start...</div>
        <div class="connection-info" id="connectionInfo">Connection Status: Not connected</div>
        <div class="debug-info" id="debugInfo">Debug info will appear here...</div>
    </div>

    <script>
        let websocket = null;
        let peerConnection = null;
        let localStream = null;
        let audioContext = null;
        let analyser = null;
        let connectionStartTime = null;
        
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');
        const visualizer = document.getElementById('visualizer');
        const debugInfo = document.getElementById('debugInfo');
        const connectionInfo = document.getElementById('connectionInfo');
        
        startBtn.addEventListener('click', startConversation);
        stopBtn.addEventListener('click', stopConversation);
        
        function addDebugMessage(message) {
            const timestamp = new Date().toLocaleTimeString();
            debugInfo.innerHTML += `[${timestamp}] ${message}<br>`;
            debugInfo.scrollTop = debugInfo.scrollHeight;
            console.log(message); // Also log to browser console for easier inspection
        }
        
        function updateConnectionInfo(info) {
            connectionInfo.textContent = info;
        }
        
        function getWebSocketUrl() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            return `${protocol}//${host}/ws`;
        }
        
        async function startConversation() {
            addDebugMessage('🚀 startConversation() called');
            try {
                connectionStartTime = Date.now();
                updateStatus('🔄 Connecting...', '');
                addDebugMessage('🚶 Attempting to get user media...');
                
                localStream = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, sampleRate: 16000, channelCount: 1, latency: 0.02 }
                });
                
                addDebugMessage('✅ Microphone access granted successfully');
                setupAudioVisualization();
                
                addDebugMessage('🔗 Establishing WebSocket connection...');
                const wsUrl = getWebSocketUrl();
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = () => {
                    updateStatus('✅ WebSocket Connected', 'connected');
                    updateConnectionInfo('WebSocket: Connected');
                    addDebugMessage('✅ WebSocket onopen event fired');
                    setupWebRTC(); // Proceed to WebRTC setup only after WebSocket is open
                };
                
                websocket.onmessage = handleSignalingMessage;
                
                websocket.onerror = (error) => {
                    updateStatus('❌ WebSocket Error', 'error');
                    updateConnectionInfo('WebSocket: Error');
                    addDebugMessage(`❌ WebSocket onerror event: ${error}`);
                };
                
                websocket.onclose = (event) => {
                    updateStatus(`🔌 WebSocket Closed (${event.code})`, 'error');
                    updateConnectionInfo(`WebSocket: Closed (${event.code})`);
                    addDebugMessage(`🔌 WebSocket onclose event: Code=${event.code}, Reason=${event.reason}`);
                    stopConversation(false); // Clean up but don't try to close WebSocket again
                };
                
                startBtn.disabled = true;
                stopBtn.disabled = false;
                
            } catch (error) {
                updateStatus('❌ Error starting: ' + error.message, 'error');
                addDebugMessage(`❌ Error in startConversation(): ${error.message}`);
                console.error("Full error in startConversation:", error); // Log full error to browser console
                // Ensure buttons are reset if an error occurs early
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        }
        
        async function setupWebRTC() {
            addDebugMessage('🔧 setupWebRTC() called');
            try {
                peerConnection = new RTCPeerConnection({
                    iceServers: [ 
                        { urls: 'stun:stun.l.google.com:19302' }, 
                        { urls: 'stun:stun1.l.google.com:19302' },
                        // Using a common testing TURN server. If this doesn't work,
                        // you will likely need to set up your own TURN server.
                        { urls: 'turn:numbers.webrtc.org:19302' } 
                    ]
                });
                addDebugMessage('✅ RTCPeerConnection created');
                
                localStream.getTracks().forEach(track => {
                    peerConnection.addTrack(track, localStream);
                    addDebugMessage(`📡 Added ${track.kind} track to PeerConnection`);
                });
                
                peerConnection.ontrack = (event) => {
                    addDebugMessage(`🎵 Received ${event.track.kind} track from server`);
                    const [remoteStream] = event.streams;
                    playAudio(remoteStream);
                };
                
                peerConnection.onicecandidate = (event) => {
                    addDebugMessage('Candidate event fired');
                    if (event.candidate && websocket && websocket.readyState === WebSocket.OPEN) {
                        // Send a flat JSON object that matches what the Python server expects.
                        websocket.send(JSON.stringify({
                            type: 'ice-candidate',
                            candidate: event.candidate.candidate,
                            sdpMid: event.candidate.sdpMid,
                            sdpMLineIndex: event.candidate.sdpMLineIndex,
                        }));
                        addDebugMessage(`📤 Sent ICE candidate: ${event.candidate.type}`);
                    } else {
                         addDebugMessage('No candidate or WebSocket not open, skipping send.');
                    }
                };
                
                peerConnection.oniceconnectionstatechange = () => {
                    const state = peerConnection.iceConnectionState;
                    updateConnectionInfo(`ICE: ${state}`);
                    addDebugMessage(`🧊 ICE connection state changed: ${state}`);
                    
                    if (state === 'connected' || state === 'completed') {
                        const duration = Date.now() - connectionStartTime;
                        updateStatus('✅ Connected - Start speaking!', 'connected');
                        addDebugMessage(`✅ Connection established in ${duration}ms`);
                    } else if (state === 'failed' || state === 'closed' || state === 'disconnected') {
                        updateStatus('❌ Connection failed or closed', 'error');
                        addDebugMessage(`❌ ICE connection state is now ${state}`);
                        // If ICE fails, it's often a network/TURN server issue.
                        // Attempt to stop gracefully if it fails.
                        if (state !== 'closed') { // Avoid double-closing
                           stopConversation(false); // Don't close WS again if it already closed.
                        }
                    }
                };
                
                peerConnection.onconnectionstatechange = () => {
                    addDebugMessage(`🔗 PeerConnection state changed: ${peerConnection.connectionState}`);
                }
                
                addDebugMessage('➡️ Creating WebRTC offer...');
                const offer = await peerConnection.createOffer();
                addDebugMessage('✅ Offer created');
                await peerConnection.setLocalDescription(offer);
                addDebugMessage('✅ Local description set');
                
                addDebugMessage('📤 Sending WebRTC offer via WebSocket...');
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'offer',
                        sdp: offer.sdp
                    }));
                    addDebugMessage('✅ WebRTC offer sent');
                } else {
                    addDebugMessage('❌ WebSocket not open, cannot send offer.');
                }
                
            } catch (error) {
                updateStatus('❌ WebRTC setup error: ' + error.message, 'error');
                addDebugMessage(`❌ Error in setupWebRTC(): ${error.message}`);
                console.error("Full error in setupWebRTC:", error); // Log full error to browser console
            }
        }
        
        async function handleSignalingMessage(event) {
            addDebugMessage('📥 handleSignalingMessage() called');
            try {
                const message = JSON.parse(event.data);
                addDebugMessage(`📥 Received signaling message: ${message.type}`);
                
                if (!peerConnection) {
                    addDebugMessage('❌ PeerConnection not initialized, cannot process message.');
                    return;
                }

                if (message.type === 'answer') {
                    addDebugMessage('✅ Processing incoming answer');
                    await peerConnection.setRemoteDescription(new RTCSessionDescription(message));
                    addDebugMessage('✅ Remote description set from answer');
                } else if (message.type === 'ice-candidate') {
                    // Client only sends candidates, server processes them.
                    // We don't expect 'ice-candidate' messages from the server in this setup.
                    addDebugMessage('ℹ️ Received ICE candidate message from server (unexpected in this flow).');
                }
            } catch (error) {
                addDebugMessage(`❌ Error handling signaling message: ${error.message}`);
                console.error("Full error in handleSignalingMessage:", error); // Log full error to browser console
            }
        }
        
        function setupAudioVisualization() {
            addDebugMessage('🎨 Setting up audio visualization...');
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                const source = audioContext.createMediaStreamSource(localStream);
                source.connect(analyser);
                analyser.fftSize = 256;
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                let lastSpeechTime = 0;
                
                function updateVisualization() {
                    if (!analyser) return;
                    analyser.getByteFrequencyData(dataArray);
                    const average = dataArray.reduce((a, b) => a + b) / bufferLength;
                    visualizer.style.background = `linear-gradient(90deg, rgba(76, 175, 80, ${average / 255}) 0%, rgba(33, 150, 243, ${average / 255}) 100%)`;
                    if (average > 20) { visualizer.textContent = '🎤 Speaking...'; lastSpeechTime = Date.now(); } else if (Date.now() - lastSpeechTime < 3000) { visualizer.textContent = '🎧 Processing...'; } else { visualizer.textContent = '👂 Listening...'; }
                    requestAnimationFrame(updateVisualization);
                }
                updateVisualization();
                addDebugMessage('✅ Audio visualization setup complete');
            } catch (error) {
                addDebugMessage(`❌ Error setting up audio visualization: ${error.message}`);
                console.error("Full error in setupAudioVisualization:", error);
            }
        }
        
        function playAudio(remoteStream) {
            addDebugMessage('🔊 Attempting to play remote audio stream...');
            try {
                const audio = new Audio();
                audio.srcObject = remoteStream;
                audio.autoplay = true;
                audio.play().then(() => { 
                    addDebugMessage('✅ Audio playback started successfully'); 
                }).catch(error => { 
                    addDebugMessage(`❌ Audio playback error: ${error.message}`);
                    console.error("Error during audio.play():", error);
                    // Attempt to recover with a user gesture
                    document.body.addEventListener('click', () => {
                        addDebugMessage('Attempting to play audio after user click...');
                        audio.play().then(() => addDebugMessage('✅ Audio playback resumed after user click')).catch(resumeError => addDebugMessage(`❌ Failed to resume audio playback after click: ${resumeError.message}`));
                    }, { once: true });
                    updateStatus('Click anywhere to enable audio playback!', 'error');
                });
            } catch (error) {
                addDebugMessage(`❌ Exception during playAudio setup: ${error.message}`);
                console.error("Full error in playAudio:", error);
            }
        }
        
        function stopConversation(closeWs = true) {
            addDebugMessage('🛑 stopConversation() called');
            if (closeWs && websocket && websocket.readyState === WebSocket.OPEN) {
                addDebugMessage('➡️ Closing WebSocket...');
                websocket.close();
            } else {
                addDebugMessage('ℹ️ WebSocket not open or closeWs is false, skipping close.');
            }
            websocket = null;
            
            if (peerConnection) {
                addDebugMessage('➡️ Closing PeerConnection...');
                peerConnection.close();
                peerConnection = null;
            }
            
            if (localStream) {
                addDebugMessage('➡️ Stopping local media tracks...');
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }
            
            if (audioContext) {
                addDebugMessage('➡️ Closing AudioContext...');
                audioContext.close().catch(e => addDebugMessage(`⚠️ Error closing AudioContext: ${e.message}`));
                audioContext = null;
                analyser = null;
            }
            
            updateStatus('🔌 Disconnected', '');
            updateConnectionInfo('Connection Status: Disconnected');
            startBtn.disabled = false;
            stopBtn.disabled = true;
            
            visualizer.style.background = 'rgba(255, 255, 255, 0.1)';
            visualizer.textContent = 'Waiting to start...';
            addDebugMessage('✅ stopConversation() finished');
        }
        
        function updateStatus(message, className) {
            status.textContent = message;
            status.className = 'status ' + className;
        }
    </script>
</body>
</html>
"""

async def handle_http(request):
    """Serve the HTML client"""
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def handle_favicon(request):
    """Handle favicon request"""
    return web.Response(status=204) # No Content

async def main():
    """Main function with enhanced error handling"""
    print("🚀 UltraChat S2S - DEFINITIVE FIX - Starting server...")
    
    if not initialize_models():
        print("❌ Failed to initialize models. Exiting.")
        return
    
    print("🎉 Starting web server...")
    app = web.Application()
    app.router.add_get('/', handle_http)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/favicon.ico', handle_favicon)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    print("✅ Server started successfully!")
    print(f"🌐 Server running on http://0.0.0.0:7860")
    print("📱 Open the URL in your browser to start chatting!")
    
    try:
        await asyncio.Future()  # Run forever
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n🛑 Shutting down...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
