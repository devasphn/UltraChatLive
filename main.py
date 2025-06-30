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

warnings.filterwarnings("ignore")

# ENHANCED LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress empty candidate warnings
logging.getLogger('aioice.ice').setLevel(logging.WARNING)

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
            # Load Silero VAD model
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils
            
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
            # Ensure audio is float32 and 1D
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor)
            
            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()
            
            if len(audio_tensor.shape) > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # Normalize audio to [-1, 1] range
            if audio_tensor.abs().max() > 1.0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            logger.debug(f"🔍 VAD processing {len(audio_tensor)} samples at {sample_rate}Hz")
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.model,
                sampling_rate=sample_rate,
                threshold=0.3,
                min_speech_duration_ms=200,
                min_silence_duration_ms=300
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
    
    # 1. Load Silero VAD
    logger.info("📥 Loading Silero VAD...")
    vad_model = SileroVAD()
    if vad_model.model is None:
        logger.error("❌ Failed to load VAD model")
        return False
    
    # 2. Load Ultravox pipeline
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
    
    # 3. Load Chatterbox TTS
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
    """Enhanced audio buffer with improved VAD logic and debugging"""
    def __init__(self, max_duration=5.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time = time.time()
        self.min_speech_samples = int(1.0 * sample_rate)
        self.process_interval = 1.5  # Process every 1.5 seconds
        self.total_samples_received = 0
        
    def add_audio(self, audio_data):
        """Add audio data to buffer with enhanced logging"""
        if isinstance(audio_data, np.ndarray):
            audio_data = audio_data.flatten()
        
        # Convert to float32 and normalize
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize to [-1, 1] range
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        for sample in audio_data:
            self.buffer.append(sample)
        
        self.total_samples_received += len(audio_data)
        
        logger.debug(f"🎵 Buffer: {len(self.buffer)}/{self.max_samples} samples "
                    f"(total received: {self.total_samples_received})")
    
    def get_audio_array(self):
        """Get current audio as numpy array"""
        if len(self.buffer) == 0:
            return np.array([])
        return np.array(list(self.buffer), dtype=np.float32)
    
    def should_process(self):
        """Enhanced processing logic with detailed logging"""
        current_time = time.time()
        
        has_enough_audio = len(self.buffer) >= self.min_speech_samples
        enough_time_passed = (current_time - self.last_process_time) >= self.process_interval
        
        logger.debug(f"🔍 Process check: audio={has_enough_audio} "
                    f"({len(self.buffer)}/{self.min_speech_samples}), "
                    f"time={enough_time_passed} "
                    f"({current_time - self.last_process_time:.1f}s)")
        
        if has_enough_audio and enough_time_passed:
            audio_array = self.get_audio_array()
            
            # Use VAD to detect speech
            has_speech = vad_model.detect_speech(audio_array, self.sample_rate)
            
            logger.info(f"🎯 Processing decision: {len(audio_array)} samples, "
                       f"Speech detected: {has_speech}")
            
            if has_speech:
                self.last_process_time = current_time
                return True
        
        return False
    
    def reset(self):
        """Reset buffer state"""
        self.buffer.clear()
        logger.info("🔄 Audio buffer reset")

class OptimizedTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def synthesize(self, text: str) -> np.ndarray:
        """Enhanced TTS synthesis with better error handling"""
        if not text.strip():
            logger.warning("⚠️ Empty text for TTS, returning silence")
            return np.zeros(1600, dtype=np.float32)

        logger.info(f"🎤 Synthesizing: '{text[:100]}...'")
        
        try:
            # Generate audio with optimizations
            with torch.inference_mode():
                wav = tts_model.generate(text)
                
            if not isinstance(wav, torch.Tensor):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
                
            # Convert to numpy array and ensure correct format
            result = wav.cpu().numpy().flatten().astype(np.float32)
            
            # Ensure sample rate is 16kHz
            if hasattr(tts_model, 'sample_rate') and tts_model.sample_rate != 16000:
                result = librosa.resample(result, orig_sr=tts_model.sample_rate, target_sr=16000)
            
            logger.info(f"✅ TTS generated {len(result)} samples")
            return result
            
        except Exception as e:
            logger.error(f"❌ TTS synthesis error: {e}")
            return np.zeros(1600, dtype=np.float32)

class AudioStreamTrack(MediaStreamTrack):
    """FIXED: Audio track with proper PyAV format and enhanced debugging"""
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.buffer = AudioBuffer()
        self.tts = OptimizedTTS()
        self._response_queue = asyncio.Queue()
        self._current_response = None
        self._response_position = 0
        # Initialize timestamp tracking
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 16000)
        
        # Consumer task
        self._consumer_task = None
        self._is_running = False
        
    def start_consumer(self):
        """Start the audio consumer task"""
        if self._consumer_task is None:
            self._is_running = True
            self._consumer_task = asyncio.create_task(self._consume_audio())
            logger.info("🎧 Audio consumer started")
        
    async def _consume_audio(self):
        """Consumer task that calls recv() continuously"""
        try:
            while self._is_running:
                await self.recv()
                await asyncio.sleep(0.1)  # 100ms intervals
        except asyncio.CancelledError:
            logger.info("🛑 Audio consumer stopped")
        except Exception as e:
            logger.error(f"❌ Audio consumer error: {e}")
        
    async def recv(self):
        """FIXED: Audio frame generation with proper PyAV format"""
        frame_samples = 1600  # 0.1 seconds at 16kHz
        
        # Check if we have response audio to send
        if self._current_response is None or self._response_position >= len(self._current_response):
            try:
                self._current_response = await asyncio.wait_for(
                    self._response_queue.get(), timeout=0.001
                )
                self._response_position = 0
                logger.info(f"🔊 Starting new response playback: {len(self._current_response)} samples")
            except asyncio.TimeoutError:
                # No response available, send silence
                frame = np.zeros(frame_samples, dtype=np.float32)
        
        if self._current_response is not None:
            # Extract frame from current response
            end_pos = min(self._response_position + frame_samples, len(self._current_response))
            frame = self._current_response[self._response_position:end_pos]
            self._response_position = end_pos
            
            # Pad if needed
            if len(frame) < frame_samples:
                frame = np.pad(frame, (0, frame_samples - len(frame)), 'constant')
        else:
            frame = np.zeros(frame_samples, dtype=np.float32)
        
        # CRITICAL FIX: PyAV requires 2D array with shape (1, samples) for mono
        frame_2d = frame.reshape(1, -1)
        
        # Create AudioFrame using PyAV with correct format
        try:
            audio_frame = av.AudioFrame.from_ndarray(
                frame_2d, 
                format="flt",
                layout="mono"
            )
        except Exception as e:
            logger.error(f"❌ PyAV AudioFrame creation error: {e}")
            # Fallback to silence
            silence_2d = np.zeros((1, frame_samples), dtype=np.float32)
            audio_frame = av.AudioFrame.from_ndarray(
                silence_2d, 
                format="flt",
                layout="mono"
            )
        
        # Set timestamp properly
        audio_frame.pts = self._timestamp
        audio_frame.time_base = self._time_base
        audio_frame.sample_rate = 16000
        
        # Update timestamp for next frame
        self._timestamp += frame_samples
        
        return audio_frame
    
    def process_audio_data(self, audio_data):
        """Process incoming audio data with enhanced logging"""
        logger.debug(f"📥 Received audio data: shape={audio_data.shape}, "
                    f"dtype={audio_data.dtype}, range=[{audio_data.min():.3f}, {audio_data.max():.3f}]")
        
        self.buffer.add_audio(audio_data)
        
        if self.buffer.should_process():
            # Get audio for processing
            audio_array = self.buffer.get_audio_array()
            if len(audio_array) > 0:
                logger.info(f"🎯 Processing speech: {len(audio_array)} samples")
                # Process in background
                asyncio.create_task(self._process_speech(audio_array))
            self.buffer.reset()
    
    async def _process_speech(self, audio_array):
        """Process speech and generate response with enhanced error handling"""
        try:
            logger.info("🧠 Processing speech with Ultravox...")
            
            # Generate response
            response_text = await self._generate_response(audio_array)
            if response_text.strip():
                logger.info(f"💬 AI Response: '{response_text}'")
                # Generate TTS
                response_audio = self.tts.synthesize(response_text)
                # Queue response audio for sending
                await self._response_queue.put(response_audio)
                logger.info("🎵 Response audio queued for playback")
            else:
                logger.warning("⚠️ Empty response generated")
                
        except Exception as e:
            logger.error(f"❌ Error processing speech: {e}")
    
    async def _generate_response(self, audio_array):
        """Generate response using Ultravox with enhanced logging"""
        try:
            # Ensure audio is in correct format for Ultravox
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize audio
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / np.abs(audio_array).max()
            
            turns = [{
                "role": "system", 
                "content": "You are a helpful voice assistant. Give brief, natural responses under 20 words."
            }]
            
            logger.info(f"🤖 Calling Ultravox with {len(audio_array)} audio samples...")
            
            with torch.inference_mode():
                result = uv_pipe({
                    'audio': audio_array,
                    'turns': turns,
                    'sampling_rate': 16000
                }, 
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=uv_pipe.tokenizer.eos_token_id
                )
            
            # Extract response text
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "generated_text" in result[0]:
                    response_text = result[0]["generated_text"]
                else:
                    response_text = str(result[0])
            elif isinstance(result, dict) and "generated_text" in result:
                response_text = result["generated_text"]
            else:
                response_text = str(result)
            
            logger.info(f"✅ Ultravox response: '{response_text}'")
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "I'm sorry, I couldn't process that."
    
    def stop_consumer(self):
        """Stop the consumer task"""
        self._is_running = False
        if self._consumer_task:
            self._consumer_task.cancel()

class WebRTCConnection:
    def __init__(self):
        self.pc = RTCPeerConnection()
        self.audio_track = AudioStreamTrack()
        self.recorder = None
        
        # Add event handlers
        @self.pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"📡 Data channel created: {channel.label}")
        
        @self.pc.on("track")
        def on_track(track):
            logger.info(f"🎧 Track received: {track.kind}")
            if track.kind == "audio":
                # Handle incoming audio
                asyncio.create_task(self._handle_audio_track(track))
    
    async def _handle_audio_track(self, track):
        """Handle incoming audio track with enhanced error handling"""
        logger.info("🎧 Starting audio track handler...")
        frame_count = 0
        try:
            while True:
                try:
                    frame = await track.recv()
                    frame_count += 1
                    
                    # Convert frame to numpy array
                    audio_data = frame.to_ndarray()
                    
                    # Log every 50th frame to avoid spam
                    if frame_count % 50 == 0:
                        logger.debug(f"🎵 Processed {frame_count} audio frames")
                    
                    # Process with our audio track
                    self.audio_track.process_audio_data(audio_data)
                    
                except Exception as frame_error:
                    logger.error(f"❌ Frame processing error: {frame_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error handling audio track: {e}")

def is_valid_ice_candidate(candidate_data):
    """Filter out empty candidates with enhanced validation"""
    if not candidate_data:
        return False
    
    candidate_string = candidate_data.get("candidate", "")
    if not candidate_string or candidate_string.strip() == "":
        return False
    
    try:
        parts = candidate_string.split()
        if len(parts) < 6:
            return False
        
        ip_address = parts[4]
        if not ip_address or ip_address == "":
            return False
            
        return True
        
    except Exception as e:
        logger.debug(f"ICE candidate validation error: {e}")
        return False

# WebSocket handler with enhanced error handling
async def websocket_handler(request):
    """Handle WebSocket connections with comprehensive error handling"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    connection_id = id(ws)
    webrtc_connection = WebRTCConnection()
    
    logger.info(f"🔌 New WebSocket connection: {connection_id}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    pc = webrtc_connection.pc
                    
                    if data["type"] == "offer":
                        logger.info("📨 Received WebRTC offer")
                        
                        # Handle WebRTC offer
                        await pc.setRemoteDescription(RTCSessionDescription(
                            sdp=data["sdp"], type=data["type"]
                        ))
                        
                        # Add audio track
                        pc.addTrack(webrtc_connection.audio_track)
                        
                        # CRITICAL FIX: Add recorder to consume the outgoing audio track
                        webrtc_connection.recorder = MediaBlackhole()
                        webrtc_connection.recorder.addTrack(webrtc_connection.audio_track)
                        await webrtc_connection.recorder.start()
                        logger.info("🎵 MediaBlackhole consumer started")
                        
                        # CRITICAL FIX: Start the audio consumer
                        webrtc_connection.audio_track.start_consumer()
                        
                        # Create answer
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        
                        await ws.send_str(json.dumps({
                            "type": "answer",
                            "sdp": pc.localDescription.sdp
                        }))
                        
                        logger.info("📤 Sent WebRTC answer")
                        
                    elif data["type"] == "ice-candidate":
                        candidate_data = data["candidate"]
                        
                        if not is_valid_ice_candidate(candidate_data):
                            continue
                        
                        try:
                            candidate = RTCIceCandidate(
                                component=candidate_data.get("component", 1),
                                foundation=candidate_data.get("foundation", ""),
                                ip=candidate_data.get("address", ""),
                                port=candidate_data.get("port", 0),
                                priority=candidate_data.get("priority", 0),
                                protocol=candidate_data.get("protocol", "udp"),
                                type=candidate_data.get("type", "host"),
                                sdpMid=candidate_data.get("sdpMid"),
                                sdpMLineIndex=candidate_data.get("sdpMLineIndex")
                            )
                            
                            await pc.addIceCandidate(candidate)
                            
                        except Exception as candidate_error:
                            logger.error(f"❌ Failed to add ICE candidate: {candidate_error}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"❌ Error handling WebSocket message: {e}")
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f'❌ WebSocket error: {ws.exception()}')
                break
                
    except Exception as e:
        logger.error(f"❌ WebSocket handler error: {e}")
    finally:
        # Cleanup
        if webrtc_connection.recorder:
            await webrtc_connection.recorder.stop()
        webrtc_connection.audio_track.stop_consumer()
        await webrtc_connection.pc.close()
        logger.info(f"🔌 Closed WebSocket connection: {connection_id}")
    
    return ws

# Enhanced HTML client with better debugging
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat S2S - FIXED</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 15px 30px;
            font-size: 16px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .start-btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
        }
        .stop-btn {
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .status {
            text-align: center;
            font-size: 18px;
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.2);
        }
        .status.connected {
            background: rgba(76, 175, 80, 0.3);
        }
        .status.error {
            background: rgba(244, 67, 54, 0.3);
        }
        .audio-visualizer {
            height: 100px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            margin: 20px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }
        .debug-info {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            font-family: monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
        }
        .instructions {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .instructions h3 {
            margin-top: 0;
        }
        .instructions ul {
            padding-left: 20px;
        }
        .instructions li {
            margin: 10px 0;
        }
        .fix-note {
            background: rgba(76, 175, 80, 0.2);
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Real-time S2S AI Chat</h1>
        
        <div class="fix-note">
            <strong>✅ FIXED:</strong> All critical issues resolved - PyAV format, MediaBlackhole consumer, debug logging, and model loading.
        </div>
        
        <div class="controls">
            <button id="startBtn" class="start-btn">Start Conversation</button>
            <button id="stopBtn" class="stop-btn" disabled>Stop Conversation</button>
        </div>
        
        <div id="status" class="status">Ready - Start speaking!</div>
        
        <div class="audio-visualizer" id="visualizer">
            Listening...
        </div>
        
        <div class="debug-info" id="debugInfo">
            Debug info will appear here...
        </div>
        
        <div class="instructions">
            <h3>Instructions:</h3>
            <ul>
                <li>Click "Start Conversation" to begin</li>
                <li>Allow microphone access when prompted</li>
                <li>Speak naturally - the AI will detect when you're done speaking</li>
                <li>The AI will respond with voice automatically</li>
                <li>Click "Stop Conversation" to end the session</li>
            </ul>
        </div>
    </div>

    <script>
        let websocket = null;
        let peerConnection = null;
        let localStream = null;
        let audioContext = null;
        let analyser = null;
        
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');
        const visualizer = document.getElementById('visualizer');
        const debugInfo = document.getElementById('debugInfo');
        
        startBtn.addEventListener('click', startConversation);
        stopBtn.addEventListener('click', stopConversation);
        
        function addDebugMessage(message) {
            const timestamp = new Date().toLocaleTimeString();
            debugInfo.innerHTML += `[${timestamp}] ${message}<br>`;
            debugInfo.scrollTop = debugInfo.scrollHeight;
            console.log(message);
        }
        
        function getWebSocketUrl() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            return `${protocol}//${host}/ws`;
        }
        
        async function startConversation() {
            try {
                updateStatus('🔄 Connecting...', '');
                addDebugMessage('🚀 Starting conversation...');
                
                localStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000
                    }
                });
                
                addDebugMessage('🎤 Microphone access granted');
                setupAudioVisualization();
                
                const wsUrl = getWebSocketUrl();
                addDebugMessage(`🔗 Connecting to: ${wsUrl}`);
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = () => {
                    updateStatus('🔄 Connected - Setting up audio...', 'connected');
                    addDebugMessage('✅ WebSocket connected');
                    setupWebRTC();
                };
                
                websocket.onmessage = handleSignalingMessage;
                websocket.onerror = (error) => {
                    updateStatus('❌ Connection error', 'error');
                    addDebugMessage(`❌ WebSocket error: ${error}`);
                };
                
                websocket.onclose = () => {
                    updateStatus('🔌 Connection closed', 'error');
                    addDebugMessage('🔌 WebSocket closed');
                };
                
                startBtn.disabled = true;
                stopBtn.disabled = false;
                
            } catch (error) {
                updateStatus('❌ Error: ' + error.message, 'error');
                addDebugMessage(`❌ Error starting conversation: ${error.message}`);
            }
        }
        
        async function setupWebRTC() {
            try {
                addDebugMessage('🔧 Setting up WebRTC...');
                
                peerConnection = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' }
                    ]
                });
                
                localStream.getTracks().forEach(track => {
                    peerConnection.addTrack(track, localStream);
                    addDebugMessage(`📡 Added ${track.kind} track`);
                });
                
                peerConnection.ontrack = (event) => {
                    addDebugMessage(`🎵 Received ${event.track.kind} track`);
                    const [remoteStream] = event.streams;
                    playAudio(remoteStream);
                };
                
                peerConnection.onicecandidate = (event) => {
                    if (event.candidate && websocket.readyState === WebSocket.OPEN) {
                        const candidateStr = event.candidate.candidate;
                        if (candidateStr && candidateStr.trim() !== '' && candidateStr !== 'null') {
                            websocket.send(JSON.stringify({
                                type: 'ice-candidate',
                                candidate: event.candidate
                            }));
                            addDebugMessage('📤 Sent ICE candidate');
                        }
                    }
                };
                
                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);
                
                if (websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'offer',
                        sdp: offer.sdp
                    }));
                    addDebugMessage('📤 Sent WebRTC offer');
                }
                
                updateStatus('✅ Ready - Start speaking!', 'connected');
                
            } catch (error) {
                updateStatus('❌ WebRTC setup error: ' + error.message, 'error');
                addDebugMessage(`❌ WebRTC error: ${error.message}`);
            }
        }
        
        async function handleSignalingMessage(event) {
            try {
                const message = JSON.parse(event.data);
                addDebugMessage(`📥 Received: ${message.type}`);
                
                if (message.type === 'answer') {
                    await peerConnection.setRemoteDescription(new RTCSessionDescription({
                        type: 'answer',
                        sdp: message.sdp
                    }));
                    addDebugMessage('✅ Set remote description');
                } else if (message.type === 'ice-candidate') {
                    if (message.candidate && message.candidate.candidate && message.candidate.candidate.trim() !== '') {
                        await peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
                        addDebugMessage('✅ Added ICE candidate');
                    }
                }
            } catch (error) {
                addDebugMessage(`❌ Error handling signaling: ${error.message}`);
            }
        }
        
        function setupAudioVisualization() {
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
                
                visualizer.style.background = `linear-gradient(90deg, 
                    rgba(76, 175, 80, ${average / 255}) 0%, 
                    rgba(33, 150, 243, ${average / 255}) 100%)`;
                
                if (average > 50) {
                    visualizer.textContent = '🎤 Speaking...';
                    lastSpeechTime = Date.now();
                } else if (Date.now() - lastSpeechTime < 2000) {
                    visualizer.textContent = '🎧 Processing...';
                } else {
                    visualizer.textContent = '👂 Listening...';
                }
                
                requestAnimationFrame(updateVisualization);
            }
            
            updateVisualization();
        }
        
        function playAudio(remoteStream) {
            const audio = new Audio();
            audio.srcObject = remoteStream;
            audio.play().then(() => {
                addDebugMessage('🔊 Playing AI response');
            }).catch(error => {
                addDebugMessage(`❌ Audio playback error: ${error.message}`);
            });
        }
        
        function stopConversation() {
            if (websocket) {
                websocket.close();
                websocket = null;
            }
            
            if (peerConnection) {
                peerConnection.close();
                peerConnection = null;
            }
            
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }
            
            if (audioContext) {
                audioContext.close();
                audioContext = null;
                analyser = null;
            }
            
            updateStatus('🔌 Disconnected', '');
            startBtn.disabled = false;
            stopBtn.disabled = true;
            
            visualizer.style.background = 'rgba(255, 255, 255, 0.1)';
            visualizer.textContent = 'Listening...';
            
            addDebugMessage('🛑 Conversation stopped');
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
    favicon_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    return web.Response(body=favicon_data, content_type='image/png')

async def main():
    """Main function with enhanced error handling"""
    print("🚀 UltraChat S2S - Starting server...")
    
    # Initialize models
    if not initialize_models():
        print("❌ Failed to initialize models. Exiting.")
        return
    
    print("🎉 Starting web server...")
    
    # Create aiohttp application
    app = web.Application()
    app.router.add_get('/', handle_http)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/favicon.ico', handle_favicon)
    
    # Setup and start the server
    runner = web_runner.AppRunner(app)
    await runner.setup()
    
    # Use port 7860 as expected by RunPod
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    print("✅ Server started successfully!")
    print(f"🌐 Server running on http://0.0.0.0:7860")
    print(f"🔌 WebSocket endpoint: ws://0.0.0.0:7860/ws")
    print("📱 Open the URL in your browser to start chatting!")
    print("Press Ctrl+C to stop...")
    
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
