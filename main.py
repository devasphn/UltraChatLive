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
from aiortc.contrib.media import MediaRecorder, MediaRelay
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
from concurrent.futures import ThreadPoolExecutor
import warnings
import torch.hub
import collections
import time

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Global variables for models
uv_pipe = None
tts_model = None
vad_model = None
executor = ThreadPoolExecutor(max_workers=4)

class SileroVAD:
    def __init__(self):
        """Initialize Silero VAD model"""
        try:
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
            
            print("Silero VAD loaded successfully")
            
        except Exception as e:
            print(f"Error loading Silero VAD: {e}")
            self.model = None
    
    def detect_speech(self, audio_tensor, sample_rate=16000):
        """Detect speech in audio tensor"""
        if self.model is None:
            return True  # Fallback: assume speech is present
        
        try:
            # Ensure audio is float32 and 1D
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor)
            
            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()
            
            if len(audio_tensor.shape) > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.model,
                sampling_rate=sample_rate,
                threshold=0.3,  # Lower threshold for better detection
                min_speech_duration_ms=200,  # Shorter minimum
                min_silence_duration_ms=300   # Shorter silence
            )
            
            return len(speech_timestamps) > 0
            
        except Exception as e:
            print(f"VAD error: {e}")
            return True  # Fallback

def initialize_models():
    """Initialize all models once at startup"""
    global uv_pipe, tts_model, vad_model
    
    print(f"Using device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    
    # 1. Load Silero VAD
    print("Loading Silero VAD...")
    vad_model = SileroVAD()
    
    # 2. Load Ultravox pipeline
    try:
        print("Loading Ultravox pipeline...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("Ultravox pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading Ultravox: {e}")
        exit(1)
    
    # 3. Load Chatterbox TTS
    try:
        print("Loading Chatterbox TTS...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        print("Chatterbox TTS loaded successfully.")
    except Exception as e:
        print(f"Error loading TTS: {e}")
        exit(1)

class AudioBuffer:
    """FIXED: Simplified audio buffer with working VAD logic"""
    def __init__(self, max_duration=5.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time = time.time()
        self.min_speech_samples = int(1.0 * sample_rate)  # 1 second minimum
        self.process_interval = 2.0  # Process every 2 seconds
        
    def add_audio(self, audio_data):
        """Add audio data to buffer"""
        if isinstance(audio_data, np.ndarray):
            audio_data = audio_data.flatten()
        
        for sample in audio_data:
            self.buffer.append(sample)
        
        print(f"🎵 Buffer size: {len(self.buffer)} samples")
    
    def get_audio_array(self):
        """Get current audio as numpy array"""
        if len(self.buffer) == 0:
            return np.array([])
        return np.array(list(self.buffer), dtype=np.float32)
    
    def should_process(self):
        """FIXED: Simplified processing logic"""
        current_time = time.time()
        
        # Check if we have minimum audio and enough time has passed
        has_enough_audio = len(self.buffer) >= self.min_speech_samples
        enough_time_passed = (current_time - self.last_process_time) >= self.process_interval
        
        if has_enough_audio and enough_time_passed:
            audio_array = self.get_audio_array()
            
            # Use VAD to detect speech
            has_speech = vad_model.detect_speech(audio_array, self.sample_rate)
            
            print(f"🔍 VAD Check: {len(audio_array)} samples, Speech detected: {has_speech}")
            
            if has_speech:
                self.last_process_time = current_time
                return True
        
        return False
    
    def reset(self):
        """Reset buffer state"""
        self.buffer.clear()
        print("🔄 Buffer reset")

class OptimizedTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def synthesize(self, text: str) -> np.ndarray:
        """Optimized TTS synthesis returning numpy array"""
        if not text.strip():
            return np.zeros(1600, dtype=np.float32)

        print(f"🎤 Synthesizing: '{text[:50]}...'")
        
        # Generate audio with optimizations
        with torch.inference_mode():
            wav = tts_model.generate(text)
            
        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            
        # Convert to numpy array
        result = wav.cpu().numpy().flatten().astype(np.float32)
        print(f"✅ TTS generated {len(result)} samples")
        return result

class AudioStreamTrack(MediaStreamTrack):
    """FIXED: Audio track with working response queue"""
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
        
    async def recv(self):
        """FIXED: Audio frame generation with proper response handling"""
        frame_samples = 1600  # 0.1 seconds at 16kHz
        
        # Check if we have response audio to send
        if self._current_response is None or self._response_position >= len(self._current_response):
            try:
                self._current_response = await asyncio.wait_for(
                    self._response_queue.get(), timeout=0.001
                )
                self._response_position = 0
                print(f"🔊 Starting new response playback: {len(self._current_response)} samples")
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
        
        # Create AudioFrame using PyAV
        audio_frame = av.AudioFrame.from_ndarray(
            frame.reshape(1, -1), 
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
        """Process incoming audio data"""
        self.buffer.add_audio(audio_data)
        
        if self.buffer.should_process():
            # Get audio for processing
            audio_array = self.buffer.get_audio_array()
            if len(audio_array) > 0:
                print(f"🎯 Processing speech: {len(audio_array)} samples")
                # Process in background
                asyncio.create_task(self._process_speech(audio_array))
            self.buffer.reset()
    
    async def _process_speech(self, audio_array):
        """Process speech and generate response"""
        try:
            print("🧠 Generating AI response...")
            # Generate response
            response_text = await self._generate_response(audio_array)
            if response_text.strip():
                print(f"💬 AI Response: '{response_text}'")
                # Generate TTS
                response_audio = self.tts.synthesize(response_text)
                # Queue response audio for sending
                await self._response_queue.put(response_audio)
            else:
                print("⚠️ Empty response generated")
                
        except Exception as e:
            print(f"❌ Error processing speech: {e}")
    
    async def _generate_response(self, audio_array):
        """Generate response using Ultravox"""
        try:
            turns = [{
                "role": "system", 
                "content": "You are a helpful voice assistant. Give brief, natural responses under 20 words."
            }]
            
            print(f"🤖 Calling Ultravox with {len(audio_array)} audio samples...")
            
            with torch.inference_mode():
                result = uv_pipe({
                    'audio': audio_array,
                    'turns': turns,
                    'sampling_rate': 16000
                }, 
                max_new_tokens=32,  # Shorter for faster response
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
            
            print(f"✅ Ultravox response: '{response_text}'")
            return response_text.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I'm sorry, I couldn't process that."

class WebRTCConnection:
    def __init__(self):
        self.pc = RTCPeerConnection()
        self.audio_track = AudioStreamTrack()
        
        # Add event handlers
        @self.pc.on("datachannel")
        def on_datachannel(channel):
            print(f"Data channel created: {channel.label}")
        
        @self.pc.on("track")
        def on_track(track):
            print(f"Track received: {track.kind}")
            if track.kind == "audio":
                # Handle incoming audio
                asyncio.create_task(self._handle_audio_track(track))
    
    async def _handle_audio_track(self, track):
        """Handle incoming audio track with proper error handling"""
        print("🎧 Starting audio track handler...")
        try:
            while True:
                try:
                    frame = await track.recv()
                    # Convert frame to numpy array
                    audio_data = frame.to_ndarray()
                    # Process with our audio track
                    self.audio_track.process_audio_data(audio_data)
                except Exception as frame_error:
                    print(f"Frame processing error: {frame_error}")
                    # Continue processing other frames
                    continue
        except Exception as e:
            print(f"Error handling audio track: {e}")

def is_valid_ice_candidate(candidate_data):
    """Filter out empty candidates (end-of-candidates markers)"""
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
        return False

# WebSocket handler with proper ICE candidate filtering
async def websocket_handler(request):
    """Handle WebSocket connections using aiohttp with proper ICE candidate filtering"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    connection_id = id(ws)
    webrtc_connection = WebRTCConnection()
    
    print(f"🔌 New WebSocket connection: {connection_id}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    pc = webrtc_connection.pc
                    
                    if data["type"] == "offer":
                        print("📨 Received WebRTC offer")
                        # Handle WebRTC offer
                        await pc.setRemoteDescription(RTCSessionDescription(
                            sdp=data["sdp"], type=data["type"]
                        ))
                        
                        # Add audio track
                        pc.addTrack(webrtc_connection.audio_track)
                        
                        # Create answer
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        
                        await ws.send_str(json.dumps({
                            "type": "answer",
                            "sdp": pc.localDescription.sdp
                        }))
                        
                        print("📤 Sent WebRTC answer")
                        
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
                            print(f"❌ Failed to add ICE candidate: {candidate_error}")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                except Exception as e:
                    print(f"Error handling WebSocket message: {e}")
                    
            elif msg.type == WSMsgType.ERROR:
                print(f'WebSocket error: {ws.exception()}')
                break
                
    except Exception as e:
        print(f"WebSocket handler error: {e}")
    finally:
        await webrtc_connection.pc.close()
        print(f"🔌 Closed WebSocket connection: {connection_id}")
    
    return ws

# HTML client interface
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat S2S - Working Version</title>
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
        <h1>🎤 UltraChat S2S - Working Version</h1>
        
        <div class="fix-note">
            <strong>✅ Fixed:</strong> Audio processing pipeline corrected. AI responses now working properly.
        </div>
        
        <div class="controls">
            <button id="startBtn" class="start-btn">Start Conversation</button>
            <button id="stopBtn" class="stop-btn" disabled>Stop Conversation</button>
        </div>
        
        <div id="status" class="status">Ready to connect</div>
        
        <div class="audio-visualizer" id="visualizer">
            Audio visualization will appear here
        </div>
        
        <div class="instructions">
            <h3>Instructions:</h3>
            <ul>
                <li>Click "Start Conversation" to begin</li>
                <li>Allow microphone access when prompted</li>
                <li>Speak clearly for 2-3 seconds</li>
                <li>Wait for AI response (processing takes 5-10 seconds)</li>
                <li>AI will respond with voice automatically</li>
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
        
        startBtn.addEventListener('click', startConversation);
        stopBtn.addEventListener('click', stopConversation);
        
        function getWebSocketUrl() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            return `${protocol}//${host}/ws`;
        }
        
        async function startConversation() {
            try {
                updateStatus('Connecting...', '');
                
                localStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000
                    }
                });
                
                setupAudioVisualization();
                
                const wsUrl = getWebSocketUrl();
                console.log('Connecting to:', wsUrl);
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = () => {
                    updateStatus('Connected - Setting up audio...', 'connected');
                    setupWebRTC();
                };
                
                websocket.onmessage = handleSignalingMessage;
                websocket.onerror = (error) => {
                    updateStatus('Connection error', 'error');
                    console.error('WebSocket error:', error);
                };
                
                websocket.onclose = () => {
                    updateStatus('Connection closed', 'error');
                };
                
                startBtn.disabled = true;
                stopBtn.disabled = false;
                
            } catch (error) {
                updateStatus('Error: ' + error.message, 'error');
                console.error('Error starting conversation:', error);
            }
        }
        
        async function setupWebRTC() {
            try {
                peerConnection = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' }
                    ]
                });
                
                localStream.getTracks().forEach(track => {
                    peerConnection.addTrack(track, localStream);
                });
                
                peerConnection.ontrack = (event) => {
                    const [remoteStream] = event.streams;
                    playAudio(remoteStream);
                };
                
                peerConnection.onicecandidate = (event) => {
                    if (event.candidate && websocket.readyState === WebSocket.OPEN) {
                        if (event.candidate.candidate && event.candidate.candidate.trim() !== '') {
                            websocket.send(JSON.stringify({
                                type: 'ice-candidate',
                                candidate: event.candidate
                            }));
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
                }
                
                updateStatus('Ready - Start speaking!', 'connected');
                
            } catch (error) {
                updateStatus('WebRTC setup error: ' + error.message, 'error');
                console.error('WebRTC error:', error);
            }
        }
        
        async function handleSignalingMessage(event) {
            try {
                const message = JSON.parse(event.data);
                
                if (message.type === 'answer') {
                    await peerConnection.setRemoteDescription(new RTCSessionDescription({
                        type: 'answer',
                        sdp: message.sdp
                    }));
                } else if (message.type === 'ice-candidate') {
                    if (message.candidate && message.candidate.candidate && message.candidate.candidate.trim() !== '') {
                        await peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
                    }
                }
            } catch (error) {
                console.error('Error handling signaling message:', error);
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
            
            function updateVisualization() {
                if (!analyser) return;
                
                analyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((a, b) => a + b) / bufferLength;
                
                visualizer.style.background = `linear-gradient(90deg, 
                    rgba(76, 175, 80, ${average / 255}) 0%, 
                    rgba(33, 150, 243, ${average / 255}) 100%)`;
                
                if (average > 50) {
                    visualizer.textContent = 'Speaking...';
                } else {
                    visualizer.textContent = 'Listening...';
                }
                
                requestAnimationFrame(updateVisualization);
            }
            
            updateVisualization();
        }
        
        function playAudio(remoteStream) {
            const audio = new Audio();
            audio.srcObject = remoteStream;
            audio.play().catch(console.error);
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
            
            updateStatus('Disconnected', '');
            startBtn.disabled = false;
            stopBtn.disabled = true;
            
            visualizer.style.background = 'rgba(255, 255, 255, 0.1)';
            visualizer.textContent = 'Audio visualization will appear here';
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
    """Main function"""
    print("🚀 Initializing Real-time S2S Agent...")
    
    # Initialize models
    initialize_models()
    
    print("🎉 Starting server...")
    
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
    
    print("✅ Server started!")
    print(f"🌐 Server running on port 7860")
    print(f"🔌 WebSocket endpoint available at /ws")
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
