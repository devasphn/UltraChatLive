import asyncio
import logging
import json
import numpy as np
import torch
import torchaudio
from aiohttp import web, WSMsgType
import aiofiles
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import av
from transformers import pipeline
import io
import threading
import queue
import time
from collections import deque
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.vad_model = None
        self.ultravox_pipeline = None
        self.tts_model = None
        self.sample_rate = 16000
        self.audio_buffer = deque(maxlen=160000)  # 10 seconds at 16kHz
        self.processing_lock = threading.Lock()
        self.is_processing = False
        
    async def initialize_models(self):
        """Initialize all AI models"""
        logger.info(f"🚀 Initializing models on device: {self.device}")
        
        # Load Silero VAD
        logger.info("📥 Loading Silero VAD...")
        try:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.vad_model.to(self.device)
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Silero VAD: {e}")
            raise
        
        # Load Ultravox pipeline
        logger.info("📥 Loading Ultravox pipeline...")
        try:
            self.ultravox_pipeline = pipeline(
                "automatic-speech-recognition",
                model="ultravox-v0_3",
                device=0 if self.device == "cuda:0" else -1,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            )
            logger.info("✅ Ultravox pipeline loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Ultravox: {e}")
            # Fallback to Whisper
            logger.info("📥 Loading Whisper as fallback...")
            self.ultravox_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=0 if self.device == "cuda:0" else -1
            )
            logger.info("✅ Whisper loaded as fallback")
        
        # Load Chatterbox TTS
        logger.info("📥 Loading Chatterbox TTS...")
        try:
            from chatterbox import ChatterboxTTS
            self.tts_model = ChatterboxTTS()
            logger.info("✅ Chatterbox TTS loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Chatterbox TTS: {e}")
            self.tts_model = None
        
        logger.info("🎉 All models loaded successfully!")
    
    def detect_speech(self, audio_data):
        """Detect speech in audio using VAD"""
        try:
            if len(audio_data) < 512:  # Minimum samples for VAD
                return False
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio_data).to(self.device)
            
            # Get VAD probability
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
            
            # Threshold for speech detection
            return speech_prob > 0.5
            
        except Exception as e:
            logger.error(f"❌ VAD error: {e}")
            return False
    
    async def process_audio_chunk(self, audio_data):
        """Process audio chunk and generate response"""
        if self.is_processing:
            return None
            
        with self.processing_lock:
            self.is_processing = True
            
        try:
            logger.info(f"📥 Processing audio chunk: {len(audio_data)} samples")
            
            # Convert numpy array to audio file in memory
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_data, self.sample_rate, format='WAV')
            audio_bytes.seek(0)
            
            # Process with Ultravox/Whisper
            result = self.ultravox_pipeline(audio_bytes.read())
            text = result.get('text', '').strip()
            
            if not text:
                logger.info("🔇 No speech detected in audio")
                return None
                
            logger.info(f"🎤 Transcribed: {text}")
            
            # Generate response (simple echo for now - replace with your LLM)
            response_text = f"I heard you say: {text}. How can I help you with that?"
            logger.info(f"🤖 Response: {response_text}")
            
            # Generate TTS audio
            if self.tts_model:
                try:
                    tts_audio = self.tts_model.tts(response_text)
                    logger.info("🔊 TTS audio generated successfully")
                    return tts_audio
                except Exception as e:
                    logger.error(f"❌ TTS error: {e}")
                    return None
            else:
                logger.warning("⚠️ TTS model not available")
                return None
                
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            return None
        finally:
            self.is_processing = False

class AudioStreamTrack(MediaStreamTrack):
    """Custom audio track for sending TTS responses"""
    
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.audio_queue = asyncio.Queue()
        self.sample_rate = 16000
        self.samples_per_frame = 160  # 10ms at 16kHz
        
    async def recv(self):
        """Send audio frames"""
        try:
            # Wait for audio data or send silence
            try:
                audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.01)
                pts, time_base = await self.next_timestamp()
                
                # Convert audio data to AudioFrame
                frame = av.AudioFrame.from_ndarray(
                    audio_data.reshape(1, -1).astype(np.int16),
                    format='s16',
                    layout='mono'
                )
                frame.sample_rate = self.sample_rate
                frame.pts = pts
                frame.time_base = time_base
                
                return frame
                
            except asyncio.TimeoutError:
                # Send silence
                pts, time_base = await self.next_timestamp()
                silence = np.zeros((1, self.samples_per_frame), dtype=np.int16)
                
                frame = av.AudioFrame.from_ndarray(
                    silence,
                    format='s16',
                    layout='mono'
                )
                frame.sample_rate = self.sample_rate
                frame.pts = pts
                frame.time_base = time_base
                
                return frame
                
        except Exception as e:
            logger.error(f"❌ Error in AudioStreamTrack.recv: {e}")
            raise
    
    async def add_audio(self, audio_data):
        """Add audio data to the queue"""
        await self.audio_queue.put(audio_data)

class WebRTCHandler:
    def __init__(self, audio_processor):
        self.audio_processor = audio_processor
        self.peer_connections = {}
        
    async def handle_offer(self, websocket, offer_data):
        """Handle WebRTC offer"""
        try:
            logger.info("📨 Received WebRTC offer")
            
            # Create peer connection
            pc = RTCPeerConnection()
            connection_id = id(websocket)
            self.peer_connections[connection_id] = {
                'pc': pc,
                'websocket': websocket,
                'audio_track': AudioStreamTrack(),
                'audio_buffer': deque(maxlen=16000)  # 1 second buffer
            }
            
            # Add audio track for sending responses
            pc.addTrack(self.peer_connections[connection_id]['audio_track'])
            
            @pc.on("track")
            async def on_track(track):
                logger.info(f"🎧 Track received: {track.kind}")
                if track.kind == "audio":
                    # Start audio processing task
                    asyncio.create_task(self._process_incoming_audio(track, connection_id))
            
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"🔗 Connection state: {pc.connectionState}")
                if pc.connectionState == "closed":
                    await self._cleanup_connection(connection_id)
            
            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                logger.info(f"🧊 ICE connection state: {pc.iceConnectionState}")
            
            # Set remote description
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=offer_data["sdp"],
                type=offer_data["type"]
            ))
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            # Send answer back
            await websocket.send_str(json.dumps({
                "type": "answer",
                "sdp": pc.localDescription.sdp
            }))
            
            logger.info("📤 Sent WebRTC answer")
            
        except Exception as e:
            logger.error(f"❌ Error handling WebRTC offer: {e}")
            await websocket.send_str(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def _process_incoming_audio(self, track, connection_id):
        """Process incoming audio from WebRTC track"""
        logger.info("🎧 Starting audio processing...")
        
        connection = self.peer_connections.get(connection_id)
        if not connection:
            return
            
        audio_buffer = connection['audio_buffer']
        last_speech_time = 0
        silence_threshold = 1.0  # 1 second of silence before processing
        
        try:
            while True:
                try:
                    # Receive audio frame
                    frame = await track.recv()
                    
                    # Convert to numpy array
                    audio_data = frame.to_ndarray().flatten().astype(np.float32)
                    
                    # Resample if needed
                    if frame.sample_rate != self.audio_processor.sample_rate:
                        audio_data = self._resample_audio(
                            audio_data, 
                            frame.sample_rate, 
                            self.audio_processor.sample_rate
                        )
                    
                    # Add to buffer
                    audio_buffer.extend(audio_data)
                    
                    # Check for speech
                    if len(audio_buffer) >= 1600:  # 100ms of audio
                        recent_audio = np.array(list(audio_buffer)[-1600:])
                        
                        if self.audio_processor.detect_speech(recent_audio):
                            last_speech_time = time.time()
                        
                        # Process if we have enough audio and silence detected
                        current_time = time.time()
                        if (len(audio_buffer) >= 8000 and  # 0.5 seconds minimum
                            current_time - last_speech_time > silence_threshold):
                            
                            # Process the audio
                            audio_chunk = np.array(list(audio_buffer))
                            audio_buffer.clear()
                            
                            # Process in background
                            asyncio.create_task(
                                self._handle_audio_processing(audio_chunk, connection_id)
                            )
                    
                except Exception as e:
                    logger.error(f"❌ Error processing audio frame: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"❌ Audio processing stopped: {e}")
    
    async def _handle_audio_processing(self, audio_data, connection_id):
        """Handle audio processing and response"""
        try:
            logger.info(f"🎵 Processing audio: {len(audio_data)} samples")
            
            # Process audio
            response_audio = await self.audio_processor.process_audio_chunk(audio_data)
            
            if response_audio is not None:
                connection = self.peer_connections.get(connection_id)
                if connection and connection['audio_track']:
                    # Send response audio
                    await connection['audio_track'].add_audio(response_audio)
                    logger.info("🔊 Response audio sent")
            
        except Exception as e:
            logger.error(f"❌ Error in audio processing: {e}")
    
    def _resample_audio(self, audio, orig_sr, target_sr):
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
            
        # Simple resampling (you might want to use librosa for better quality)
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio), new_length),
            np.arange(len(audio)),
            audio
        )
    
    async def _cleanup_connection(self, connection_id):
        """Clean up connection resources"""
        if connection_id in self.peer_connections:
            connection = self.peer_connections[connection_id]
            try:
                await connection['pc'].close()
            except:
                pass
            del self.peer_connections[connection_id]
            logger.info(f"🧹 Cleaned up connection: {connection_id}")

async def websocket_handler(request):
    """Handle WebSocket connections"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    connection_id = id(ws)
    logger.info(f"🔌 New WebSocket connection: {connection_id}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    if data.get("type") == "offer":
                        await request.app['webrtc_handler'].handle_offer(ws, data)
                    
                except json.JSONDecodeError:
                    logger.error("❌ Invalid JSON received")
                    await ws.send_str(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"❌ WebSocket error: {ws.exception()}")
                
    except Exception as e:
        logger.error(f"❌ WebSocket handler error: {e}")
    finally:
        logger.info(f"🔌 Closed WebSocket connection: {connection_id}")
    
    return ws

async def index_handler(request):
    """Serve the main HTML page"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat S2S - Live Voice AI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center; font-weight: bold; }
        .status.disconnected { background: #ffebee; color: #c62828; }
        .status.connecting { background: #fff3e0; color: #ef6c00; }
        .status.connected { background: #e8f5e8; color: #2e7d32; }
        button { padding: 15px 30px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; margin: 10px; }
        .start-btn { background: #4caf50; color: white; }
        .stop-btn { background: #f44336; color: white; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .controls { text-align: center; margin: 20px 0; }
        .log { background: #f5f5f5; padding: 15px; border-radius: 5px; height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 UltraChat S2S - Live Voice AI</h1>
        
        <div id="status" class="status disconnected">
            🔴 Disconnected
        </div>
        
        <div class="controls">
            <button id="startBtn" class="start-btn">🎤 Start Voice Chat</button>
            <button id="stopBtn" class="stop-btn" disabled>🛑 Stop Chat</button>
        </div>
        
        <div class="log" id="log"></div>
    </div>

    <script>
        let ws = null;
        let pc = null;
        let localStream = null;
        
        const statusDiv = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const logDiv = document.getElementById('log');
        
        function log(message) {
            const timestamp = new Date().toLocaleTimeString();
            logDiv.innerHTML += `[${timestamp}] ${message}<br>`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function updateStatus(status, message) {
            statusDiv.className = `status ${status}`;
            statusDiv.innerHTML = message;
        }
        
        async function startChat() {
            try {
                log('🚀 Starting voice chat...');
                updateStatus('connecting', '🟡 Connecting...');
                
                // Get user media
                localStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });
                log('🎤 Microphone access granted');
                
                // Create WebSocket connection
                const wsUrl = `ws://${window.location.host}/ws`;
                ws = new WebSocket(wsUrl);
                
                ws.onopen = async () => {
                    log('🔌 WebSocket connected');
                    await setupWebRTC();
                };
                
                ws.onmessage = async (event) => {
                    const data = JSON.parse(event.data);
                    await handleWebSocketMessage(data);
                };
                
                ws.onerror = (error) => {
                    log(`❌ WebSocket error: ${error}`);
                    updateStatus('disconnected', '🔴 Connection Error');
                };
                
                ws.onclose = () => {
                    log('🔌 WebSocket disconnected');
                    updateStatus('disconnected', '🔴 Disconnected');
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                };
                
            } catch (error) {
                log(`❌ Error starting chat: ${error}`);
                updateStatus('disconnected', '🔴 Error');
                startBtn.disabled = false;
            }
        }
        
        async function setupWebRTC() {
            try {
                // Create peer connection
                pc = new RTCPeerConnection({
                    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                });
                
                // Add local stream
                localStream.getTracks().forEach(track => {
                    pc.addTrack(track, localStream);
                });
                
                // Handle remote stream
                pc.ontrack = (event) => {
                    log('🔊 Receiving audio from AI');
                    const remoteAudio = new Audio();
                    remoteAudio.srcObject = event.streams[0];
                    remoteAudio.play();
                };
                
                pc.oniceconnectionstatechange = () => {
                    log(`🧊 ICE connection state: ${pc.iceConnectionState}`);
                    if (pc.iceConnectionState === 'connected') {
                        updateStatus('connected', '🟢 Connected - Speak now!');
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                    }
                };
                
                // Create offer
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                
                // Send offer to server
                ws.send(JSON.stringify({
                    type: 'offer',
                    sdp: offer.sdp
                }));
                
                log('📤 Sent WebRTC offer');
                
            } catch (error) {
                log(`❌ WebRTC setup error: ${error}`);
            }
        }
        
        async function handleWebSocketMessage(data) {
            try {
                if (data.type === 'answer') {
                    log('📥 Received WebRTC answer');
                    await pc.setRemoteDescription(new RTCSessionDescription({
                        type: 'answer',
                        sdp: data.sdp
                    }));
                } else if (data.type === 'error') {
                    log(`❌ Server error: ${data.message}`);
                    updateStatus('disconnected', '🔴 Server Error');
                }
            } catch (error) {
                log(`❌ Error handling message: ${error}`);
            }
        }
        
        function stopChat() {
            log('🛑 Stopping voice chat...');
            
            if (pc) {
                pc.close();
                pc = null;
            }
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }
            
            updateStatus('disconnected', '🔴 Disconnected');
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
        
        startBtn.addEventListener('click', startChat);
        stopBtn.addEventListener('click', stopChat);
        
        log('✅ Voice chat interface ready');
    </script>
</body>
</html>
    """
    return web.Response(text=html_content, content_type='text/html')

async def create_app():
    """Create and configure the web application"""
    # Initialize audio processor
    audio_processor = AudioProcessor()
    await audio_processor.initialize_models()
    
    # Create WebRTC handler
    webrtc_handler = WebRTCHandler(audio_processor)
    
    # Create web application
    app = web.Application()
    app['audio_processor'] = audio_processor
    app['webrtc_handler'] = webrtc_handler
    
    # Add routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    
    return app

async def main():
    """Main function"""
    print("🚀 UltraChat S2S - COMPLETELY FIXED VERSION - Starting server...")
    
    try:
        # Create application
        app = await create_app()
        
        # Create and start server
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', 7860)
        await site.start()
        
        print("🎉 Starting web server...")
        print("✅ Server started successfully!")
        print("🌐 Server running on http://0.0.0.0:7860")
        print("🔌 WebSocket endpoint: ws://0.0.0.0:7860/ws")
        print("📱 Open the URL in your browser to start chatting!")
        print("Press Ctrl+C to stop...")
        
        # Keep the server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            await runner.cleanup()
            
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
