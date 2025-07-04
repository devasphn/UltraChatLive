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
import gc
import psutil
import os

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av

from transformers import pipeline
from moshi.models import build_model
from moshi.models.loaders import load_model
import torch.hub

# --- Performance Optimizations ---
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Enable memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

try:
    import uvloop
    uvloop.install()
    print("🚀 Using uvloop for maximum performance")
except ImportError:
    print("⚠️ uvloop not found, performance may be suboptimal")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimize logging for production
logging.getLogger('aioice.ice').setLevel(logging.ERROR)
logging.getLogger('aiortc').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

# --- Global Variables ---
uv_pipe, tts_model, vad_model = None, None, None
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='audio-proc')
pcs = set()

# --- Optimized HTML Client ---
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Ultra-Low Latency Voice AI</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 0; padding: 20px; background: #1a1a1a; color: #fff; 
               display: flex; align-items: center; justify-content: center; min-height: 100vh; }
        .container { background: #2d2d2d; padding: 40px; border-radius: 15px; 
                     box-shadow: 0 20px 40px rgba(0,0,0,0.3); text-align: center; 
                     max-width: 700px; width: 100%; border: 1px solid #404040; }
        h1 { margin-bottom: 30px; font-weight: 300; color: #00ff88; }
        .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
        .metric { background: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #404040; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00ff88; }
        .metric-label { font-size: 12px; color: #888; }
        button { background: linear-gradient(45deg, #00ff88, #00cc66); color: #000; 
                 border: none; padding: 15px 30px; font-size: 18px; border-radius: 8px; 
                 cursor: pointer; margin: 10px; transition: all 0.3s; font-weight: 600; }
        button:hover { transform: translateY(-2px); box-shadow: 0 8px 16px rgba(0,255,136,0.3); }
        button:disabled { background: #555; cursor: not-allowed; transform: none; }
        .stop-btn { background: linear-gradient(45deg, #ff4444, #cc0000); color: #fff; }
        .status { margin: 20px 0; padding: 15px; border-radius: 8px; font-weight: 500; 
                  transition: all 0.3s; border: 1px solid #404040; }
        .status.connected { background: #004d00; border-color: #00ff88; }
        .status.disconnected { background: #4d0000; border-color: #ff4444; }
        .status.connecting { background: #4d4d00; border-color: #ffff00; }
        .status.processing { background: #00004d; border-color: #4488ff; 
                            animation: pulse 1s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(68,136,255,0.7); } 
                          70% { box-shadow: 0 0 0 10px rgba(68,136,255,0); } 
                          100% { box-shadow: 0 0 0 0 rgba(68,136,255,0); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Ultra-Low Latency Voice AI</h1>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="latency">--</div>
                <div class="metric-label">Latency (ms)</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="quality">--</div>
                <div class="metric-label">Audio Quality</div>
            </div>
        </div>
        <div class="controls">
            <button id="startBtn" onclick="start()">START CONVERSATION</button>
            <button id="stopBtn" onclick="stop()" class="stop-btn" disabled>STOP</button>
        </div>
        <div id="status" class="status disconnected">⚡ Ready to Connect</div>
        <audio id="remoteAudio" autoplay playsinline></audio>
    </div>

<script>
    let pc, ws, localStream, startTime;
    const remoteAudio = document.getElementById('remoteAudio');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDiv = document.getElementById('status');
    const latencyDiv = document.getElementById('latency');
    const qualityDiv = document.getElementById('quality');

    // Performance monitoring
    let audioContext, analyser, dataArray;
    let lastLatencyMeasure = 0;

    function updateStatus(message, className) {
        statusDiv.textContent = message;
        statusDiv.className = `status ${className}`;
    }

    function measureLatency() {
        const currentTime = Date.now();
        const latency = currentTime - startTime;
        latencyDiv.textContent = latency;
        return latency;
    }

    function updateAudioQuality() {
        if (!analyser) return;
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
        const quality = Math.min(100, Math.round(average / 2.55));
        qualityDiv.textContent = quality + '%';
    }

    async function start() {
        if (ws || pc) stop();
        startBtn.disabled = true;
        updateStatus('🔄 Connecting...', 'connecting');
        
        try {
            // Request high-quality, low-latency audio
            localStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    channelCount: 1,
                    sampleRate: 48000,
                    latency: 0.01 // Request 10ms latency
                }
            });

            // Setup audio context for monitoring
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            dataArray = new Uint8Array(analyser.frequencyBinCount);
            
            const source = audioContext.createMediaStreamSource(localStream);
            source.connect(analyser);

            // Configure RTCPeerConnection for low latency
            pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
                iceCandidatePoolSize: 10
            });

            // Add local stream with optimized settings
            localStream.getTracks().forEach(track => {
                const sender = pc.addTrack(track, localStream);
                // Optimize encoding parameters for low latency
                if (track.kind === 'audio') {
                    sender.setParameters({
                        encodings: [{
                            maxBitrate: 64000,
                            priority: 'high'
                        }]
                    });
                }
            });

            pc.ontrack = e => {
                console.log('Remote audio track received');
                if (remoteAudio.srcObject !== e.streams[0]) {
                    remoteAudio.srcObject = e.streams[0];
                    remoteAudio.play().catch(err => console.error("Playback error:", err));
                    
                    remoteAudio.onplaying = () => {
                        const latency = measureLatency();
                        updateStatus(`🤖 AI Response (${latency}ms)`, 'processing');
                    };
                    
                    remoteAudio.onended = () => {
                        updateStatus('🎤 Listening...', 'connected');
                    };
                }
            };

            pc.onicecandidate = e => {
                if (e.candidate && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'ice-candidate',
                        candidate: e.candidate.toJSON()
                    }));
                }
            };

            pc.onconnectionstatechange = () => {
                const state = pc.connectionState;
                console.log('Connection state:', state);
                if (state === 'connected') {
                    updateStatus('🎤 Listening...', 'connected');
                    stopBtn.disabled = false;
                    // Start quality monitoring
                    setInterval(updateAudioQuality, 100);
                } else if (state === 'failed' || state === 'closed' || state === 'disconnected') {
                    stop();
                }
            };

            // WebSocket connection
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onopen = async () => {
                console.log('WebSocket connected');
                const offer = await pc.createOffer({
                    offerToReceiveAudio: true,
                    voiceActivityDetection: false // Disable VAD for lower latency
                });
                await pc.setLocalDescription(offer);
                startTime = Date.now();
                ws.send(JSON.stringify({
                    type: 'offer',
                    sdp: offer.sdp
                }));
            };

            ws.onmessage = async e => {
                const data = JSON.parse(e.data);
                if (data.type === 'answer' && !pc.currentRemoteDescription) {
                    await pc.setRemoteDescription(new RTCSessionDescription(data));
                }
            };

            ws.onclose = ws.onerror = () => {
                if (pc && pc.connectionState !== 'closed') stop();
            };

        } catch (err) {
            console.error('Setup error:', err);
            updateStatus(`❌ Error: ${err.message}`, 'disconnected');
            stop();
        }
    }

    function stop() {
        if (ws) {
            ws.onclose = ws.onerror = null;
            ws.close();
            ws = null;
        }
        if (pc) {
            pc.onconnectionstatechange = null;
            pc.onicecandidate = null;
            pc.ontrack = null;
            pc.close();
            pc = null;
        }
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        
        updateStatus('⚡ Ready to Connect', 'disconnected');
        startBtn.disabled = false;
        stopBtn.disabled = true;
        latencyDiv.textContent = '--';
        qualityDiv.textContent = '--';
    }

    // Cleanup on page unload
    window.addEventListener('beforeunload', stop);
</script>
</body>
</html>
"""

# --- Enhanced VAD with Optimizations ---
class OptimizedSileroVAD:
    def __init__(self):
        try:
            logger.info("Loading optimized Silero VAD...")
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            (self.get_speech_timestamps, _, _, _, _) = utils
            self.speech_threshold = 0.5
            self.min_speech_duration_ms = 150  # Reduced for lower latency
            logger.info("✅ Optimized VAD loaded successfully")
            
        except Exception as e:
            logger.error(f"VAD loading failed: {e}")
            self.model = None

    def detect_speech(self, audio_tensor, sample_rate=16000):
        if self.model is None:
            return True
        
        try:
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor)
            
            if torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
            
            # Quick energy check first
            if audio_tensor.abs().max() < 0.01:
                return False
            
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, sample_rate).item()
                return speech_prob > self.speech_threshold
                
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return True

# --- Optimized Audio Buffer ---
class UltraLowLatencyBuffer:
    def __init__(self, target_duration=0.5, sample_rate=16000):  # Reduced from 3.0s
        self.sample_rate = sample_rate
        self.target_samples = int(target_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.target_samples)
        self.min_speech_samples = int(0.2 * sample_rate)  # Reduced from 0.4s
        self.last_process_time = time.time()
        self.process_interval = 0.1  # Reduced from 0.5s
        self.energy_threshold = 0.002  # Reduced threshold
        
    def add_audio(self, audio_data):
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
        
        # Normalize to prevent clipping
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
            
        self.buffer.extend(audio_data.flatten())

    def get_audio_array(self):
        return np.array(list(self.buffer), dtype=np.float32)
    
    def should_process(self):
        current_time = time.time()
        if (len(self.buffer) >= self.min_speech_samples and 
            (current_time - self.last_process_time) >= self.process_interval):
            
            self.last_process_time = current_time
            audio_array = self.get_audio_array()
            
            # Quick energy check
            if np.abs(audio_array).max() < self.energy_threshold:
                return False
                
            # VAD check
            return vad_model.detect_speech(audio_array, self.sample_rate)
        return False

    def reset(self):
        self.buffer.clear()

# --- Optimized Audio Track ---
class OptimizedResponseAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=10)  # Limit queue size
        self._current_chunk = None
        self._chunk_pos = 0
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 48000)
        self._frame_size = 480  # Reduced frame size for lower latency

    async def recv(self):
        frame = np.zeros(self._frame_size, dtype=np.int16)
        
        # Try to get new chunk if current one is exhausted
        if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
            try:
                self._current_chunk = await asyncio.wait_for(
                    self._queue.get(), timeout=0.005
                )
                self._chunk_pos = 0
            except asyncio.TimeoutError:
                pass
        
        # Fill frame with audio data
        if self._current_chunk is not None:
            end_pos = min(self._chunk_pos + self._frame_size, len(self._current_chunk))
            chunk_part = self._current_chunk[self._chunk_pos:end_pos]
            frame[:len(chunk_part)] = chunk_part
            self._chunk_pos = end_pos
        
        # Create audio frame
        audio_frame = av.AudioFrame.from_ndarray(
            np.array([frame]), format="s16", layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = 48000
        self._timestamp += self._frame_size
        
        return audio_frame

    async def queue_audio(self, audio_float32):
        if audio_float32.size > 0:
            # Convert to int16 and queue
            audio_int16 = (audio_float32 * 32767).astype(np.int16)
            try:
                await asyncio.wait_for(self._queue.put(audio_int16), timeout=0.01)
            except asyncio.TimeoutError:
                logger.warning("Audio queue full, dropping frame")

# --- Model Initialization ---
def initialize_models():
    global uv_pipe, tts_model, vad_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    logger.info(f"🚀 Initializing models on {device} with {dtype}")
    
    # Initialize VAD first
    vad_model = OptimizedSileroVAD()
    if not vad_model.model:
        logger.error("VAD initialization failed")
        return False
    
    try:
        # Load Ultravox with optimizations
        logger.info("📥 Loading Ultravox pipeline...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=dtype,
            use_fast=True,
            model_kwargs={
                "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else "eager"
            }
        )
        
        # Warm up the model
        dummy_audio = np.random.randn(16000).astype(np.float32)
        with torch.no_grad():
            _ = uv_pipe({
                'audio': dummy_audio,
                'sampling_rate': 16000,
                'turns': []
            }, max_new_tokens=5)
        
        logger.info("✅ Ultravox loaded and warmed up")
        
        # Load TTS model
        logger.info("📥 Loading Kyutai TTS...")
        tts_model = load_model('kyutai/tts-1.6b-en_fr', device=device)
        
        # Warm up TTS
        with torch.no_grad():
            _ = tts_model.sample("Hello", voice="voice_1", temperature=0.8)
        
        logger.info("✅ TTS loaded and warmed up")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🎉 All models initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}", exc_info=True)
        return False

# --- Enhanced Audio Processor ---
class UltraLowLatencyProcessor:
    def __init__(self, output_track: OptimizedResponseAudioTrack, executor: ThreadPoolExecutor):
        self.track = None
        self.buffer = UltraLowLatencyBuffer()
        self.output_track = output_track
        self.executor = executor
        self.task = None
        self.is_processing = False
        self.stats = {
            'total_processed': 0,
            'avg_latency': 0,
            'last_process_time': time.time()
        }

    def add_track(self, track):
        self.track = track

    async def start(self):
        self.task = asyncio.create_task(self._audio_loop())

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def _audio_loop(self):
        logger.info("🎧 Starting ultra-low latency audio processing loop")
        try:
            while True:
                if self.is_processing:
                    # Skip audio input during processing to prevent echo
                    try:
                        await asyncio.wait_for(self.track.recv(), timeout=0.01)
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.001)
                    continue

                try:
                    # Get audio frame
                    frame = await asyncio.wait_for(self.track.recv(), timeout=0.05)
                    
                    # Process audio
                    audio_data = frame.to_ndarray().flatten().astype(np.float32)
                    if frame.format.name == 's16':
                        audio_data = audio_data / 32768.0
                    
                    # Resample if needed (optimized)
                    if frame.sample_rate != 16000:
                        audio_data = librosa.resample(
                            audio_data, 
                            orig_sr=frame.sample_rate, 
                            target_sr=16000,
                            res_type='kaiser_fast'  # Faster resampling
                        )
                    
                    # Add to buffer
                    self.buffer.add_audio(audio_data)
                    
                    # Check if we should process
                    if self.buffer.should_process():
                        audio_to_process = self.buffer.get_audio_array()
                        self.buffer.reset()
                        
                        # Process asynchronously
                        asyncio.create_task(self._process_speech_async(audio_to_process))
                        
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.001)
                except mediastreams.MediaStreamError:
                    logger.info("Media stream ended")
                    break
                except Exception as e:
                    logger.error(f"Audio loop error: {e}")
                    await asyncio.sleep(0.01)
                    
        except asyncio.CancelledError:
            logger.info("Audio processing cancelled")
        except Exception as e:
            logger.error(f"Fatal audio processing error: {e}")
        finally:
            logger.info("Audio processing loop stopped")

    async def _process_speech_async(self, audio_array):
        """Process speech with ultra-low latency optimizations"""
        if self.is_processing:
            return
            
        self.is_processing = True
        start_time = time.time()
        
        try:
            logger.info(f"🧠 Processing {len(audio_array)} samples...")
            
            # Run inference in thread pool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._blocking_inference,
                audio_array
            )
            
            if result is not None and result.size > 0:
                # Queue audio for playback
                await self.output_track.queue_audio(result)
                
                # Calculate and log latency
                latency = (time.time() - start_time) * 1000
                logger.info(f"⚡ Total latency: {latency:.1f}ms")
                
                # Update stats
                self.stats['total_processed'] += 1
                self.stats['avg_latency'] = (
                    self.stats['avg_latency'] * 0.9 + latency * 0.1
                )
                
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
        finally:
            self.is_processing = False
            # Small delay to prevent feedback
            await asyncio.sleep(0.1)

    def _blocking_inference(self, audio_array):
        """Optimized blocking inference for minimal latency"""
        try:
            # Step 1: ASR + LLM (Ultravox)
            asr_start = time.time()
            with torch.no_grad():
                result = uv_pipe({
                    'audio': audio_array,
                    'sampling_rate': 16000,
                    'turns': []
                }, max_new_tokens=30)  # Reduced for faster generation
            
            # Parse response
            response_text = self._parse_response(result).strip()
            if not response_text:
                return None
                
            asr_latency = (time.time() - asr_start) * 1000
            logger.info(f"ASR+LLM: {asr_latency:.1f}ms - '{response_text}'")
            
            # Step 2: TTS Generation
            tts_start = time.time()
            with torch.no_grad():
                wav_output = tts_model.sample(
                    response_text,
                    voice="voice_1",
                    temperature=0.7,
                    top_p=0.9
                )
            
            tts_latency = (time.time() - tts_start) * 1000
            logger.info(f"TTS: {tts_latency:.1f}ms")
            
            # Step 3: Resample for output
            if wav_output.shape[-1] != 48000:
                wav_output = librosa.resample(
                    wav_output.astype(np.float32),
                    orig_sr=22050,  # Typical TTS sample rate
                    target_sr=48000,
                    res_type='kaiser_fast'
                )
            
            return wav_output.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def _parse_response(self, result):
        """Parse Ultravox response efficiently"""
        try:
            if isinstance(result, str):
                return result
            elif isinstance(result, list) and len(result) > 0:
                item = result[0]
                if isinstance(item, str):
                    return item
                elif isinstance(item, dict):
                    return item.get('generated_text', '')
            return ""
        except Exception:
            return ""

# --- WebSocket Handler ---
async def optimized_websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    
    # Configure RTCPeerConnection for low latency
    config = RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(urls="stun:stun1.l.google.com:19302")
    ])
    
    pc = RTCPeerConnection(config)
    pcs.add(pc)
    processor = None

    @pc.on("track")
    def on_track(track):
        nonlocal processor
        logger.info(f"🎧 {track.kind} track received")
        if track.kind == "audio":
            # Create optimized audio track
            output_track = OptimizedResponseAudioTrack()
            pc.addTrack(output_track)
            
            # Create processor
            processor = UltraLowLatencyProcessor(output_track, executor)
            processor.add_track(track)
            
            # Start processing
            asyncio.create_task(processor.start())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        logger.info(f"Connection state: {state}")
        if state in ["failed", "closed", "disconnected"]:
            if pc in pcs:
                pcs.remove(pc)
            await pc.close()

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                
                if data["type"] == "offer":
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                    )
                    
                    # Create answer with optimized settings
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    
                    await ws.send_json({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp
                    })
                    
                elif data["type"] == "ice-candidate" and data.get("candidate"):
                    try:
                        candidate_data = data["candidate"]
                        candidate = RTCIceCandidate(
                            candidate=candidate_data["candidate"],
                            sdpMid=candidate_data.get("sdpMid"),
                            sdpMLineIndex=candidate_data.get("sdpMLineIndex")
                        )
                        await pc.addIceCandidate(candidate)
                    except Exception as e:
                        logger.error(f"ICE candidate error: {e}")
                        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")
        if processor:
            await processor.stop()
        if pc in pcs:
            pcs.remove(pc)
        if pc.connectionState != "closed":
            await pc.close()
    
    return ws

# --- Main Application ---
async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def on_shutdown(app):
    logger.info("🛑 Shutting down server...")
    for pc in list(pcs):
        await pc.close()
    pcs.clear()
    executor.shutdown(wait=True)
    
    # Cleanup GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("✅ Shutdown complete")

async def main():
    # Set process priority for better real-time performance
    try:
        process = psutil.Process()
        process.nice(-10)  # Higher priority
        logger.info("🚀 Set high process priority")
    except Exception as e:
        logger.warning(f"Could not set process priority: {e}")
    
    # Initialize models
    if not initialize_models():
        logger.error("❌ Failed to initialize models")
        return
    
    # Create web application
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', optimized_websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    logger.info("✅ Ultra-Low Latency Voice AI Server Started!")
    logger.info("🚀 Server: http://0.0.0.0:7860")
    logger.info("⚡ Optimized for sub-500ms latency")
    logger.info("🎯 Press Ctrl+C to stop")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())
