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
import threading
from typing import Optional, List

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av

from transformers import pipeline
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel
import torch.hub

# --- Ultra-Low Latency Setup ---
try:
    import uvloop
    uvloop.install()
    print("🚀 Using uvloop for maximum performance")
except ImportError:
    print("⚠️ uvloop not found, performance may be reduced")

# GPU Optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')

# Disable unnecessary warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Silence verbose loggers
for log_name in ['aioice.ice', 'aiortc', 'transformers', 'torch']:
    logging.getLogger(log_name).setLevel(logging.ERROR)

# --- Global Variables ---
uv_pipe, tts_model, vad_model = None, None, None
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio_proc")
pcs = set()

# Memory pool for audio processing
AUDIO_BUFFER_POOL = []
POOL_SIZE = 10

def get_audio_buffer(size: int) -> np.ndarray:
    """Reuse audio buffers to reduce GC pressure"""
    for i, buf in enumerate(AUDIO_BUFFER_POOL):
        if buf.size >= size:
            AUDIO_BUFFER_POOL.pop(i)
            return buf[:size]
    return np.zeros(size, dtype=np.float32)

def return_audio_buffer(buf: np.ndarray):
    """Return buffer to pool"""
    if len(AUDIO_BUFFER_POOL) < POOL_SIZE:
        AUDIO_BUFFER_POOL.append(buf)

# --- HTML Client (same as before) ---
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat Voice Assistant</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 20px; 
               background: #1a1a1a; color: #fff; display: flex; align-items: center; 
               justify-content: center; min-height: 100vh; }
        .container { background: #2d2d2d; padding: 30px; border-radius: 8px; 
                    text-align: center; max-width: 500px; width: 100%; }
        h1 { margin-bottom: 20px; font-weight: 400; color: #00ff88; }
        button { background: #00ff88; color: #000; border: none; padding: 12px 24px; 
                font-size: 16px; border-radius: 4px; cursor: pointer; margin: 8px; 
                transition: all 0.2s; font-weight: 600; }
        button:hover { transform: scale(1.05); }
        button:disabled { background: #666; cursor: not-allowed; transform: none; }
        .stop-btn { background: #ff4444; color: #fff; }
        .status { margin: 15px 0; padding: 10px; border-radius: 4px; font-weight: 500; }
        .status.connected { background: #00ff88; color: #000; }
        .status.disconnected { background: #ff4444; }
        .status.connecting { background: #ffaa00; color: #000; }
        .status.speaking { background: #0088ff; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .latency { font-size: 12px; color: #888; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ UltraChat Voice Assistant</h1>
        <div class="controls">
            <button id="startBtn" onclick="start()">START</button>
            <button id="stopBtn" onclick="stop()" class="stop-btn" disabled>STOP</button>
        </div>
        <div id="status" class="status disconnected">🔌 Disconnected</div>
        <div id="latency" class="latency">Latency: -- ms</div>
        <audio id="remoteAudio" autoplay playsinline></audio>
    </div>
<script>
    let pc, ws, localStream, startTime;
    const remoteAudio = document.getElementById('remoteAudio');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDiv = document.getElementById('status');
    const latencyDiv = document.getElementById('latency');

    function updateStatus(message, className) { 
        statusDiv.textContent = message; 
        statusDiv.className = `status ${className}`; 
    }

    function updateLatency(ms) {
        latencyDiv.textContent = `Latency: ${ms}ms`;
    }

    async function start() {
        if(ws || pc) stop();
        startBtn.disabled = true;
        updateStatus('🔄 Connecting...', 'connecting');
        
        try {
            localStream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    echoCancellation: true, 
                    noiseSuppression: true, 
                    autoGainControl: false,
                    latency: 0.01,
                    sampleRate: 48000,
                    channelCount: 1
                } 
            });
            
            pc = new RTCPeerConnection({ 
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
                bundlePolicy: 'max-bundle',
                rtcpMuxPolicy: 'require'
            });
            
            localStream.getTracks().forEach(track => {
                pc.addTrack(track, localStream);
            });

            pc.ontrack = e => {
                if (remoteAudio.srcObject !== e.streams[0]) {
                    remoteAudio.srcObject = e.streams[0];
                    remoteAudio.play().catch(console.error);
                    
                    remoteAudio.onplaying = () => {
                        if(startTime) {
                            const latency = Date.now() - startTime;
                            updateLatency(latency);
                        }
                        updateStatus('🤖 AI Speaking...', 'speaking');
                    };
                    
                    remoteAudio.onended = () => {
                        updateStatus('✅ Listening...', 'connected');
                    };
                }
            };

            pc.onicecandidate = e => {
                if (e.candidate && ws?.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ 
                        type: 'ice-candidate', 
                        candidate: e.candidate.toJSON() 
                    }));
                }
            };
            
            pc.onconnectionstatechange = () => {
                const state = pc.connectionState;
                if (state === 'connected') {
                    updateStatus('✅ Listening...', 'connected');
                    stopBtn.disabled = false;
                } else if (['failed', 'closed', 'disconnected'].includes(state)) {
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
                if (data.type === 'answer' && !pc.currentRemoteDescription) {
                    await pc.setRemoteDescription(new RTCSessionDescription(data));
                } else if (data.type === 'processing') {
                    startTime = Date.now();
                }
            };

            ws.onclose = ws.onerror = () => stop();

        } catch (err) {
            console.error(err);
            updateStatus(`❌ Error: ${err.message}`, 'disconnected');
            stop();
        }
    }

    function stop() {
        [ws, pc, localStream].forEach(obj => {
            if (obj) {
                if (obj.close) obj.close();
                if (obj.getTracks) obj.getTracks().forEach(track => track.stop());
            }
        });
        ws = pc = localStream = null;
        updateStatus('🔌 Disconnected', 'disconnected');
        updateLatency('--');
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }

    window.addEventListener('load', () => {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        document.addEventListener('click', () => audioContext.resume(), { once: true });
    });
</script>
</body>
</html>
"""

# --- Utility Functions ---
def candidate_from_sdp(candidate_string: str) -> dict:
    """Optimized SDP parsing"""
    if candidate_string.startswith("candidate:"):
        candidate_string = candidate_string[10:]
    
    bits = candidate_string.split()
    if len(bits) < 8:
        raise ValueError(f"Invalid candidate: {candidate_string}")
    
    params = {
        'foundation': bits[0],
        'component': int(bits[1]),
        'protocol': bits[2],
        'priority': int(bits[3]),
        'ip': bits[4],
        'port': int(bits[5]),
        'type': bits[7]
    }
    
    for i in range(8, len(bits) - 1, 2):
        key = bits[i]
        if key == "raddr":
            params['relatedAddress'] = bits[i + 1]
        elif key == "rport":
            params['relatedPort'] = int(bits[i + 1])
    
    return params

def parse_ultravox_response(result) -> str:
    """Fast response parsing"""
    try:
        if isinstance(result, str):
            return result
        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, str):
                return item
            if isinstance(item, dict) and 'generated_text' in item:
                return item['generated_text']
        return ""
    except:
        return ""

# --- Optimized VAD ---
class OptimizedSileroVAD:
    def __init__(self):
        self.model = None
        self.get_speech_timestamps = None
        try:
            logger.info("Loading Silero VAD...")
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.model = model.cuda() if torch.cuda.is_available() else model
            self.model.eval()
            self.get_speech_timestamps = utils[0]
            
            # Warm up the model
            dummy_audio = torch.randn(16000).cuda() if torch.cuda.is_available() else torch.randn(16000)
            with torch.no_grad():
                self.get_speech_timestamps(dummy_audio, self.model, sampling_rate=16000)
            
            logger.info("✅ Silero VAD loaded and warmed up")
        except Exception as e:
            logger.error(f"VAD loading failed: {e}")

    def detect_speech(self, audio_tensor: torch.Tensor, sample_rate: int = 16000) -> bool:
        if self.model is None:
            return True
        
        try:
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor)
            
            if torch.cuda.is_available() and not audio_tensor.is_cuda:
                audio_tensor = audio_tensor.cuda()
            
            if audio_tensor.abs().max() < 0.005:
                return False
            
            with torch.no_grad():
                speech_timestamps = self.get_speech_timestamps(
                    audio_tensor, 
                    self.model, 
                    sampling_rate=sample_rate,
                    min_speech_duration_ms=200,
                    threshold=0.3
                )
            
            return len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True

# --- FIXED MODEL LOADING ---
def initialize_models() -> bool:
    global uv_pipe, tts_model, vad_model
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    logger.info(f"🚀 Initializing models on {device} with {torch_dtype}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    try:
        # 1. Load VAD first
        vad_model = OptimizedSileroVAD()
        if vad_model.model is None:
            return False
        
        # 2. Load Ultravox
        logger.info("📥 Loading Ultravox...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_fast_tokenizer=True
        )
        
        # Warm up Ultravox
        dummy_audio = np.random.randn(16000).astype(np.float32)
        with torch.no_grad():
            uv_pipe({
                'audio': dummy_audio, 
                'sampling_rate': 16000, 
                'turns': []
            }, max_new_tokens=10)
        
        logger.info("✅ Ultravox loaded and warmed up")
        
        # 3. Load TTS model - FIXED VERSION
        logger.info("📥 Loading Kyutai TTS...")
        checkpoint_info = CheckpointInfo.from_hf_repo('kyutai/tts-1.6B-en_fr')
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info,
            device=torch.device(device),
            dtype=torch_dtype
        )
        
        # FIXED: Warm up TTS with proper API handling
        logger.info("🔥 Warming up TTS model...")
        try:
            with torch.no_grad():
                # Test with a simple word
                test_text = "Hello"
                
                # Prepare script - returns a single Entry or list of Entry objects
                entries = tts_model.prepare_script([test_text])
                
                # Get voice path
                voice_path = tts_model.get_voice_path("expresso/ex03-ex01_happy_001_channel1_334s.wav")
                
                # Create condition attributes
                condition_attributes = tts_model.make_condition_attributes([voice_path])
                
                # Generate - handle both single Entry and list cases
                if hasattr(entries, '__iter__') and not isinstance(entries, str):
                    # entries is iterable (list of Entry objects)
                    results_list = tts_model.generate(list(entries), [condition_attributes])
                else:
                    # entries is a single Entry object
                    results_list = tts_model.generate([entries], [condition_attributes])
                
                logger.info("✅ TTS model warmed up successfully")
        
        except Exception as warmup_error:
            logger.warning(f"TTS warmup failed, but model loaded: {warmup_error}")
            # Continue anyway - the model is loaded, warmup just failed
        
        logger.info("✅ TTS loaded successfully")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🎉 All models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

# --- Audio Processing Classes ---
class OptimizedAudioBuffer:
    def __init__(self, max_duration: float = 2.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.last_process_time = time.time()
        self.min_speech_samples = int(0.3 * sample_rate)
        self.process_interval = 0.3
    
    def add_audio(self, audio_data: np.ndarray):
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.dtype == np.int16:
                audio_data /= 32768.0
        
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data /= max_val
        
        audio_flat = audio_data.flatten()
        samples_to_add = len(audio_flat)
        
        if self.write_pos + samples_to_add <= self.max_samples:
            self.buffer[self.write_pos:self.write_pos + samples_to_add] = audio_flat
            self.write_pos += samples_to_add
        else:
            remaining = self.max_samples - self.write_pos
            self.buffer[self.write_pos:] = audio_flat[:remaining]
            self.buffer[:samples_to_add - remaining] = audio_flat[remaining:]
            self.write_pos = samples_to_add - remaining

    def get_audio_array(self) -> np.ndarray:
        if self.write_pos < self.max_samples:
            return self.buffer[:self.write_pos].copy()
        else:
            return self.buffer.copy()
    
    def should_process(self) -> bool:
        current_time = time.time()
        
        if (self.write_pos > self.min_speech_samples and 
            (current_time - self.last_process_time) > self.process_interval):
            
            self.last_process_time = current_time
            
            recent_samples = min(self.write_pos, self.min_speech_samples)
            recent_audio = self.buffer[max(0, self.write_pos - recent_samples):self.write_pos]
            
            if np.abs(recent_audio).max() < 0.003:
                return False
            
            audio_tensor = torch.from_numpy(recent_audio)
            return vad_model.detect_speech(audio_tensor, self.sample_rate)
        
        return False

    def reset(self):
        self.write_pos = 0

class UltraFastResponseTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=100)
        self._current_chunk = None
        self._chunk_pos = 0
        self._timestamp = 0
        self._frame_size = 960
    
    async def recv(self):
        frame = np.zeros(self._frame_size, dtype=np.int16)
        
        if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
            try:
                self._current_chunk = await asyncio.wait_for(
                    self._queue.get(), timeout=0.005
                )
                self._chunk_pos = 0
            except asyncio.TimeoutError:
                pass
        
        if self._current_chunk is not None:
            end_pos = self._chunk_pos + self._frame_size
            chunk_part = self._current_chunk[self._chunk_pos:end_pos]
            frame[:len(chunk_part)] = chunk_part
            self._chunk_pos += len(chunk_part)
        
        audio_frame = av.AudioFrame.from_ndarray(
            np.array([frame]), format="s16", layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = 48000
        self._timestamp += self._frame_size
        
        return audio_frame
    
    async def queue_audio(self, audio_float32: np.ndarray):
        if audio_float32.size > 0:
            audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
            try:
                self._queue.put_nowait(audio_int16)
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(audio_int16)
                except asyncio.QueueEmpty:
                    pass

class UltraFastAudioProcessor:
    def __init__(self, output_track: UltraFastResponseTrack, executor: ThreadPoolExecutor):
        self.track = None
        self.buffer = OptimizedAudioBuffer()
        self.output_track = output_track
        self.task = None
        self.executor = executor
        self.is_speaking = False
        self.processing_lock = asyncio.Lock()
    
    def add_track(self, track):
        self.track = track
    
    async def start(self):
        self.task = asyncio.create_task(self._run())
    
    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
    
    async def _run(self):
        try:
            while True:
                if self.is_speaking:
                    try:
                        await asyncio.wait_for(self.track.recv(), timeout=0.005)
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.001)
                    continue
                
                try:
                    frame = await self.track.recv()
                except mediastreams.MediaStreamError:
                    logger.warning("Media stream ended")
                    break
                
                audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
                
                if frame.sample_rate != 16000:
                    audio_data = librosa.resample(
                        audio_data, 
                        orig_sr=frame.sample_rate, 
                        target_sr=16000,
                        res_type='kaiser_fast'
                    )
                
                self.buffer.add_audio(audio_data)
                
                if self.buffer.should_process() and not self.processing_lock.locked():
                    audio_to_process = self.buffer.get_audio_array()
                    self.buffer.reset()
                    
                    asyncio.create_task(self._process_speech_async(audio_to_process))
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Audio processor error: {e}")
        finally:
            logger.info("Audio processor stopped")
    
    async def _process_speech_async(self, audio_array: np.ndarray):
        async with self.processing_lock:
            try:
                self.is_speaking = True
                
                loop = asyncio.get_running_loop()
                result_audio = await loop.run_in_executor(
                    self.executor, 
                    self._ultra_fast_asr_llm_tts, 
                    audio_array
                )
                
                if result_audio.size > 0:
                    await self.output_track.queue_audio(result_audio)
                    
                    playback_duration = result_audio.size / 48000
                    await asyncio.sleep(playback_duration + 0.05)
            
            except Exception as e:
                logger.error(f"Speech processing error: {e}")
            finally:
                self.is_speaking = False
                self.buffer.reset()
    
    def _ultra_fast_asr_llm_tts(self, audio_array: np.ndarray) -> np.ndarray:
        """FIXED TTS generation pipeline"""
        try:
            # 1. ASR + LLM (Ultravox)
            with torch.inference_mode():
                result = uv_pipe({
                    'audio': audio_array,
                    'sampling_rate': 16000,
                    'turns': []
                }, max_new_tokens=30)
            
            response_text = parse_ultravox_response(result).strip()
            if not response_text:
                return np.array([], dtype=np.float32)
            
            logger.info(f"Response: '{response_text}'")
            
            # 2. TTS (Kyutai/Moshi) - FIXED VERSION
            with torch.inference_mode():
                # Prepare script
                entries = tts_model.prepare_script([response_text])
                
                # Get voice path
                voice_path = tts_model.get_voice_path(
                    "expresso/ex03-ex01_happy_001_channel1_334s.wav"
                )
                
                # Create condition attributes
                condition_attributes = tts_model.make_condition_attributes([voice_path])
                
                # Generate with proper handling
                if hasattr(entries, '__iter__') and not isinstance(entries, str):
                    # entries is iterable (list of Entry objects)
                    results_list = tts_model.generate(list(entries), [condition_attributes])
                else:
                    # entries is a single Entry object
                    results_list = tts_model.generate([entries], [condition_attributes])
                
                # Get the result
                result = results_list[0]
                wav = result.wav
                sr = result.sample_rate
                
                # Resample to 48kHz if needed
                if sr != 48000:
                    wav = librosa.resample(
                        wav.astype(np.float32), 
                        orig_sr=sr, 
                        target_sr=48000,
                        res_type='kaiser_fast'
                    )
                
                return wav.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Processing pipeline error: {e}")
            return np.array([], dtype=np.float32)

# --- WebSocket Handler ---
async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30, compress=False)
    await ws.prepare(request)
    
    pc = RTCPeerConnection(RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302")
    ]))
    pcs.add(pc)
    processor = None
    
    @pc.on("track")
    def on_track(track):
        nonlocal processor
        logger.info(f"Track {track.kind} received")
        if track.kind == "audio":
            output_track = UltraFastResponseTrack()
            pc.addTrack(output_track)
            processor = UltraFastAudioProcessor(output_track, executor)
            processor.add_track(track)
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
                    await pc.setRemoteDescription(RTCSessionDescription(
                        sdp=data["sdp"], type=data["type"]
                    ))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({
                        "type": "answer", 
                        "sdp": pc.localDescription.sdp
                    })
                
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
                        logger.error(f"ICE candidate error: {e}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if processor:
            await processor.stop()
        if pc in pcs:
            pcs.remove(pc)
        if pc.connectionState != "closed":
            await pc.close()
    
    return ws

async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type='text/html')

# --- Main Application ---
async def on_shutdown(app):
    logger.info("Shutting down...")
    for pc in list(pcs):
        await pc.close()
    pcs.clear()
    executor.shutdown(wait=True)

async def main():
    try:
        import os
        os.nice(-10)
    except:
        pass
    
    if not initialize_models():
        logger.error("Failed to initialize models")
        return
    
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    print("✅ Ultra-low latency server started on http://0.0.0.0:7860")
    print("⚡ Optimized for A40 GPU performance")
    print("🎯 Target latency: <500ms end-to-end")
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
