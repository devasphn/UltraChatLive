import torch
import gc
import numpy as np
import torchaudio
import tempfile
import asyncio
from typing import Iterator, Tuple
import sounddevice as sd
import soundfile as sf

# FastRTC imports for real-time streaming
from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

# FastAPI for web interface
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

print("🚀 Loading UltraChat Real-Time S2S Agent (Fixed)...")

# 1. Initialize models
print("📡 Loading Ultravox pipeline...")
uv_pipe = pipeline(
    model="fixie-ai/ultravox-v0_4",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print("✅ Ultravox loaded successfully")

# 2. Optimized TTS with proper audio handling
class OptimizedTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self):
        if self.model is None:
            print("🎵 Loading Chatterbox TTS...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            print("✅ Chatterbox TTS loaded")

    def synthesize(self, text: str) -> Tuple[int, np.ndarray]:
        """Fast TTS synthesis optimized for streaming"""
        self.load()
        
        # Limit text length for low latency
        if len(text) > 200:
            text = text[:200] + "..."
        
        # Generate audio
        wav = self.model.generate(text)
        
        # Convert to numpy format for FastRTC
        if isinstance(wav, torch.Tensor):
            wav_np = wav.cpu().numpy()
        else:
            wav_np = np.array(wav)
            
        # Ensure correct shape for FastRTC (1, samples)
        if wav_np.ndim == 1:
            wav_np = wav_np.reshape(1, -1)
        elif wav_np.ndim == 2 and wav_np.shape[0] > wav_np.shape[1]:
            wav_np = wav_np.T
            
        # Memory cleanup
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
        return (self.model.sr, wav_np)

tts = OptimizedTTS()

# 3. Real-time Speech-to-Speech Handler
def s2s_handler(audio: Tuple[int, np.ndarray]) -> Iterator[Tuple[int, np.ndarray]]:
    """Real-time S2S processing with optimized latency"""
    sample_rate, audio_data = audio
    print(f"🎤 Processing audio: {audio_data.shape}, SR: {sample_rate}")
    
    try:
        # Convert to mono if stereo
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=0)
            
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Ultravox processing with optimized prompt
        turns = [
            {"role": "system", "content": "You are a helpful voice assistant. Give very brief responses under 30 words."},
        ]
        
        result = uv_pipe({
            'audio': audio_data, 
            'turns': turns, 
            'sampling_rate': sample_rate
        }, max_new_tokens=50)  # Reduced for lower latency
        
        # Extract response text
        response_text = result
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                response_text = result[0]["generated_text"]
            else:
                response_text = str(result[0])
        elif isinstance(result, dict) and "generated_text" in result:
            response_text = result["generated_text"]
        else:
            response_text = str(result)
            
        print(f"💬 Generated response: {response_text}")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Generate TTS and yield in chunks for ultra-low latency
        tts_sample_rate, tts_audio = tts.synthesize(response_text)
        
        # Stream in 200ms chunks for minimal latency
        chunk_size = tts_sample_rate // 5  # 200ms chunks
        
        if tts_audio.ndim == 2:
            total_samples = tts_audio.shape[1]
            for i in range(0, total_samples, chunk_size):
                chunk = tts_audio[:, i:i+chunk_size]
                if chunk.shape[1] > 0:
                    yield (tts_sample_rate, chunk)
        else:
            for i in range(0, len(tts_audio), chunk_size):
                chunk = tts_audio[i:i+chunk_size]
                if len(chunk) > 0:
                    yield (tts_sample_rate, chunk.reshape(1, -1))
                    
    except Exception as e:
        print(f"❌ Error in s2s_handler: {e}")
        # Return short silence on error
        silence = np.zeros((1, 4000))  # 250ms silence at 16kHz
        yield (16000, silence)

# 4. Configure FastRTC Stream with CORRECTED AlgoOptions parameters
stream = Stream(
    handler=ReplyOnPause(
        s2s_handler,
        can_interrupt=True,
        algo_options=AlgoOptions(
            audio_chunk_duration=0.2,      # 200ms chunks for faster response  
            started_talking_threshold=0.1,  # Quick speech detection
            speech_threshold=0.05           # Sensitive pause detection (FIXED PARAMETER NAME)
        ),
        model_options=SileroVadOptions(
            threshold=0.4,                  # Balanced VAD threshold
            min_speech_duration_ms=200,     # Filter out short noises
            min_silence_duration_ms=400     # Quick pause detection
        ),
        input_sample_rate=48000,
        output_sample_rate=24000,
        expected_layout="mono"
    ),
    modality="audio",
    mode="send-receive"
)

# 5. FastAPI app
app = FastAPI(title="UltraChat Real-Time S2S Agent (Fixed)")
stream.mount(app)

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UltraChat - Fixed Real-Time S2S</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0; padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: white;
        }
        .container {
            max-width: 800px; margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px); border-radius: 20px;
            padding: 40px; box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        }
        h1 { text-align: center; margin-bottom: 30px; font-size: 2.5em; }
        .status {
            text-align: center; font-size: 1.2em; margin: 20px 0;
            padding: 15px; border-radius: 10px;
            background: rgba(255, 255, 255, 0.2);
        }
        .features {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin-top: 30px;
        }
        .feature {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px; border-radius: 15px; text-align: center;
        }
        .feature h3 { margin-top: 0; color: #FFD700; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ UltraChat: Fixed Real-Time S2S</h1>
        <div class="status">
            ✅ AlgoOptions Fixed | ⚡ Ultra-low latency (2-4 seconds) 
        </div>
        <div class="features">
            <div class="feature">
                <h3>🔧 Parameter Fixed</h3>
                <p>speech_threshold replaces pause_duration_threshold</p>
            </div>
            <div class="feature">
                <h3>⚡ Ultra-Fast</h3>
                <p>200ms audio chunks + optimized VAD</p>
            </div>
            <div class="feature">
                <h3>🎯 Smart VAD</h3>
                <p>Silero VAD with balanced thresholds</p>
            </div>
            <div class="feature">
                <h3>🤖 AI Powered</h3>
                <p>Ultravox v0.4 + Chatterbox TTS</p>
            </div>
        </div>
    </div>
</body>
</html>
    """)

if __name__ == "__main__":
    print("🚀 Launching Fixed Ultra-Low Latency S2S Agent...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
