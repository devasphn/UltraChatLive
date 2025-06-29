import torch
import gc
import numpy as np
import torchaudio
import tempfile
from typing import Iterator, Tuple

from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

print("🚀 Loading UltraChat Real-Time S2S Agent...")

# 1. Initialize models
print("📡 Loading Ultravox pipeline...")
uv_pipe = pipeline(
    model="fixie-ai/ultravox-v0_4",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print("✅ Ultravox loaded successfully")

# 2. TTS class (same as before)
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
        self.load()
        if len(text) > 200:
            text = text[:200] + "..."
        
        wav = self.model.generate(text)
        
        if isinstance(wav, torch.Tensor):
            wav_np = wav.cpu().numpy()
        else:
            wav_np = np.array(wav)
            
        if wav_np.ndim == 1:
            wav_np = wav_np.reshape(1, -1)
        elif wav_np.ndim == 2 and wav_np.shape[0] > wav_np.shape[1]:
            wav_np = wav_np.T
            
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
        return (self.model.sr, wav_np)

tts = OptimizedTTS()

# 3. S2S handler
def s2s_handler(audio: Tuple[int, np.ndarray]) -> Iterator[Tuple[int, np.ndarray]]:
    sample_rate, audio_data = audio
    print(f"🎤 Processing audio: {audio_data.shape}, SR: {sample_rate}")
    
    try:
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=0)
            
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        turns = [
            {"role": "system", "content": "You are a helpful voice assistant. Give very brief responses under 30 words."},
        ]
        
        result = uv_pipe({
            'audio': audio_data, 
            'turns': turns, 
            'sampling_rate': sample_rate
        }, max_new_tokens=50)
        
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        tts_sample_rate, tts_audio = tts.synthesize(response_text)
        
        chunk_size = tts_sample_rate // 5
        
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
        silence = np.zeros((1, 4000))
        yield (16000, silence)

# 4. Configure FastRTC Stream
stream = Stream(
    handler=ReplyOnPause(
        s2s_handler,
        can_interrupt=True,
        algo_options=AlgoOptions(
            audio_chunk_duration=0.2,
            started_talking_threshold=0.1,
            speech_threshold=0.05
        ),
        model_options=SileroVadOptions(
            threshold=0.4,
            min_speech_duration_ms=200,
            min_silence_duration_ms=400
        ),
        input_sample_rate=48000,
        output_sample_rate=24000,
        expected_layout="mono"
    ),
    modality="audio",
    mode="send-receive"
)

# 5. LAUNCH WITH BUILT-IN UI (Key Change)
if __name__ == "__main__":
    print("🚀 Launching UltraChat with FastRTC built-in UI...")
    # Use the built-in UI instead of custom FastAPI
    stream.ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
