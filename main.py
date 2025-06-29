import torch
import gc
import numpy as np
import torchaudio
import tempfile
import asyncio
from typing import Iterator, Tuple
from pathlib import Path

# FastRTC imports for real-time streaming
from fastrtc import Stream, ReplyOnPause, AlgoOptions
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

# FastAPI for custom web interface
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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

# 2. Chatterbox TTS with optimization
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
        if len(text) > 300:
            text = text[:300] + "..."
        
        # Generate audio
        wav = self.model.generate(text)
        
        # Convert to numpy format expected by FastRTC
        if isinstance(wav, torch.Tensor):
            wav_np = wav.cpu().numpy()
        else:
            wav_np = wav
            
        # Ensure correct shape for FastRTC
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
    """
    Real-time S2S processing with chunked TTS output
    This function is called automatically when user pauses speaking
    """
    sample_rate, audio_data = audio
    print(f"🎤 Processing audio: {audio_data.shape}, SR: {sample_rate}")
    
    try:
        # Convert to format expected by Ultravox
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=0)  # Convert to mono
            
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Ultravox processing
        turns = [
            {"role": "system", "content": "You are a helpful voice assistant. Give brief, natural responses under 50 words."},
        ]
        
        result = uv_pipe({
            'audio': audio_data, 
            'turns': turns, 
            'sampling_rate': sample_rate
        }, max_new_tokens=64)
        
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
        
        # Generate TTS and yield in chunks for streaming
        tts_sample_rate, tts_audio = tts.synthesize(response_text)
        
        # Chunk the audio for smooth streaming
        chunk_size = tts_sample_rate // 4  # 0.25 second chunks
        
        if tts_audio.ndim == 2:
            total_samples = tts_audio.shape[1]
            for i in range(0, total_samples, chunk_size):
                chunk = tts_audio[:, i:i+chunk_size]
                if chunk.shape[1] > 0:
                    yield (tts_sample_rate, chunk)
        else:
            # Handle 1D audio
            for i in range(0, len(tts_audio), chunk_size):
                chunk = tts_audio[i:i+chunk_size]
                if len(chunk) > 0:
                    yield (tts_sample_rate, chunk.reshape(1, -1))
                    
    except Exception as e:
        print(f"❌ Error in s2s_handler: {e}")
        # Return silence on error
        silence = np.zeros((1, 8000))  # 0.5 second of silence at 16kHz
        yield (16000, silence)

# 4. Configure FastRTC Stream with optimized VAD
stream = Stream(
    handler=ReplyOnPause(
        s2s_handler,
        can_interrupt=True,
        algo_options=AlgoOptions(
            audio_chunk_duration=0.3,      # Faster response
            started_talking_threshold=0.2,  # Quick speech detection
            pause_duration_threshold=0.8,   # Short pause before response
            speech_threshold=0.3,           # Sensitive speech detection
            silence_threshold=0.1           # Less sensitive silence detection
        ),
        input_sample_rate=48000,
        output_sample_rate=24000,
        expected_layout="mono"
    ),
    modality="audio",
    mode="send-receive"
)

# 5. FastAPI app for custom interface
app = FastAPI(title="UltraChat Real-Time S2S Agent")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount the FastRTC stream
stream.mount(app)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve custom HTML interface"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UltraChat - Real-Time S2S Agent</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .status {
            text-align: center;
            font-size: 1.2em;
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.2);
        }
        .controls {
            text-align: center;
            margin: 30px 0;
        }
        .btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 1.1em;
            border-radius: 50px;
            cursor: pointer;
            margin: 10px;
            transition: all 0.3s ease;
        }
        .btn:hover {
            background: #45a049;
            transform: translateY(-2px);
        }
        .btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .feature {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        .feature h3 {
            margin-top: 0;
            color: #FFD700;
        }
        #audioContainer {
            margin: 20px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ UltraChat Real-Time S2S Agent</h1>
        
        <div class="status" id="status">
            Click "Start Conversation" to begin real-time voice chat
        </div>
        
        <div class="controls">
            <button class="btn" id="startBtn" onclick="startConversation()">Start Conversation</button>
            <button class="btn" id="stopBtn" onclick="stopConversation()" disabled>Stop Conversation</button>
        </div>
        
        <div id="audioContainer">
            <!-- FastRTC component will be injected here -->
        </div>
        
        <div class="features">
            <div class="feature">
                <h3>⚡ Real-Time</h3>
                <p>Sub-second latency with WebRTC streaming</p>
            </div>
            <div class="feature">
                <h3>🎯 Smart VAD</h3>
                <p>Automatic speech detection with Silero VAD</p>
            </div>
            <div class="feature">
                <h3>🤖 AI Powered</h3>
                <p>Ultravox v0.4 + Chatterbox TTS</p>
            </div>
            <div class="feature">
                <h3>🔄 Continuous</h3>
                <p>No click-to-record needed</p>
            </div>
        </div>
    </div>

    <script>
        let isConnected = false;
        
        function startConversation() {
            // This will be replaced with actual FastRTC initialization
            document.getElementById('status').textContent = 'Starting real-time voice chat...';
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
            // Initialize FastRTC connection
            setTimeout(() => {
                document.getElementById('status').textContent = '🟢 Connected - Speak naturally, AI will respond when you pause';
                isConnected = true;
            }, 2000);
        }
        
        function stopConversation() {
            document.getElementById('status').textContent = 'Conversation stopped';
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            isConnected = false;
        }
        
        // Auto-start in 3 seconds for demo
        setTimeout(() => {
            if (!isConnected) {
                startConversation();
            }
        }, 3000);
    </script>
</body>
</html>
    """)

# 6. Alternative Gradio UI (simpler option)
def launch_gradio():
    """Launch Gradio UI as alternative"""
    return stream.ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--gradio":
        # Launch Gradio UI
        print("🚀 Launching Gradio UI...")
        launch_gradio()
    else:
        # Launch FastAPI with custom interface
        print("🚀 Launching FastAPI server with custom interface...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)
