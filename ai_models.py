import torch
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
import torch.hub
import logging
import numpy as np

import config # Application configuration

# Global model variables
uv_pipe = None
tts_model = None
vad_model = None

class SileroVAD:
    def __init__(self):
        """Initialize Silero VAD model"""
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir=config.VAD_MODEL_REPO,
                model=config.VAD_MODEL_NAME,
                force_reload=config.VAD_FORCE_RELOAD,
                onnx=config.VAD_ONNX
            )
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils
            logging.info("Silero VAD loaded successfully")
        except Exception as e:
            logging.error(f"Error loading Silero VAD: {e}", exc_info=True)
            self.model = None

    def detect_speech(self, audio_tensor, sample_rate=config.AUDIO_SAMPLE_RATE):
        """Detect speech in audio tensor"""
        if self.model is None:
            logging.warning("VAD model not loaded, assuming speech is present.")
            return True
        try:
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor)
            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()
            if len(audio_tensor.shape) > 1:
                audio_tensor = audio_tensor.squeeze()

            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=sample_rate,
                threshold=config.VAD_THRESHOLD,
                min_speech_duration_ms=config.VAD_MIN_SPEECH_DURATION_MS,
                min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS
            )
            return len(speech_timestamps) > 0
        except Exception as e:
            logging.error(f"VAD error processing audio chunk: {e}", exc_info=True)
            return True

class OptimizedTTS:
    def __init__(self):
        """Initializes OptimizedTTS. Note: TTS model is loaded globally in initialize_models."""
        if config.TTS_DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.TTS_DEVICE
        # This class relies on the global 'tts_model' being initialized.
        if tts_model is None:
            logging.warning("OptimizedTTS initialized before global tts_model. This might indicate an issue if called early.")

    def synthesize(self, text: str) -> np.ndarray:
        """Optimized TTS synthesis returning numpy array"""
        global tts_model # Ensure we are using the globally loaded model
        if tts_model is None:
            logging.error("TTS model not available for synthesis.")
            silence_samples = int(config.TTS_EMPTY_RESPONSE_DURATION_S * config.AUDIO_SAMPLE_RATE)
            return np.zeros(silence_samples, dtype=np.float32)

        if not text.strip():
            logging.warning("TTS synthesize called with empty or whitespace text.")
            silence_samples = int(config.TTS_EMPTY_RESPONSE_DURATION_S * config.AUDIO_SAMPLE_RATE)
            return np.zeros(silence_samples, dtype=np.float32)

        logging.info(f"🎤 Synthesizing TTS for: '{text[:80]}...'")
        try:
            with torch.inference_mode():
                wav = tts_model.generate(text)
            if not isinstance(wav, torch.Tensor): wav = torch.from_numpy(wav)
            if wav.dim() == 1: wav = wav.unsqueeze(0)
            result = wav.cpu().numpy().flatten().astype(np.float32)
            logging.info(f"✅ TTS generated {len(result)} audio samples")
            return result
        except Exception as e:
            logging.error(f"Error during TTS synthesis for text '{text[:80]}...': {e}", exc_info=True)
            silence_samples = int(config.TTS_EMPTY_RESPONSE_DURATION_S * config.AUDIO_SAMPLE_RATE)
            return np.zeros(silence_samples, dtype=np.float32)

def initialize_all_models():
    """Initialize all AI models once at startup and store them in global variables."""
    global uv_pipe, tts_model, vad_model

    if config.DEFAULT_DEVICE == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = config.DEFAULT_DEVICE
    logging.info(f"Using resolved device for model loading: {resolved_device} (based on config.DEFAULT_DEVICE: {config.DEFAULT_DEVICE})")

    logging.info(f"Loading Silero VAD (repo: {config.VAD_MODEL_REPO}, model: {config.VAD_MODEL_NAME})...")
    vad_model = SileroVAD() # Instantiates the VAD class, which loads the model
    if vad_model.model is None:
        logging.critical("Silero VAD failed to load. This is critical. Exiting.")
        exit(1)

    try:
        logging.info(f"Loading Ultravox pipeline ({config.ULTRAVOX_MODEL_ID})...")
        torch_dtype_val = getattr(torch, config.ULTRAVOX_TORCH_DTYPE, torch.float16)
        uv_pipe = pipeline(
            model=config.ULTRAVOX_MODEL_ID,
            trust_remote_code=True,
            device_map="auto", # Ultravox handles its own device mapping
            torch_dtype=torch_dtype_val,
        )
        logging.info(f"Ultravox pipeline loaded successfully with dtype: {torch_dtype_val}.")
    except Exception as e:
        logging.critical(f"Error loading Ultravox pipeline: {e}", exc_info=True)
        exit(1)

    try:
        # Determine device for TTS, prioritizing config then resolved_device
        tts_device_to_use = config.TTS_DEVICE if config.TTS_DEVICE != "auto" else resolved_device
        logging.info(f"Loading Chatterbox TTS (device: {tts_device_to_use})...")
        tts_model = ChatterboxTTS.from_pretrained(device=tts_device_to_use)
        logging.info(f"Chatterbox TTS loaded successfully on device: {tts_device_to_use}.")
    except Exception as e:
        logging.critical(f"Error loading Chatterbox TTS: {e}", exc_info=True)
        exit(1)

    logging.info("All AI models initialized.")
    return vad_model, uv_pipe, tts_model

if __name__ == '__main__':
    # Basic test for model initialization
    from logger_setup import setup_logging
    setup_logging() # Need to setup logging to see output

    logging.info("Testing AI model initialization...")
    v, u, t = initialize_all_models()
    if v and u and t:
        logging.info("Models initialized successfully (test).")
        # Test VAD
        if v.model:
             logging.info("VAD model object exists.")
        # Test Ultravox
        if u:
            logging.info(f"Ultravox model device: {u.device}")
        # Test TTS
        if t:
            # Try a simple synthesis
            try:
                sample_text = "Hello world."
                audio_output = OptimizedTTS().synthesize(sample_text) # Uses global tts_model
                if audio_output.size > 0 :
                    logging.info(f"Test TTS synthesis successful for '{sample_text}', generated {audio_output.size} samples.")
                else:
                    logging.warning(f"Test TTS synthesis for '{sample_text}' produced empty output.")
            except Exception as e_tts:
                logging.error(f"Test TTS synthesis failed: {e_tts}")

    else:
        logging.error("Model initialization failed (test).")
