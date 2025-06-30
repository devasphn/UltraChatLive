# --- Configuration Settings for Real-time S2S Agent ---

# Server Configuration
SERVER_HOST = "0.0.0.0"
# Port can be overridden by environment variable PORT
SERVER_PORT = 7860

# Logging Configuration
LOG_LEVEL = "INFO"  # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'


# --- Model Configuration ---

# Silero VAD (Voice Activity Detection)
VAD_MODEL_REPO = 'snakers4/silero-vad'
VAD_MODEL_NAME = 'silero_vad'
VAD_FORCE_RELOAD = False # Set to True to always re-download VAD model
VAD_ONNX = False         # Set to True to use ONNX version of VAD model

# Ultravox (STT and Language Model for response generation)
ULTRAVOX_MODEL_ID = "fixie-ai/ultravox-v0_4"
ULTRAVOX_TORCH_DTYPE = "float16" # "float16" for faster inference if supported, "auto" or "float32" otherwise

# ChatterboxTTS (Text-to-Speech)
# ChatterboxTTS device is auto-detected (cuda/cpu) but can be specified if needed, e.g., "cuda:0"
TTS_DEVICE = "auto" # "auto", "cuda", "cpu"

# Model loading device (General: "cuda" if available, else "cpu")
# Specific models might override this (e.g., Ultravox uses device_map="auto")
DEFAULT_DEVICE = "auto"


# --- Audio Processing Configuration ---

# AudioBuffer settings
AUDIO_BUFFER_MAX_DURATION_S = 5.0  # Max duration of audio to keep in buffer (seconds)
AUDIO_BUFFER_MIN_SPEECH_S = 1.0  # Min duration of speech to consider for processing (seconds)
AUDIO_BUFFER_PROCESS_INTERVAL_S = 2.0  # How often to check buffer for processing (seconds)
AUDIO_SAMPLE_RATE = 16000 # Expected sample rate for all audio processing

# Silero VAD operational parameters
VAD_THRESHOLD = 0.4               # Sensitivity: lower is more sensitive (0.0 to 1.0)
VAD_MIN_SPEECH_DURATION_MS = 250  # Minimum duration of speech to be detected (milliseconds)
VAD_MIN_SILENCE_DURATION_MS = 300 # Minimum duration of silence after speech to consider it ended (milliseconds)

# AudioStreamTrack frame settings
AUDIO_FRAME_SAMPLES = 1600 # Samples per frame (0.1s at 16kHz) corresponds to 100ms


# --- AI Generation Parameters ---

# ThreadPoolExecutor for blocking model inferences
THREAD_POOL_MAX_WORKERS = 4

# Ultravox generation parameters (for LLM response)
UV_MAX_NEW_TOKENS = 40       # Max new tokens for the LLM response (affects length)
UV_DO_SAMPLE = True          # Whether to use sampling; False for greedy decoding
UV_TEMPERATURE = 0.75        # Softness of sampling (higher is more random)
UV_TOP_P = 0.9               # Nucleus sampling probability
UV_REPETITION_PENALTY = 1.1  # Penalize token repetition

# System prompt for Ultravox
UV_SYSTEM_PROMPT = "You are a friendly and concise voice assistant. Respond in a natural, conversational way, typically in under 25 words."

# TTS Synthesis
TTS_EMPTY_RESPONSE_DURATION_S = 0.1 # Duration of silence for empty TTS input (seconds)

# WebRTC Configuration
# STUN servers for ICE NAT traversal (add more if needed)
STUN_SERVERS = [
    'stun:stun.l.google.com:19302',
    'stun:stun1.l.google.com:19302',
    'stun:stun.services.mozilla.com'
]
