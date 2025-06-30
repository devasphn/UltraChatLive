# Real-Time Speech-to-Speech Agent

This project implements a real-time speech-to-speech (S2S) agent using Python. It captures audio input via WebRTC, transcribes it to text, generates a conversational response using a language model, synthesizes this response back to audio, and streams it to the client.

## Features

*   **Real-time Audio Streaming:** Uses WebRTC (via `aiortc`) for bidirectional audio streaming between a web client and the Python server.
*   **Voice Activity Detection (VAD):** Employs Silero VAD to detect speech segments in the incoming audio, optimizing processing.
*   **Speech-to-Text (STT):** Utilizes the `fixie-ai/ultravox-v0_4` model from Hugging Face Transformers for audio transcription.
*   **Conversational AI Response:** The same `fixie-ai/ultravox-v0_4` model is used to generate a textual response based on the transcribed input and a system prompt.
*   **Text-to-Speech (TTS):** Uses `chatterbox-tts` to synthesize the AI's textual response into audible speech.
*   **Async Architecture:** Built with `asyncio` and `aiohttp` for efficient handling of concurrent operations and network requests.
*   **Configurable:** Key parameters for models, audio processing, and server settings are managed via a `config.py` file.
*   **Modular Code Structure:** The codebase is organized into logical modules for better readability and maintainability.

## Models Used

*   **VAD:** `snakers4/silero-vad` (loaded via `torch.hub`)
*   **STT & Language Model:** `fixie-ai/ultravox-v0_4` (from Hugging Face Transformers)
*   **TTS:** `chatterbox-tts`

## Project Structure

```
.
├── config.py               # Configuration settings
├── main.py                 # Main application entry point
├── ai_models.py            # AI model loading and helper classes (VAD, TTS)
├── audio_processing.py     # Audio buffering and custom WebRTC audio track
├── webrtc_server.py        # WebRTC connection management and WebSocket signaling
├── logger_setup.py         # Logging configuration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup Instructions

### Prerequisites

*   Python 3.9 or higher.
*   `pip` for package installation.
*   A CUDA-enabled GPU is strongly recommended for optimal performance, especially for the Ultravox and TTS models. The application will attempt to run on CPU if a GPU is not available, but expect significantly higher latency.
*   Microphone access for your web browser.

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository-url>
    # cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary packages, including PyTorch (select the correct version for your CUDA setup if not using CPU-only). The `requirements.txt` specifies `torch==2.6.0`. If you need a different PyTorch version for your CUDA toolkit, you might need to adjust this or install it manually first.

## Running the Application

1.  **Start the server:**
    ```bash
    python main.py
    ```

2.  **Access the web interface:**
    Open your web browser and navigate to `http://localhost:7860` (or the port specified in `config.py` or via the `PORT` environment variable).

3.  **Usage:**
    *   Click the "Start Conversation" button.
    *   Allow microphone access when prompted by your browser.
    *   Speak clearly for a few seconds. The system uses VAD to determine when you've finished speaking (or when the audio buffer is due for processing).
    *   Wait for the AI to process your speech and generate a response. This can take a few seconds depending on your hardware.
    *   The AI's voice response will be streamed back and played automatically.
    *   Click "Stop Conversation" to end the session.

## Configuration

Key parameters can be modified in `config.py`. This includes:
*   Server host and port (`SERVER_HOST`, `SERVER_PORT`).
*   Logging settings (`LOG_LEVEL`, `LOG_FORMAT`).
*   Model identifiers and settings (VAD, Ultravox, TTS device).
*   Audio processing parameters (buffer durations, VAD thresholds, sample rates).
*   AI generation parameters (max tokens for LLM, temperature, system prompt).
*   STUN server list for WebRTC.

The server port can also be overridden by setting the `PORT` environment variable.

## How It Works

1.  The client (web browser) connects to the server via WebSocket for WebRTC signaling.
2.  A WebRTC peer connection is established. The client streams audio from its microphone to the server.
3.  The server's `AudioStreamTrack` (via `WebRTCConnection`) receives incoming audio frames.
4.  This audio is passed to an `AudioBuffer`.
5.  The `AudioBuffer`, using Silero VAD, detects segments of speech.
6.  When a speech segment is ready, it's sent to the `ai_models` pipeline:
    *   `fixie-ai/ultravox-v0_4` performs STT.
    *   The same Ultravox model generates a textual response based on the STT output and the system prompt defined in `config.UV_SYSTEM_PROMPT`.
7.  The generated text is synthesized into audio using `chatterbox-tts`.
8.  The synthesized audio (as a NumPy array) is queued in the `AudioStreamTrack`.
9.  The `AudioStreamTrack.recv()` method, called by the WebRTC connection, dequeues this audio and sends it frame by frame back to the client, where it's played.

## Troubleshooting & Known Issues

*   **Microphone Permissions:** Ensure your browser has permission to access your microphone for the site.
*   **Latency:** Performance, especially latency, is highly dependent on hardware (CPU/GPU) and network conditions. Running models on a GPU significantly improves speed.
*   **Network Configuration (ICE/STUN/TURN):** WebRTC can be sensitive to network configurations (firewalls, NATs). The application is configured to use public STUN servers (`config.STUN_SERVERS`). For more restrictive networks, a TURN server might be necessary (not currently configured in this project).
*   **Model Download:** The first time you run the application, the AI models will be downloaded and cached. This might take some time.
*   **Error Logs:** Check the server console output for detailed logs, including any errors during model loading or processing. The log level can be adjusted in `config.py`.

## Potential Future Enhancements

*   **Conversation History:** Implement context management for multi-turn conversations.
*   **Choice of Different ASR/TTS/LLM Models:** Make model selection more flexible.
*   **HTML Client from File:** Load `HTML_CLIENT` from a separate `.html` file using a templating engine for cleaner separation.
*   **TURN Server Configuration:** Add support for TURN servers for improved WebRTC NAT traversal.
*   **More Robust Client-Side Error Handling:** Enhance the JavaScript client with more detailed status updates and error recovery.
*   **Unit and Integration Tests:** Add a test suite for better reliability.
```
