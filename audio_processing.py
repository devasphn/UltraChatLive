import asyncio
import collections
import fractions
import logging
import time
import numpy as np
import av
from aiortc import MediaStreamTrack

import config

# Import models from the ai_models module
# These are expected to be initialized and populated by calling initialize_all_models()
# from ai_models module before AudioStreamTrack is instantiated.
from ai_models import vad_model, OptimizedTTS, uv_pipe


class AudioBuffer:
    """Manages buffering of incoming audio chunks."""
    def __init__(self, max_duration=config.AUDIO_BUFFER_MAX_DURATION_S,
                 sample_rate=config.AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time = time.time()
        # Ensure min_speech_samples uses an int conversion for multiplication
        self.min_speech_samples = int(float(config.AUDIO_BUFFER_MIN_SPEECH_S) * sample_rate)
        self.process_interval = float(config.AUDIO_BUFFER_PROCESS_INTERVAL_S)
        logging.info(f"AudioBuffer initialized: max_samples={self.max_samples}, min_speech_samples={self.min_speech_samples}, process_interval={self.process_interval}s")

    def add_audio(self, audio_data: np.ndarray):
        """Adds audio data (numpy array) to the buffer."""
        if not isinstance(audio_data, np.ndarray):
            logging.warning(f"AudioBuffer.add_audio received non-numpy data: {type(audio_data)}")
            return

        audio_data_flat = audio_data.flatten()
        for sample in audio_data_flat:
            self.buffer.append(sample)
        logging.debug(f"🎵 AudioBuffer size: {len(self.buffer)} samples")

    def get_audio_array(self) -> np.ndarray:
        """Returns the current buffer content as a float32 numpy array."""
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.array(list(self.buffer), dtype=np.float32)

    def should_process(self) -> bool:
        """Determines if the buffered audio should be processed."""
        global vad_model # Uses the globally loaded VAD model
        if vad_model is None or vad_model.model is None:
            logging.warning("VAD model not available in AudioBuffer.should_process. Assuming speech.")
            # Fallback behavior if VAD isn't ready, might process more often
            return len(self.buffer) >= self.min_speech_samples


        current_time = time.time()
        has_enough_audio = len(self.buffer) >= self.min_speech_samples
        enough_time_passed = (current_time - self.last_process_time) >= self.process_interval

        if has_enough_audio and enough_time_passed:
            audio_array = self.get_audio_array()
            if audio_array.size == 0:
                return False

            # Use VAD to detect speech
            has_speech = vad_model.detect_speech(audio_array, self.sample_rate)
            logging.info(f"🔍 VAD Check: Buffer {len(audio_array)} samples, Speech detected: {has_speech}")

            if has_speech:
                self.last_process_time = current_time
                return True
        return False

    def reset(self):
        """Clears the audio buffer."""
        self.buffer.clear()
        logging.info("🔄 AudioBuffer reset")


class AudioStreamTrack(MediaStreamTrack):
    """
    A custom MediaStreamTrack for sending synthesized audio back to the client.
    It also handles processing of incoming audio from the client.
    """
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.audio_buffer = AudioBuffer()
        self.tts_processor = OptimizedTTS() # Uses globally loaded tts_model

        self._response_queue = asyncio.Queue() # Queue for TTS audio data (np.ndarray)
        self._current_response_audio: np.ndarray = None
        self._current_response_position: int = 0

        self._timestamp = 0
        self._time_base = fractions.Fraction(1, config.AUDIO_SAMPLE_RATE)

        self._consumer_task = None
        logging.info("AudioStreamTrack initialized.")

    def start_consumer_if_needed(self):
        """Starts the consumer task if it's not running or has finished."""
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._consume_and_send_audio_frames())
            logging.info("🎧 AudioStreamTrack frame consumer task started/restarted.")

    async def _consume_and_send_audio_frames(self):
        """Continuously generates and sends audio frames based on TTS output."""
        try:
            while True:
                # The recv() method is what WebRTC calls to get frames.
                # This loop ensures that recv() is called, effectively pulling data.
                # However, the actual data generation logic is in recv().
                # This task is more about ensuring aiortc's machinery keeps pulling.
                # A MediaBlackhole or similar on the consuming end of this track will drive calls to recv().
                # We just need to ensure this task keeps running to allow recv to be called.
                # A small sleep prevents this from being a super tight loop if nothing is pulling.
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logging.info("🛑 AudioStreamTrack frame consumer task cancelled.")
        except Exception as e:
            logging.error(f"❌ AudioStreamTrack frame consumer task error: {e}", exc_info=True)

    async def recv(self) -> av.AudioFrame:
        """
        Called by the WebRTC peer (e.g., MediaBlackhole) to get the next audio frame.
        This method synthesizes silence or plays back TTS audio from the queue.
        """
        frame_samples = config.AUDIO_FRAME_SAMPLES
        audio_data_for_frame = np.zeros(frame_samples, dtype=np.float32) # Default: silence

        # Check if new TTS audio is available or if current one is finished
        if self._current_response_audio is None or self._current_response_position >= len(self._current_response_audio):
            try:
                # Non-blocking get from queue; if empty, will send silence for this frame.
                self._current_response_audio = self._response_queue.get_nowait()
                self._current_response_position = 0
                logging.info(f"🔊 Starting new TTS response playback: {len(self._current_response_audio)} samples")
                self._response_queue.task_done() # Mark as processed
            except asyncio.QueueEmpty:
                # No new TTS audio, current_response_audio remains None or depleted
                pass
            except Exception as e: # Should not happen with QueueEmpty
                logging.error(f"Error getting from response queue: {e}", exc_info=True)
                self._current_response_audio = None

        # If there's TTS audio to play, segment it for the current frame
        if self._current_response_audio is not None:
            start = self._current_response_position
            end = start + frame_samples
            segment = self._current_response_audio[start:end]

            audio_data_for_frame[:len(segment)] = segment # Copy segment into frame data
            self._current_response_position = end

            if self._current_response_position >= len(self._current_response_audio):
                logging.debug("Finished playing current TTS response.")
                self._current_response_audio = None # Reset for next TTS
                self._current_response_position = 0

        # Create PyAV AudioFrame
        audio_frame = av.AudioFrame.from_ndarray(
            audio_data_for_frame.reshape(1, -1), # Must be 2D (channels, samples)
            format="fltp", # Planar float
            layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.time_base = self._time_base
        audio_frame.sample_rate = config.AUDIO_SAMPLE_RATE

        self._timestamp += frame_samples # Increment PTS for the next frame
        return audio_frame

    def receive_client_audio(self, audio_data_np: np.ndarray):
        """
        Receives audio data from the client (forwarded by WebRTCConnection).
        Buffers it and triggers processing if VAD conditions are met.
        """
        logging.debug(f"📥 AudioStreamTrack received client audio: {audio_data_np.shape}, dtype: {audio_data_np.dtype}")
        self.audio_buffer.add_audio(audio_data_np)

        if self.audio_buffer.should_process():
            audio_to_process = self.audio_buffer.get_audio_array()
            if audio_to_process.size > 0:
                logging.info(f"🎯 Scheduling speech segment for processing: {len(audio_to_process)} samples")
                asyncio.create_task(self._process_speech_segment(audio_to_process))
            self.audio_buffer.reset()

    async def _process_speech_segment(self, audio_array: np.ndarray):
        """Processes a segment of speech: STT -> LLM -> TTS, then queues TTS audio."""
        global uv_pipe # Uses the globally loaded Ultravox pipe
        if uv_pipe is None:
            logging.error("Ultravox pipe (uv_pipe) not available for speech processing.")
            return

        try:
            logging.info("🧠 Performing STT and generating LLM response...")
            # Generate text response from audio using Ultravox
            response_text = await self._generate_llm_response_from_audio(audio_array, uv_pipe)

            if response_text and response_text.strip():
                logging.info(f"💬 AI Response Text: '{response_text}'")
                # Synthesize audio from text response
                response_audio_np = self.tts_processor.synthesize(response_text)
                if response_audio_np.size > 0:
                    await self._response_queue.put(response_audio_np)
                    logging.debug(f"Queued {len(response_audio_np)} samples of TTS audio for playback.")
                else:
                    logging.warning("TTS synthesis resulted in empty audio, not queuing.")
            else:
                logging.warning("⚠️ Empty or whitespace response generated by LLM, nothing to synthesize.")

        except Exception as e:
            logging.error(f"❌ Error in speech segment processing pipeline: {e}", exc_info=True)

    async def _generate_llm_response_from_audio(self, audio_array: np.ndarray, current_uv_pipe) -> str:
        """Generates text response from audio using the provided Ultravox pipeline."""
        try:
            turns = [{"role": "system", "content": config.UV_SYSTEM_PROMPT}]
            logging.info(f"🤖 Calling Ultravox with {len(audio_array)} audio samples...")

            with torch.inference_mode(): # Ensure this is still here if uv_pipe is a global
                result = current_uv_pipe(
                    {'audio': audio_array, 'turns': turns, 'sampling_rate': config.AUDIO_SAMPLE_RATE},
                    max_new_tokens=config.UV_MAX_NEW_TOKENS,
                    do_sample=config.UV_DO_SAMPLE,
                    temperature=config.UV_TEMPERATURE,
                    top_p=config.UV_TOP_P,
                    repetition_penalty=config.UV_REPETITION_PENALTY,
                    pad_token_id=current_uv_pipe.tokenizer.eos_token_id
                )

            response_text = ""
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "generated_text" in result[0]:
                    response_text = result[0]["generated_text"]
                else: response_text = str(result[0])
            elif isinstance(result, dict) and "generated_text" in result:
                response_text = result["generated_text"]
            else: response_text = str(result)

            logging.info(f"✅ Ultravox generated text: '{response_text}'")
            return response_text.strip()

        except Exception as e:
            logging.error(f"❌ Error generating LLM response with Ultravox: {e}", exc_info=True)
            # Consider returning a specific error message or raising to be handled by caller
            return "I'm sorry, I encountered an issue trying to understand that."

if __name__ == '__main__':
    # Basic test for AudioBuffer (requires a VAD model to be available or will fallback)
    from logger_setup import setup_logging
    setup_logging()

    async def test_audio_processing():
        logging.info("Testing Audio Processing module...")
        # Mock VAD model for testing if ai_models.vad_model is None
        class MockVAD:
            class MockModel:
                pass
            model = MockModel() # Simulate that a model object exists
            def detect_speech(self, audio, rate):
                logging.info(f"MockVAD.detect_speech called with {len(audio)} samples at {rate}Hz. Returning True.")
                return True

        global vad_model # Allow modification for test
        if vad_model is None:
            vad_model = MockVAD()
            logging.info("Using MockVAD for AudioBuffer test.")

        buffer = AudioBuffer(sample_rate=16000)
        buffer.min_speech_samples = 1600 # 0.1s for testing
        buffer.process_interval = 0.1 # Process quickly for testing

        # Simulate adding audio data
        for _ in range(5): # Add 0.5s of audio
            buffer.add_audio(np.random.randn(1600).astype(np.float32))
            await asyncio.sleep(0.05) # short sleep
            if buffer.should_process():
                logging.info("Test: AudioBuffer.should_process() is True. Getting audio array.")
                processed_audio = buffer.get_audio_array()
                logging.info(f"Test: Got {len(processed_audio)} samples for processing.")
                buffer.reset()
            else:
                logging.info("Test: AudioBuffer.should_process() is False.")

        logging.info("AudioBuffer test finished.")

        # AudioStreamTrack test would be more involved as it needs a running event loop
        # and potentially a mock WebRTC peer to call recv().
        # For now, this basic buffer test is a start.

    asyncio.run(test_audio_processing())
