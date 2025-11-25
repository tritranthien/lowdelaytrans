import numpy as np
import time
from faster_whisper import WhisperModel
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config
import queue

class WhisperASRProcess(ProcessBase):
    def __init__(self):
        config = get_config("asr")
        super().__init__("ASR", config)
        
        self.model_size = config.get("whisper.model_size", "base")
        self.device = config.get("whisper.device", "cuda")
        self.compute_type = config.get("whisper.compute_type", "float16")
        self.beam_size = config.get("whisper.beam_size", 1)
        
        # Audio buffer for streaming
        self.audio_buffer = []
        self.buffer_duration_ms = 0
        self.min_duration_ms = 2500  # Minimum audio to process (2.5s - balanced)
        self.max_duration_ms = 4000 # Max buffer size
        
        # Queues
        self.input_queue = QueueManager.get_queue("audio_input")
        self.output_queue = QueueManager.create_queue("asr_output", maxsize=50)
        self.register_input_queue("audio_input", self.input_queue)
        self.register_output_queue("asr_output", self.output_queue)
        
        # Sentence segmentation buffer
        self.sentence_buffer = ""
        self.sentence_endings = ['.', '!', '?', '。', '！', '？']  # Multiple language support

    def setup(self):
        self.logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            self.stop()

    def loop(self):
        try:
            # Get audio chunk
            chunk = self.input_queue.get(timeout=0.1)
            
            # Add to buffer
            self.audio_buffer.append(chunk)
            
            # Calculate buffer duration (assuming 16kHz)
            # chunk size 1024 @ 16000Hz = 64ms
            chunk_duration = (len(chunk) / 16000) * 1000
            self.buffer_duration_ms += chunk_duration
            
            # Process if we have enough audio
            if self.buffer_duration_ms >= self.min_duration_ms:
                self._process_buffer()
                
        except queue.Empty:
            # If queue is empty but we have data, process it after a timeout
            if self.buffer_duration_ms > 0:
                # self._process_buffer() # Optional: process remaining buffer on timeout
                pass
        except Exception as e:
            self.logger.error(f"ASR Error: {e}")

    def _process_buffer(self):
        if not self.audio_buffer:
            return
            
        # Concatenate buffer
        audio_data = np.concatenate(self.audio_buffer)
        
        # Clear buffer
        self.audio_buffer = []
        self.buffer_duration_ms = 0
        
        # Run inference
        start_time = time.time()
        segments, info = self.model.transcribe(
            audio_data,
            beam_size=self.beam_size,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5)
        )
        
        # Collect text
        text = " ".join([segment.text for segment in segments]).strip()
        
        latency = (time.time() - start_time) * 1000
        
        if text:
            # Add to sentence buffer
            self.sentence_buffer += " " + text if self.sentence_buffer else text
            self.sentence_buffer = self.sentence_buffer.strip()
            
            # Check if we have a complete sentence
            has_ending = any(self.sentence_buffer.endswith(ending) for ending in self.sentence_endings)
            
            # Output if: (1) has sentence ending, or (2) buffer too long (>100 chars)
            if has_ending or len(self.sentence_buffer) > 100:
                self.logger.info(f"ASR Output ({latency:.1f}ms): {self.sentence_buffer}")
                self.output_queue.put(self.sentence_buffer)
                self.sentence_buffer = ""  # Clear for next sentence
            else:
                # Partial sentence - log but don't output yet
                self.logger.debug(f"ASR Partial ({latency:.1f}ms): {self.sentence_buffer}")
