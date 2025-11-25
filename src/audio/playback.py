import time
import numpy as np
import sounddevice as sd
import queue
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config

class AudioPlaybackProcess(ProcessBase):
    def __init__(self):
        config = get_config("playback")
        super().__init__("AudioPlayback", config)
        self.device = config.get("device", "default")
        self.buffer_size = config.get("buffer_size", 512)
        self.sample_rate = 24000  # Default for many TTS engines (e.g., Edge TTS)
        
        # Input queue
        self.input_queue = QueueManager.create_queue("audio_playback", maxsize=50)
        self.register_input_queue("audio_playback", self.input_queue)

    def setup(self):
        self.logger.info(f"Starting playback on device: {self.device}")
        
        # We'll open the stream when we receive the first chunk to know the sample rate
        # Or we can assume a fixed rate from TTS config
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.buffer_size
        )
        self.stream.start()

    def loop(self):
        try:
            # Get audio chunk from queue
            audio_chunk = self.input_queue.get(timeout=0.1)
            
            # Write to stream
            if isinstance(audio_chunk, bytes):
                # Convert bytes to float32 numpy array if needed
                # This depends on TTS output format. Edge TTS usually outputs mp3/raw bytes
                # For now assuming float32 array or decoding needed
                pass 
            elif isinstance(audio_chunk, np.ndarray):
                self.stream.write(audio_chunk)
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Playback error: {e}")

    def cleanup(self):
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
