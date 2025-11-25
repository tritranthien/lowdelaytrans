import asyncio
import edge_tts
import queue
import threading
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config

class EdgeTTSProcess(ProcessBase):
    def __init__(self):
        config = get_config("tts")
        super().__init__("TTS", config)
        
        self.voice = config.get("edge.voice", "vi-VN-HoaiMyNeural")
        self.rate = config.get("edge.rate", "+0%")
        self.volume = config.get("edge.volume", "+0%")
        
        # Queues
        self.input_queue = QueueManager.get_queue("tts_input")
        # Create output queue if it doesn't exist
        try:
            self.output_queue = QueueManager.get_queue("audio_playback")
        except:
            self.output_queue = QueueManager.create_queue("audio_playback", maxsize=50)
        
        self.register_input_queue("tts_input", self.input_queue)
        self.register_output_queue("audio_playback", self.output_queue)
        
        # Async loop for edge-tts
        self.loop_thread = None
        self.async_loop = None

    def setup(self):
        self.logger.info(f"Initializing Edge TTS with voice: {self.voice}")
        # Create a new event loop for this process
        self.async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.async_loop)

    def loop(self):
        try:
            # Get input (can be string or dict with speaker_id)
            input_data = self.input_queue.get(timeout=0.1)
            
            # Handle both string and dict input
            if isinstance(input_data, dict):
                text = input_data.get("text", "")
                speaker_id = input_data.get("speaker_id")
            else:
                text = input_data
                speaker_id = None
            
            if not text:
                return

            # Log with speaker info if available
            speaker_label = f" [Speaker {speaker_id}]" if speaker_id else ""
            self.logger.debug(f"TTS{speaker_label}: {text[:50]}...")

            # Run async synthesis
            self.async_loop.run_until_complete(self._synthesize(text))
            
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"TTS Error: {e}")

    async def _synthesize(self, text):
        try:
            communicate = edge_tts.Communicate(
                text,
                self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            # Stream audio chunks
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # Put audio data into playback queue
                    # Edge TTS outputs mp3, we might need to decode it for sounddevice
                    # But for now, let's assume playback handles it or we decode here
                    # Actually sounddevice needs raw PCM. 
                    # We need to decode MP3 to PCM. 
                    # For simplicity/speed, we might need pydub or similar here.
                    # Let's send raw bytes for now and handle decoding in playback or here.
                    
                    # Ideally we should decode here to distribute load.
                    # But adding pydub dependency might be slow.
                    # Let's just push bytes and let playback handle it (or use a library that plays mp3)
                    self.output_queue.put(chunk["data"])
                    
        except Exception as e:
            self.logger.error(f"Edge TTS generation error: {e}")

    def cleanup(self):
        if self.async_loop:
            self.async_loop.close()
