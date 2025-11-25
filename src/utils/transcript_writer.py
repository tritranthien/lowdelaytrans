"""
Transcript Writer Process

Handles writing translations to a text file with speaker identification and segment merging.
"""

import os
import time
import datetime
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config
import queue

class TranscriptWriterProcess(ProcessBase):
    def __init__(self):
        config = get_config("transcription")
        super().__init__("TranscriptWriter", config)
        
        self.enabled = config.get("enabled", True)
        self.output_dir = config.get("output_dir", "records")
        self.filename_format = config.get("filename_format", "transcript_%Y%m%d_%H%M%S.txt")
        self.include_original = config.get("include_original", True)
        self.merge_segments = config.get("merge_segments", True)
        self.speaker_timeout = config.get("speaker_timeout", 5.0)
        
        # Buffer for merging segments: {speaker_id, text_parts, start_time, last_update}
        self.current_buffer = None
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Generate filename
        timestamp = datetime.datetime.now().strftime(self.filename_format.replace("transcript_", "").replace(".txt", ""))
        # Simple strftime might not handle the full format string if it has custom chars, 
        # but standard format works. Let's use the config format directly.
        try:
            filename = datetime.datetime.now().strftime(self.filename_format)
        except:
            filename = f"transcript_{int(time.time())}.txt"
            
        self.filepath = os.path.join(self.output_dir, filename)
        
        # Input queue
        self.input_queue = QueueManager.create_queue("transcript_input", maxsize=100)
        self.register_input_queue("transcript_input", self.input_queue)
        
        self.file_handle = None

    def setup(self):
        if self.enabled:
            try:
                self.file_handle = open(self.filepath, "w", encoding="utf-8")
                self.logger.info(f"Writing transcript to: {self.filepath}")
                # Write header
                self.file_handle.write(f"Transcript started at {datetime.datetime.now()}\n")
                self.file_handle.write("="*50 + "\n\n")
                self.file_handle.flush()
            except Exception as e:
                self.logger.error(f"Failed to open transcript file: {e}")
                self.enabled = False

    def loop(self):
        if not self.enabled:
            time.sleep(1)
            return

        try:
            # Check for timeout flush
            if self.current_buffer:
                if time.time() - self.current_buffer["last_update"] > self.speaker_timeout:
                    self._flush_buffer()

            # Get data with timeout to allow periodic flushing
            data = self.input_queue.get(timeout=0.5)
            
            if not data:
                return
                
            # data format: {"text": str, "original": str, "speaker_id": int, "timestamp": float}
            self._process_data(data)
            
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Transcript error: {e}")

    def _process_data(self, data):
        speaker_id = data.get("speaker_id")
        text = data.get("text", "")
        original = data.get("original", "")
        timestamp = data.get("timestamp", time.time())
        
        if not text:
            return

        # If merging is disabled, write immediately
        if not self.merge_segments:
            self._write_entry(speaker_id, text, original, timestamp)
            return

        # Check if we need to flush current buffer
        if self.current_buffer:
            # Flush if speaker changed
            if self.current_buffer["speaker_id"] != speaker_id:
                self._flush_buffer()
        
        # Add to buffer
        if not self.current_buffer:
            self.current_buffer = {
                "speaker_id": speaker_id,
                "text_parts": [],
                "original_parts": [],
                "start_time": timestamp,
                "last_update": time.time()
            }
        
        self.current_buffer["text_parts"].append(text)
        if original:
            self.current_buffer["original_parts"].append(original)
        self.current_buffer["last_update"] = time.time()

    def _flush_buffer(self):
        if not self.current_buffer:
            return
            
        text = " ".join(self.current_buffer["text_parts"])
        original = " ".join(self.current_buffer["original_parts"])
        speaker_id = self.current_buffer["speaker_id"]
        timestamp = self.current_buffer["start_time"]
        
        self._write_entry(speaker_id, text, original, timestamp)
        self.current_buffer = None

    def _write_entry(self, speaker_id, text, original, timestamp):
        if not self.file_handle:
            return
            
        time_str = datetime.datetime.fromtimestamp(timestamp).strftime("[%H:%M:%S]")
        speaker_str = f"[Speaker {speaker_id}]" if speaker_id is not None else "[Unknown]"
        
        entry = f"{time_str} {speaker_str}: {text}\n"
        if self.include_original and original:
            entry += f"{' ' * len(time_str)} {' ' * len(speaker_str)}  (Orig: {original})\n"
            
        try:
            self.file_handle.write(entry)
            self.file_handle.flush()
        except Exception as e:
            self.logger.error(f"Write error: {e}")

    def cleanup(self):
        self._flush_buffer()
        if self.file_handle:
            self.file_handle.write(f"\nTranscript ended at {datetime.datetime.now()}\n")
            self.file_handle.close()
            self.logger.info("Transcript file closed")
