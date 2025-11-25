"""
Test Transcript Writer

Tests the transcript writer's ability to buffer, merge segments, and write to file.
"""

import sys
import os
import time
import shutil

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.transcript_writer import TranscriptWriterProcess
from src.utils.pipeline import QueueManager

def test_transcript_writer():
    print("="*60)
    print("Transcript Writer Test")
    print("="*60)
    
    # Clean up old records for testing
    if os.path.exists("test_records"):
        shutil.rmtree("test_records")
    
    # Override config for testing
    from src.utils.config import Config
    config = Config()
    # Access the singleton instance's _config
    config._config["transcription"] = {
        "enabled": True,
        "output_dir": "test_records",
        "filename_format": "test_transcript.txt",
        "include_original": True,
        "merge_segments": True,
        "speaker_timeout": 1.0  # Short timeout for testing
    }
    
    print("\n1. Initializing writer...")
    writer = TranscriptWriterProcess()
    writer.setup()
    
    # Start writer loop in background (simulated)
    import threading
    stop_event = threading.Event()
    
    def writer_loop():
        while not stop_event.is_set():
            writer.loop()
            time.sleep(0.01)
            
    thread = threading.Thread(target=writer_loop)
    thread.start()
    
    print("✓ Writer started")
    
    # Send test data
    print("\n2. Sending test data...")
    queue = writer.input_queue
    
    # Speaker 1 - Segment 1
    queue.put({
        "text": "Xin chào.",
        "original": "Hello.",
        "speaker_id": 1,
        "timestamp": time.time()
    })
    time.sleep(0.1)
    
    # Speaker 1 - Segment 2 (Should merge)
    queue.put({
        "text": "Bạn khỏe không?",
        "original": "How are you?",
        "speaker_id": 1,
        "timestamp": time.time()
    })
    time.sleep(0.1)
    
    # Speaker 2 - Segment 1 (Should flush Speaker 1 and start new)
    queue.put({
        "text": "Tôi khỏe, cảm ơn.",
        "original": "I'm fine, thanks.",
        "speaker_id": 2,
        "timestamp": time.time()
    })
    time.sleep(0.1)
    
    # Speaker 1 again (Should flush Speaker 2)
    queue.put({
        "text": "Rất vui được gặp bạn.",
        "original": "Nice to meet you.",
        "speaker_id": 1,
        "timestamp": time.time()
    })
    
    # Wait for processing
    time.sleep(0.5)
    
    # Wait for timeout flush (timeout is 1.0s)
    print("\n3. Waiting for timeout flush...")
    time.sleep(1.2)
    
    # Stop writer
    stop_event.set()
    thread.join()
    writer.cleanup()
    print("✓ Writer stopped")
    
    # Verify file content
    print("\n4. Verifying file content...")
    filepath = os.path.join("test_records", "test_transcript.txt")
    
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            print("\n--- File Content ---")
            print(content)
            print("--------------------")
            
            # Checks
            if "Xin chào. Bạn khỏe không?" in content:
                print("✓ Speaker 1 segments merged correctly")
            else:
                print("✗ Speaker 1 segments NOT merged")
                
            if "[Speaker 2]: Tôi khỏe, cảm ơn." in content:
                print("✓ Speaker 2 written correctly")
            else:
                print("✗ Speaker 2 NOT written")
                
            if "(Orig: Hello. How are you?)" in content:
                print("✓ Original text included and merged")
            else:
                print("✗ Original text missing or not merged")
    else:
        print(f"✗ File not found: {filepath}")

if __name__ == "__main__":
    test_transcript_writer()
