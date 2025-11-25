import time
import signal
import sys
from src.utils.pipeline import ProcessManager
from src.utils.config import Config
from src.audio.capture import AudioCaptureProcess
from src.audio.playback import AudioPlaybackProcess
from src.asr.nemo_asr import NeMoASRProcess
from src.translation.google_translator import GoogleTranslatorProcess
from src.tts.edge_tts_engine import EdgeTTSProcess
from src.ui.overlay import OverlayProcess

def main():
    print("Initializing Low-Latency Voice Translation System...")
    
    # Initialize config
    Config()
    
    # Create process manager
    pm = ProcessManager()
    
    # 1. Audio Capture (WASAPI Loopback)
    audio_capture = AudioCaptureProcess()
    pm.add_process(audio_capture)
    
    # 2. ASR (NeMo)
    asr = NeMoASRProcess()
    pm.add_process(asr)
    
    # 3. Translation (Google Translate)
    translator = GoogleTranslatorProcess()
    pm.add_process(translator)
    
    # 4. TTS
    tts = EdgeTTSProcess()
    pm.add_process(tts)
    
    # 5. Audio Playback
    playback = AudioPlaybackProcess()
    pm.add_process(playback)
    
    # 6. Overlay UI
    overlay = OverlayProcess()
    pm.add_process(overlay)
    
    # Signal handling
    def signal_handler(sig, frame):
        print("\nShutdown signal received...")
        pm.stop_all()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start all processes
    print("Starting pipeline...")
    pm.start_all()
    
    print("System running. Press Ctrl+C to stop.")
    
    # Monitor loop
    try:
        pm.monitor()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    # Windows support for multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()
