import time
import signal
import sys
from src.utils.pipeline import ProcessManager
from src.utils.config import Config
from src.audio.capture import AudioCaptureProcess
from src.audio.playback import AudioPlaybackProcess
from src.asr.nemo_asr import NeMoASRProcess
from src.tts.edge_tts_engine import EdgeTTSProcess
from src.ui.overlay import OverlayProcess
from src.utils.transcript_writer import TranscriptWriterProcess

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
    
    # 3. Translation (Dynamic engine selection)
    config = Config()
    translation_engine = config.get("translation.engine", "marian")
    
    try:
        if translation_engine == "marian":
            from src.translation.marian_translator import MarianTranslatorProcess
            translator = MarianTranslatorProcess()
            print(f"Using MarianMT translation engine")
        elif translation_engine == "nllb":
            from src.translation.nllb_translator import NLLBTranslatorProcess
            translator = NLLBTranslatorProcess()
            print(f"Using NLLB translation engine")
        elif translation_engine == "google":
            from src.translation.google_translator import GoogleTranslatorProcess
            translator = GoogleTranslatorProcess()
            print(f"Using Google Translate engine")
        else:
            print(f"Unknown translation engine '{translation_engine}', falling back to MarianMT")
            from src.translation.marian_translator import MarianTranslatorProcess
            translator = MarianTranslatorProcess()
    except Exception as e:
        print(f"Failed to load {translation_engine} translator: {e}")
        print("Falling back to Google Translate...")
        from src.translation.google_translator import GoogleTranslatorProcess
        translator = GoogleTranslatorProcess()
    
    pm.add_process(translator)
    
    # 4. TTS
    tts = EdgeTTSProcess()
    pm.add_process(tts)
    
    # 5. Transcript Writer
    transcript_writer = TranscriptWriterProcess()
    pm.add_process(transcript_writer)
    
    # 6. Audio Playback
    playback = AudioPlaybackProcess()
    pm.add_process(playback)
    
    # 7. Overlay UI
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
