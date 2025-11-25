import time
from deep_translator import GoogleTranslator
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config
import queue

class GoogleTranslatorProcess(ProcessBase):
    def __init__(self):
        config = get_config("translation")
        super().__init__("Translation", config)
        
        self.src_lang = config.get("source_lang", "en")
        self.tgt_lang = config.get("target_lang", "vi")
        
        # Queues
        self.input_queue = QueueManager.get_queue("asr_output")
        
        # Create output queues
        self.tts_queue = QueueManager.create_queue("tts_input", maxsize=50)
        self.ui_queue = QueueManager.create_queue("ui_input", maxsize=50)
        
        self.register_input_queue("asr_output", self.input_queue)
        self.register_output_queue("tts_input", self.tts_queue)
        self.register_output_queue("ui_input", self.ui_queue)

    def setup(self):
        self.logger.info(f"Initializing Google Translator (deep-translator): {self.src_lang} -> {self.tgt_lang}")
        try:
            self.translator = GoogleTranslator(source=self.src_lang, target=self.tgt_lang)
            # Test translation
            test = self.translator.translate("Hello")
            self.logger.info(f"Google Translator initialized successfully. Test: Hello -> {test}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Translator: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Don't stop, just retry in loop or handle gracefully
            self.translator = None

    def loop(self):
        try:
            # Get text from ASR
            text = self.input_queue.get(timeout=0.1)
            
            if not text or len(text.strip()) == 0:
                return

            start_time = time.time()
            
            # Translate using deep-translator
            if hasattr(self, 'translator') and self.translator:
                try:
                    translated = self.translator.translate(text)
                except Exception as e:
                    self.logger.error(f"Translation API error: {e}")
                    translated = text
            else:
                self.logger.warning("Translator not initialized, returning original text")
                translated = text
            
            latency = (time.time() - start_time) * 1000
            self.logger.info(f"Translation ({latency:.1f}ms): {len(text)} chars -> {len(translated)} chars")
            
            # Push to output queues
            self.tts_queue.put(translated)
            self.ui_queue.put(translated)
            
        except queue.Empty:
            pass
        except Exception as e:
            import traceback
            self.logger.error(f"Translation Error: {e}")
            self.logger.error(traceback.format_exc())
