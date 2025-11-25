import time
from deep_translator import GoogleTranslator
from src.translation.context_translator import ContextTranslator
from src.utils.config import get_config


class GoogleTranslatorProcess(ContextTranslator):
    """Google Translator with Context Awareness and Retry Logic"""
    
    def __init__(self):
        super().__init__("Google Translation")
        
        config = get_config("translation")
        self.src_lang = config.get("source_lang", "en")
        self.tgt_lang = config.get("target_lang", "vi")
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 0.5  # seconds

    def setup(self):
        """Initialize Google Translator"""
        self.logger.info(f"Initializing Google Translator (deep-translator): {self.src_lang} -> {self.tgt_lang}")
        try:
            self.translator = GoogleTranslator(source=self.src_lang, target=self.tgt_lang)
            
            # Test translation
            test_text = "Hello, how are you?"
            test_result = self._translate_text(test_text)
            self.logger.info(f"Google Translator initialized successfully. Test: '{test_text}' -> '{test_result}'")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Translator: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Don't stop, just retry in loop or handle gracefully
            self.translator = None

    def _translate_text(self, text: str, context: str = "") -> str:
        """
        Translate a single text with optional context
        
        Note: Google Translate API doesn't support context directly,
        but we can add it as a comment for better understanding
        
        Args:
            text: Text to translate
            context: Context string from previous translations (not used by API)
            
        Returns:
            Translated text
        """
        if not hasattr(self, 'translator') or self.translator is None:
            self.logger.warning("Translator not initialized, returning original text")
            return text
        
        # Google Translate doesn't support context injection like local models
        # We just translate the text directly
        # Context is logged for reference but not sent to API
        
        for attempt in range(self.max_retries):
            try:
                translated = self.translator.translate(text)
                return translated
            except Exception as e:
                self.logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"All translation attempts failed, returning original text")
                    return text
        
        return text

    def translate_with_context(self, text: str) -> str:
        """
        Translate text with context (implements ContextTranslator abstract method)
        
        Note: Google Translate API doesn't support context, but we still
        build context string for logging purposes
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        # Get context from buffer (for logging only)
        context = self._build_context_string()
        
        if context:
            self.logger.debug(f"Context (not sent to API): {context[:100]}...")
        
        # Translate without context (Google API limitation)
        translated = self._translate_text(text, context)
        
        return translated
