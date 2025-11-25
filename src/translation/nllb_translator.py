import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.translation.context_translator import ContextTranslator
from src.utils.config import get_config
import torch


class NLLBTranslatorProcess(ContextTranslator):
    """NLLB Translator with Context Awareness"""
    
    def __init__(self):
        super().__init__("NLLB Translation")
        
        config = get_config("translation")
        self.model_name = config.get("nllb.model_name", "facebook/nllb-200-distilled-600M")
        self.device = config.get("nllb.device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = config.get("nllb.max_length", 512)
        self.num_beams = config.get("nllb.num_beams", 1)  # Use 1 for speed
        
        # Language codes for NLLB
        self.src_lang = config.get("nllb.src_lang", "eng_Latn")  # English
        self.tgt_lang = config.get("nllb.tgt_lang", "vie_Latn")  # Vietnamese
        
        # Optimization settings
        opt_config = config.get("optimization", {})
        self.use_fp16 = opt_config.get("use_fp16", True)

    def setup(self):
        """Initialize NLLB model with optimizations"""
        self.logger.info(f"Loading NLLB model: {self.model_name} on {self.device}")
        try:
            # Load tokenizer
            self.logger.info("Initializing tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                src_lang=self.src_lang,
                use_fast=False  # Use slow tokenizer for compatibility
            )
            self.logger.info("Tokenizer loaded successfully")
            
            # Load model
            self.logger.info("Loading model (this may take a while on first run)...")
            dtype = torch.float16 if self.use_fp16 and self.device == "cuda" else torch.float32
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            # Set to eval mode for inference
            self.model.eval()
            
            # Test translation
            test_text = "Hello, how are you?"
            test_result = self._translate_text(test_text)
            self.logger.info(f"NLLB initialized successfully. Test: '{test_text}' -> '{test_result}'")
            
        except Exception as e:
            self.logger.error(f"Failed to load NLLB model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.stop()
            raise

    def _translate_text(self, text: str, context: str = "") -> str:
        """
        Translate a single text with optional context
        
        Args:
            text: Text to translate
            context: Context string from previous translations
            
        Returns:
            Translated text
        """
        # Build input with context if available
        if context and self.context_enabled:
            # Format: "Context: {context} | Current: {text}"
            input_text = f"{context} || {text}"
        else:
            input_text = text
        
        # Tokenize - NLLB tokenizer handles language automatically
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate translation
        # For NLLB, we need to set the target language in generate()
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False  # Deterministic for consistency
            )
        
        # Decode
        translated = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated

    def translate_with_context(self, text: str) -> str:
        """
        Translate text with context (implements ContextTranslator abstract method)
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        # Get context from buffer
        context = self._build_context_string()
        
        # Translate
        translated = self._translate_text(text, context)
        
        return translated
