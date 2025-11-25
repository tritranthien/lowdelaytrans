import time
import torch
from transformers import MarianMTModel, MarianTokenizer
from src.translation.context_translator import ContextTranslator
from src.utils.config import get_config


class MarianTranslatorProcess(ContextTranslator):
    """MarianMT Translator with Context Awareness"""
    
    def __init__(self):
        super().__init__("MarianMT Translation")
        
        config = get_config("translation")
        self.model_name = config.get("marian.model_name", "Helsinki-NLP/opus-mt-en-vi")
        self.device = config.get("marian.device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = config.get("marian.max_length", 200)
        self.num_beams = config.get("marian.num_beams", 1)  # Use 1 for speed
        
        # Optimization settings
        opt_config = config.get("optimization", {})
        self.use_fp16 = opt_config.get("use_fp16", True)
        self.compile_model = opt_config.get("compile_model", False)  # PyTorch 2.0+
        self.batch_size = opt_config.get("batch_size", 1)

    def setup(self):
        """Initialize MarianMT model with optimizations"""
        self.logger.info(f"Loading MarianMT model: {self.model_name} on {self.device}")
        try:
            # Load tokenizer
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            
            # Load model with optimizations
            self.logger.info("Loading model...")
            self.model = MarianMTModel.from_pretrained(self.model_name)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Set to eval mode
            self.model.eval()
            
            # Apply FP16 optimization if enabled and on CUDA
            if self.use_fp16 and self.device == "cuda":
                self.logger.info("Applying FP16 optimization...")
                self.model = self.model.half()
            
            # Compile model for faster inference (PyTorch 2.0+)
            if self.compile_model:
                try:
                    self.logger.info("Compiling model with torch.compile()...")
                    self.model = torch.compile(self.model)
                except Exception as e:
                    self.logger.warning(f"torch.compile() failed (requires PyTorch 2.0+): {e}")
            
            # Test translation
            test_text = "Hello, how are you?"
            test_result = self._translate_text(test_text)
            self.logger.info(f"MarianMT initialized successfully. Test: '{test_text}' -> '{test_result}'")
            
        except Exception as e:
            self.logger.error(f"Failed to load MarianMT model: {e}")
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
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False  # Deterministic for consistency
            )
        
        # Decode
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
