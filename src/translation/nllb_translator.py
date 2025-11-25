import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config
import queue
import torch

class NLLBTranslatorProcess(ProcessBase):
    def __init__(self):
        config = get_config("translation")
        super().__init__("Translation", config)
        
        self.model_name = config.get("nllb.model_name", "facebook/nllb-200-distilled-600M")
        self.device = config.get("nllb.device", "cuda")
        self.max_length = config.get("nllb.max_length", 512)
        self.num_beams = config.get("nllb.num_beams", 3)
        
        # Language codes for NLLB
        self.src_lang = config.get("nllb.src_lang", "eng_Latn")  # English
        self.tgt_lang = config.get("nllb.tgt_lang", "vie_Latn")  # Vietnamese
        
        # Queues
        self.input_queue = QueueManager.get_queue("asr_output")
        
        # Create output queues
        self.tts_queue = QueueManager.create_queue("tts_input", maxsize=50)
        self.ui_queue = QueueManager.create_queue("ui_input", maxsize=50)
        
        self.register_input_queue("asr_output", self.input_queue)
        self.register_output_queue("tts_input", self.tts_queue)
        self.register_output_queue("ui_input", self.ui_queue)

    def setup(self):
        self.logger.info(f"Loading NLLB model: {self.model_name} on {self.device}")
        try:
            import traceback
            self.logger.info("Initializing tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                src_lang=self.src_lang,
                use_fast=False  # Use slow tokenizer for compatibility
            )
            self.logger.info("Tokenizer loaded successfully")
            
            self.logger.info("Loading model (this may take a while on first run)...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use FP16 for faster inference
                low_cpu_mem_usage=True
            ).to(self.device)
            
            # Set to eval mode for inference
            self.model.eval()
            
            self.logger.info("NLLB model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load NLLB model: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.stop()
            raise  # Re-raise to ensure process dies properly

    def loop(self):
        try:
            # Get text from ASR
            text = self.input_queue.get(timeout=0.1)
            
            if not text or len(text.strip()) == 0:
                return
            
            # Translate
            start_time = time.time()
            
            # Tokenize - NLLB tokenizer handles language automatically
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            
            # Generate translation
            # For NLLB, we need to set the target language in generate()
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )
            
            # Decode
            translated = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            latency = (time.time() - start_time) * 1000
            self.logger.info(f"Translation ({latency:.1f}ms): {text} -> {translated}")
            
            # Push to output queues
            self.tts_queue.put(translated)
            self.ui_queue.put(translated)
            
        except queue.Empty:
            pass
        except Exception as e:
            import traceback
            self.logger.error(f"Translation Error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
