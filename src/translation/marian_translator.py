import time
import torch
from transformers import MarianMTModel, MarianTokenizer
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config
import queue

class MarianTranslatorProcess(ProcessBase):
    def __init__(self):
        config = get_config("translation")
        super().__init__("Translation", config)
        
        self.model_name = config.get("marian.model_name", "Helsinki-NLP/opus-mt-en-vi")
        self.device = config.get("marian.device", "cuda")
        self.max_length = config.get("marian.max_length", 200)
        self.num_beams = config.get("marian.num_beams", 3)  # Use beam search for better quality
        
        # Queues
        self.input_queue = QueueManager.get_queue("asr_output")
        
        # Create output queues
        self.tts_queue = QueueManager.create_queue("tts_input", maxsize=50)
        self.ui_queue = QueueManager.create_queue("ui_input", maxsize=50)
        
        self.register_input_queue("asr_output", self.input_queue)
        self.register_output_queue("tts_input", self.tts_queue)
        self.register_output_queue("ui_input", self.ui_queue)

    def setup(self):
        self.logger.info(f"Loading MarianMT model: {self.model_name} on {self.device}")
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            # Force use_safetensors=True to avoid PyTorch pickle security issues
            self.model = MarianMTModel.from_pretrained(
                self.model_name,
                use_safetensors=True
            ).to(self.device)
            self.logger.info("MarianMT model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load MarianMT model: {e}")
            # Try falling back to original loading if safetensors fails
            try:
                self.logger.info("Retrying with default loading...")
                self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)
            except Exception as e2:
                self.logger.error(f"Retry failed: {e2}")
                self.stop()

    def loop(self):
        try:
            # Get text from ASR
            text = self.input_queue.get(timeout=0.1)
            
            if not text or len(text.strip()) == 0:
                return

            start_time = time.time()
            
            # Simple direct translation
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
            self.logger.error(traceback.format_exc())
