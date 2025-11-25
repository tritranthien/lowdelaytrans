import numpy as np
import time
import nemo.collections.asr as nemo_asr
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config
import queue
import torch

class NeMoASRProcess(ProcessBase):
    def __init__(self):
        config = get_config("asr")
        super().__init__("ASR", config)
        
        # NeMo model name
        self.model_name = config.get("nemo.model_name", "stt_en_jasper10x5dr")
        self.device = config.get("nemo.device", "cuda")
        
        # Audio buffer for streaming
        self.audio_buffer = []
        self.buffer_duration_ms = 0
        self.min_duration_ms = 4000  # 4 seconds - longer for complete sentences
        self.max_duration_ms = 8000  # 8 seconds max
        
        # Queues
        self.input_queue = QueueManager.get_queue("audio_input")
        self.output_queue = QueueManager.create_queue("asr_output", maxsize=50)
        self.register_input_queue("audio_input", self.input_queue)
        self.register_output_queue("asr_output", self.output_queue)

    def setup(self):
        self.logger.info(f"Initializing NeMo ASR Process...")
        self.logger.info(f"Model: {self.model_name}, Device: {self.device}")
        
        try:
            import torch
            self.logger.info(f"Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
            
            import nemo.collections.asr as nemo_asr
            self.logger.info("NeMo ASR module imported")
            
            # Load pre-trained NeMo model
            self.logger.info(f"Loading model {self.model_name}...")
            self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                model_name=self.model_name
            )
            self.logger.info("Model loaded from pretrained")
            
            # Move to GPU
            if self.device == "cuda" and torch.cuda.is_available():
                self.logger.info("Moving model to CUDA...")
                self.model = self.model.to(self.device)
                self.logger.info("Model moved to CUDA")
            
            self.model.eval()
            self.logger.info("NeMo ASR model initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to load NeMo model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.stop()

    def loop(self):
        try:
            # Get audio chunk
            chunk = self.input_queue.get(timeout=0.1)
            
            # Add to buffer
            self.audio_buffer.append(chunk)
            
            # Calculate buffer duration (assuming 16kHz)
            chunk_duration = (len(chunk) / 16000) * 1000
            self.buffer_duration_ms += chunk_duration
            
            # Process if we have enough audio
            if self.buffer_duration_ms >= self.min_duration_ms:
                self._process_buffer()
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"ASR Error: {e}")

    def _process_buffer(self):
        if not self.audio_buffer:
            return
            
        # Concatenate buffer
        audio_data = np.concatenate(self.audio_buffer)
        
        # Clear buffer
        self.audio_buffer = []
        self.buffer_duration_ms = 0
        
        # Run inference
        start_time = time.time()
        
        try:
            # NeMo expects audio as float32 numpy array
            audio_signal = torch.from_numpy(audio_data).float()
            
            if self.device == "cuda":
                audio_signal = audio_signal.to(self.device)
            
            # Add batch dimension
            audio_signal = audio_signal.unsqueeze(0)
            
            # Get audio length
            audio_length = torch.tensor([len(audio_data)]).to(audio_signal.device)
            
            # Transcribe
            with torch.no_grad():
                # Forward pass
                logits = self.model.forward(
                    input_signal=audio_signal,
                    input_signal_length=audio_length
                )
                
                # Handle tuple return (logits, lengths)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # Decode
                greedy_predictions = logits.argmax(dim=-1, keepdim=False)
                text = self.model.decoding.ctc_decoder_predictions_tensor(
                    greedy_predictions
                )[0]
            
            latency = (time.time() - start_time) * 1000
            
            if isinstance(text, list):
                text = text[0]
            
            if text and len(text.strip()) > 0:
                self.logger.info(f"ASR Output ({latency:.1f}ms): {text}")
                self.output_queue.put(text.strip())
                
        except Exception as e:
            self.logger.error(f"NeMo transcription error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
