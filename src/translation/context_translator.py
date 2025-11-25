"""
Context-Aware Translation Base Class

Provides context management and caching for translation engines.
"""

import time
from collections import deque
from typing import Dict, Optional, Tuple
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config
import queue
import hashlib


class ContextTranslator(ProcessBase):
    """Base class for context-aware translators"""
    
    def __init__(self, name: str = "Translation"):
        config = get_config("translation")
        super().__init__(name, config)
        
        # Context settings
        context_config = config.get("context", {})
        self.context_enabled = context_config.get("enabled", True)
        self.context_buffer_size = context_config.get("buffer_size", 5)
        self.max_context_length = context_config.get("max_context_length", 200)
        self.include_source = context_config.get("include_source", True)
        self.include_target = context_config.get("include_target", True)
        
        # Cache settings
        cache_config = config.get("cache", {})
        self.cache_enabled = cache_config.get("enabled", True)
        self.cache_max_size = cache_config.get("max_size", 1000)
        self.cache_ttl = cache_config.get("ttl", 3600)
        
        # Context buffer: stores (source_text, translated_text, timestamp)
        # Changed to per-speaker buffers for speaker diarization
        self.speaker_contexts = {}  # {speaker_id: deque()}
        self.global_context_buffer = deque(maxlen=self.context_buffer_size)  # Fallback
        self.current_speaker_id = None
        
        # Translation cache: {text_hash: (translation, timestamp)}
        self.translation_cache: Dict[str, Tuple[str, float]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_translations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency": 0.0,
            "with_context": 0,
            "without_context": 0
        }
        
        # Queues
        self.input_queue = QueueManager.get_queue("asr_output")
        
        # Create output queues
        self.tts_queue = QueueManager.create_queue("tts_input", maxsize=50)
        self.ui_queue = QueueManager.create_queue("ui_input", maxsize=50)
        
        # Transcript queue (optional, will be created by writer if enabled)
        try:
            self.transcript_queue = QueueManager.get_queue("transcript_input")
        except:
            self.transcript_queue = None
            
        self.register_input_queue("asr_output", self.input_queue)
        self.register_output_queue("tts_input", self.tts_queue)
        self.register_output_queue("ui_input", self.ui_queue)
        if self.transcript_queue:
            self.register_output_queue("transcript_input", self.transcript_queue)
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for cache key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, text: str) -> Optional[str]:
        """Get translation from cache if available and not expired"""
        if not self.cache_enabled:
            return None
        
        text_hash = self._hash_text(text)
        if text_hash in self.translation_cache:
            translation, timestamp = self.translation_cache[text_hash]
            
            # Check if cache entry is still valid
            if time.time() - timestamp < self.cache_ttl:
                self.metrics["cache_hits"] += 1
                return translation
            else:
                # Remove expired entry
                del self.translation_cache[text_hash]
        
        self.metrics["cache_misses"] += 1
        return None
    
    def _add_to_cache(self, text: str, translation: str):
        """Add translation to cache"""
        if not self.cache_enabled:
            return
        
        # Remove oldest entries if cache is full
        if len(self.translation_cache) >= self.cache_max_size:
            # Remove oldest 10% of entries
            sorted_items = sorted(
                self.translation_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            num_to_remove = max(1, self.cache_max_size // 10)
            for key, _ in sorted_items[:num_to_remove]:
                del self.translation_cache[key]
        
        text_hash = self._hash_text(text)
        self.translation_cache[text_hash] = (translation, time.time())
    
    def _get_speaker_context_buffer(self, speaker_id=None):
        """Get context buffer for specific speaker or global buffer."""
        if speaker_id is None:
            return self.global_context_buffer
        
        if speaker_id not in self.speaker_contexts:
            self.speaker_contexts[speaker_id] = deque(maxlen=self.context_buffer_size)
        
        return self.speaker_contexts[speaker_id]
    
    def _build_context_string(self, speaker_id=None) -> str:
        """Build context string from buffer for specific speaker."""
        if not self.context_enabled:
            return ""
        
        # Get appropriate context buffer
        context_buffer = self._get_speaker_context_buffer(speaker_id)
        
        if len(context_buffer) == 0:
            return ""
        
        context_parts = []
        total_length = 0
        
        # Iterate from most recent to oldest
        for source, target, _ in reversed(context_buffer):
            part = ""
            if self.include_source:
                part += f"EN: {source}"
            if self.include_target:
                if part:
                    part += " | "
                part += f"VI: {target}"
            
            # Check if adding this would exceed max length
            if total_length + len(part) > self.max_context_length:
                break
            
            context_parts.insert(0, part)
            total_length += len(part)
        
        return " || ".join(context_parts) if context_parts else ""
    
    def _add_to_context(self, source: str, target: str, speaker_id=None):
        """Add translation pair to context buffer for specific speaker."""
        if self.context_enabled:
            context_buffer = self._get_speaker_context_buffer(speaker_id)
            context_buffer.append((source, target, time.time()))
    
    def translate_with_context(self, text: str) -> str:
        """
        Translate text with context.
        Subclasses should override this method to implement actual translation.
        """
        raise NotImplementedError("Subclasses must implement translate_with_context()")
    
    def loop(self):
        """Main processing loop"""
        # Try to get transcript queue if not already got (it might be created later)
        if not hasattr(self, 'transcript_queue') or self.transcript_queue is None:
            try:
                self.transcript_queue = QueueManager.get_queue("transcript_input")
                if self.transcript_queue:
                    self.register_output_queue("transcript_input", self.transcript_queue)
            except:
                pass

        try:
            # Get input from ASR (can be string or dict with speaker_id)
            input_data = self.input_queue.get(timeout=0.1)
            
            # Handle both string and dict input
            if isinstance(input_data, dict):
                text = input_data.get("text", "")
                speaker_id = input_data.get("speaker_id")
                timestamp = input_data.get("timestamp")
            else:
                text = input_data
                speaker_id = None
                timestamp = time.time()
            
            if not text or len(text.strip()) == 0:
                return
            
            # Track speaker changes
            if speaker_id != self.current_speaker_id:
                if self.current_speaker_id is not None:
                    self.logger.debug(f"Speaker context switch: {self.current_speaker_id} -> {speaker_id}")
                self.current_speaker_id = speaker_id
            
            self.metrics["total_translations"] += 1
            start_time = time.time()
            
            # Check cache first
            cached_translation = self._get_from_cache(text)
            if cached_translation:
                translated = cached_translation
                latency = (time.time() - start_time) * 1000
                self.logger.info(
                    f"Translation [CACHED] ({latency:.1f}ms): {text[:50]}... -> {translated[:50]}..."
                )
            else:
                # Translate with context for this speaker
                context = self._build_context_string(speaker_id)
                has_context = len(context) > 0
                
                if has_context:
                    self.metrics["with_context"] += 1
                else:
                    self.metrics["without_context"] += 1
                
                # Call subclass implementation
                translated = self.translate_with_context(text)
                
                latency = (time.time() - start_time) * 1000
                self.metrics["total_latency"] += latency
                
                # Add to cache
                self._add_to_cache(text, translated)
                
                # Log with context and speaker indicator
                context_indicator = "[CTX]" if has_context else "[NO-CTX]"
                speaker_label = f" [Spk{speaker_id}]" if speaker_id else ""
                self.logger.info(
                    f"Translation {context_indicator}{speaker_label} ({latency:.1f}ms): "
                    f"{text[:50]}... -> {translated[:50]}..."
                )
            
            # Add to context buffer for this speaker
            self._add_to_context(text, translated, speaker_id)
            
            # Create output with speaker info
            output = {
                "text": translated,
                "speaker_id": speaker_id,
                "timestamp": timestamp
            }
            
            # Push to output queues
            self.tts_queue.put(output)
            self.ui_queue.put(output)
            
            # Push to transcript queue if available
            if self.transcript_queue:
                transcript_data = {
                    "text": translated,
                    "original": text,
                    "speaker_id": speaker_id,
                    "timestamp": timestamp
                }
                self.transcript_queue.put(transcript_data)
            
            # Log metrics periodically (every 50 translations)
            if self.metrics["total_translations"] % 50 == 0:
                self._log_metrics()
            
        except queue.Empty:
            pass
        except Exception as e:
            import traceback
            self.logger.error(f"Translation Error: {e}")
            self.logger.error(traceback.format_exc())
    
    def _log_metrics(self):
        """Log performance metrics"""
        total = self.metrics["total_translations"]
        if total == 0:
            return
        
        cache_hit_rate = (self.metrics["cache_hits"] / total) * 100
        avg_latency = self.metrics["total_latency"] / max(1, self.metrics["cache_misses"])
        context_rate = (self.metrics["with_context"] / total) * 100
        
        self.logger.info(
            f"Metrics: {total} translations | "
            f"Cache hit: {cache_hit_rate:.1f}% | "
            f"Avg latency: {avg_latency:.1f}ms | "
            f"Context usage: {context_rate:.1f}%"
        )
