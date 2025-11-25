"""
Speaker Diarization Module using Resemblyzer

Provides real-time speaker identification and tracking for multi-speaker audio.
"""

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import time
import logging


class SpeakerDiarizer:
    """
    Real-time speaker diarization using voice embeddings.
    
    Uses Resemblyzer to extract voice embeddings and online clustering
    to identify and track speakers in real-time.
    """
    
    def __init__(self, 
                 similarity_threshold=0.75,
                 min_duration=1.0,
                 max_speakers=10,
                 speaker_timeout=300):
        """
        Initialize speaker diarizer.
        
        Args:
            similarity_threshold: Cosine similarity threshold for speaker matching (0-1)
            min_duration: Minimum audio duration (seconds) for reliable embedding
            max_speakers: Maximum number of speakers to track
            speaker_timeout: Remove inactive speakers after this many seconds
        """
        self.logger = logging.getLogger("SpeakerDiarization")
        
        # Load Resemblyzer voice encoder
        self.logger.info("Loading Resemblyzer voice encoder...")
        self.encoder = VoiceEncoder()
        self.logger.info("Voice encoder loaded successfully")
        
        # Configuration
        self.similarity_threshold = similarity_threshold
        self.min_duration = min_duration
        self.max_speakers = max_speakers
        self.speaker_timeout = speaker_timeout
        
        # Speaker tracking
        self.speakers = {}  # {speaker_id: {"embedding": np.array, "last_seen": timestamp}}
        self.next_speaker_id = 1
        self.current_speaker_id = None
        
        # Audio buffer for minimum duration
        self.audio_buffer = deque(maxlen=100)  # Store recent audio chunks
        self.buffer_duration = 0.0
        
        # Statistics
        self.stats = {
            "total_identifications": 0,
            "new_speakers_detected": 0,
            "speaker_changes": 0
        }
    
    def _compute_embedding(self, audio_data, sample_rate=16000):
        """
        Compute voice embedding for audio data.
        
        Args:
            audio_data: Audio samples (numpy array)
            sample_rate: Sample rate of audio
            
        Returns:
            Voice embedding (256-dim numpy array) or None if failed
        """
        try:
            # Preprocess audio (Resemblyzer expects specific format)
            # Audio should be 16kHz, mono, normalized
            if len(audio_data) < sample_rate * 0.5:  # At least 0.5 seconds
                return None
            
            # Normalize audio to [-1, 1] if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Preprocess with Resemblyzer
            wav = preprocess_wav(audio_data, source_sr=sample_rate)
            
            # Extract embedding
            embedding = self.encoder.embed_utterance(wav)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to compute embedding: {e}")
            return None
    
    def _find_matching_speaker(self, embedding):
        """
        Find matching speaker for given embedding.
        
        Args:
            embedding: Voice embedding to match
            
        Returns:
            speaker_id if match found, None otherwise
        """
        if not self.speakers:
            return None
        
        # Compute similarity with all known speakers
        best_match_id = None
        best_similarity = -1.0
        
        for speaker_id, speaker_data in self.speakers.items():
            speaker_embedding = speaker_data["embedding"]
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                speaker_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = speaker_id
        
        # Check if best match exceeds threshold
        if best_similarity >= self.similarity_threshold:
            return best_match_id
        
        return None
    
    def _create_new_speaker(self, embedding):
        """
        Create a new speaker with given embedding.
        
        Args:
            embedding: Voice embedding for new speaker
            
        Returns:
            New speaker ID
        """
        speaker_id = self.next_speaker_id
        self.next_speaker_id += 1
        
        self.speakers[speaker_id] = {
            "embedding": embedding,
            "last_seen": time.time(),
            "sample_count": 1
        }
        
        self.stats["new_speakers_detected"] += 1
        self.logger.info(f"New speaker detected: Speaker {speaker_id}")
        
        return speaker_id
    
    def _update_speaker_embedding(self, speaker_id, new_embedding, alpha=0.3):
        """
        Update speaker embedding with moving average.
        
        Args:
            speaker_id: Speaker to update
            new_embedding: New embedding to incorporate
            alpha: Weight for new embedding (0-1)
        """
        if speaker_id not in self.speakers:
            return
        
        old_embedding = self.speakers[speaker_id]["embedding"]
        
        # Moving average: new = alpha * new + (1 - alpha) * old
        updated_embedding = alpha * new_embedding + (1 - alpha) * old_embedding
        
        # Normalize
        updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
        
        self.speakers[speaker_id]["embedding"] = updated_embedding
        self.speakers[speaker_id]["last_seen"] = time.time()
        self.speakers[speaker_id]["sample_count"] += 1
    
    def _cleanup_inactive_speakers(self):
        """Remove speakers that haven't been seen recently."""
        current_time = time.time()
        inactive_speakers = []
        
        for speaker_id, speaker_data in self.speakers.items():
            if current_time - speaker_data["last_seen"] > self.speaker_timeout:
                inactive_speakers.append(speaker_id)
        
        for speaker_id in inactive_speakers:
            del self.speakers[speaker_id]
            self.logger.info(f"Removed inactive Speaker {speaker_id}")
    
    def identify_speaker(self, audio_data, sample_rate=16000):
        """
        Identify speaker from audio data.
        
        Args:
            audio_data: Audio samples (numpy array)
            sample_rate: Sample rate of audio
            
        Returns:
            speaker_id (int) or None if identification failed
        """
        # Compute embedding
        embedding = self._compute_embedding(audio_data, sample_rate)
        
        if embedding is None:
            return None
        
        # Find matching speaker
        speaker_id = self._find_matching_speaker(embedding)
        
        if speaker_id is None:
            # Create new speaker if under max limit
            if len(self.speakers) < self.max_speakers:
                speaker_id = self._create_new_speaker(embedding)
            else:
                # Use most similar speaker even if below threshold
                self.logger.warning(f"Max speakers ({self.max_speakers}) reached")
                speaker_id = self._find_matching_speaker(embedding) or 1
        else:
            # Update existing speaker embedding
            self._update_speaker_embedding(speaker_id, embedding)
        
        # Track speaker changes
        if speaker_id != self.current_speaker_id:
            if self.current_speaker_id is not None:
                self.stats["speaker_changes"] += 1
                self.logger.info(f"Speaker changed: {self.current_speaker_id} -> {speaker_id}")
            self.current_speaker_id = speaker_id
        
        self.stats["total_identifications"] += 1
        
        # Cleanup inactive speakers periodically
        if self.stats["total_identifications"] % 50 == 0:
            self._cleanup_inactive_speakers()
        
        return speaker_id
    
    def get_current_speaker(self):
        """Get current speaker ID."""
        return self.current_speaker_id
    
    def get_speaker_count(self):
        """Get number of active speakers."""
        return len(self.speakers)
    
    def get_stats(self):
        """Get diarization statistics."""
        return {
            **self.stats,
            "active_speakers": len(self.speakers),
            "current_speaker": self.current_speaker_id
        }
    
    def reset(self):
        """Reset all speaker data."""
        self.speakers.clear()
        self.next_speaker_id = 1
        self.current_speaker_id = None
        self.audio_buffer.clear()
        self.buffer_duration = 0.0
        self.logger.info("Speaker diarization reset")
