"""
Test Speaker Diarization

Simple test to verify speaker diarization works.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio.speaker_diarization import SpeakerDiarizer


def test_speaker_diarization():
    """Test basic speaker diarization functionality"""
    print("="*60)
    print("Speaker Diarization Test")
    print("="*60)
    
    # Initialize diarizer
    print("\n1. Initializing speaker diarizer...")
    diarizer = SpeakerDiarizer(
        similarity_threshold=0.75,
        min_duration=1.0,
        max_speakers=5
    )
    print("✓ Diarizer initialized")
    
    # Generate dummy audio (in real use, this comes from microphone)
    print("\n2. Generating test audio...")
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    
    # Create 3 different "speakers" with different frequencies
    speakers_audio = []
    for i in range(3):
        # Generate sine wave with different frequency for each speaker
        freq = 200 + (i * 100)  # 200Hz, 300Hz, 400Hz
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.1, audio.shape).astype(np.float32)
        audio = audio + noise
        
        speakers_audio.append(audio)
    
    print(f"✓ Generated {len(speakers_audio)} test speakers")
    
    # Test speaker identification
    print("\n3. Testing speaker identification...")
    speaker_ids = []
    
    for i, audio in enumerate(speakers_audio):
        speaker_id = diarizer.identify_speaker(audio, sample_rate)
        speaker_ids.append(speaker_id)
        print(f"   Audio {i+1}: Identified as Speaker {speaker_id}")
    
    # Verify different speakers get different IDs
    unique_speakers = len(set(speaker_ids))
    print(f"\n✓ Detected {unique_speakers} unique speakers")
    
    # Test speaker re-identification (same speaker should get same ID)
    print("\n4. Testing speaker re-identification...")
    for i, audio in enumerate(speakers_audio):
        speaker_id = diarizer.identify_speaker(audio, sample_rate)
        expected_id = speaker_ids[i]
        
        if speaker_id == expected_id:
            print(f"   Audio {i+1}: ✓ Correctly re-identified as Speaker {speaker_id}")
        else:
            print(f"   Audio {i+1}: ✗ Mis-identified as Speaker {speaker_id} (expected {expected_id})")
    
    # Show statistics
    print("\n5. Diarization Statistics:")
    stats = diarizer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
    
    print("\nNote: This is a basic test with synthetic audio.")
    print("Real-world performance will vary with actual voice recordings.")
    print("\nTo test with real audio:")
    print("  1. Run the full application: python run.py")
    print("  2. Play audio with multiple speakers (game/movie)")
    print("  3. Check logs for speaker detection")


if __name__ == "__main__":
    try:
        test_speaker_diarization()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
