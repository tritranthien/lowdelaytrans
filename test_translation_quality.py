"""
Translation Quality Test

Tests context-aware translation quality with sentences that depend on context.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.translation.marian_translator import MarianTranslatorProcess


# Test cases with context dependency
TEST_CASES = [
    {
        "name": "Pronoun Resolution",
        "sentences": [
            "John went to the store.",
            "He bought some milk.",
            "Then he went home."
        ],
        "description": "Test if 'he' is translated correctly based on context"
    },
    {
        "name": "Technical Terms",
        "sentences": [
            "We are building a machine learning model.",
            "The model uses neural networks.",
            "It achieves 95% accuracy."
        ],
        "description": "Test if technical terms maintain consistency"
    },
    {
        "name": "Conversation Flow",
        "sentences": [
            "How are you doing today?",
            "I'm doing great, thanks for asking.",
            "That's wonderful to hear!"
        ],
        "description": "Test conversational context"
    },
    {
        "name": "Story Continuation",
        "sentences": [
            "Once upon a time, there was a brave knight.",
            "The knight fought a fierce dragon.",
            "In the end, the knight won the battle."
        ],
        "description": "Test narrative consistency"
    }
]


def test_context_quality(translator_class, name):
    """Test translation quality with context"""
    print(f"\n{'='*70}")
    print(f"Testing {name} - Context-Aware Translation Quality")
    print(f"{'='*70}")
    
    try:
        # Initialize translator
        print(f"Initializing {name}...")
        translator = translator_class()
        translator.setup()
        
        for test_case in TEST_CASES:
            print(f"\n{'-'*70}")
            print(f"Test Case: {test_case['name']}")
            print(f"Description: {test_case['description']}")
            print(f"{'-'*70}")
            
            # Reset context for each test case
            translator.context_buffer.clear()
            
            for i, sentence in enumerate(test_case['sentences']):
                # Get context before translation
                context = translator._build_context_string()
                
                # Translate
                translation = translator.translate_with_context(sentence)
                
                # Display
                context_indicator = "[WITH CONTEXT]" if context else "[NO CONTEXT]"
                print(f"\n{i+1}. {context_indicator}")
                print(f"   EN: {sentence}")
                print(f"   VI: {translation}")
                
                if context and i > 0:
                    print(f"   Context used: {context[:100]}...")
        
        # Show cache statistics
        if hasattr(translator, 'metrics'):
            metrics = translator.metrics
            print(f"\n{'='*70}")
            print("Translation Metrics:")
            print(f"{'='*70}")
            print(f"Total translations:  {metrics['total_translations']}")
            print(f"With context:        {metrics['with_context']}")
            print(f"Without context:     {metrics['without_context']}")
            print(f"Cache hits:          {metrics['cache_hits']}")
            print(f"Cache misses:        {metrics['cache_misses']}")
            
            if metrics['total_translations'] > 0:
                context_rate = (metrics['with_context'] / metrics['total_translations']) * 100
                cache_hit_rate = (metrics['cache_hits'] / metrics['total_translations']) * 100
                print(f"Context usage rate:  {context_rate:.1f}%")
                print(f"Cache hit rate:      {cache_hit_rate:.1f}%")
        
        print(f"\n✅ {name} quality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("Translation Quality Test - Context Awareness")
    print("="*70)
    print("\nThis test evaluates how well the translator uses context")
    print("from previous sentences to improve translation quality.\n")
    
    # Test MarianMT (default engine)
    success = test_context_quality(MarianTranslatorProcess, "MarianMT")
    
    if success:
        print("\n" + "="*70)
        print("Test completed successfully! ✅")
        print("="*70)
        print("\nReview the translations above to verify:")
        print("  1. Pronouns are translated correctly based on context")
        print("  2. Technical terms maintain consistency")
        print("  3. Conversational flow is natural")
        print("  4. Narrative continuity is preserved")
    else:
        print("\n" + "="*70)
        print("Test failed ❌")
        print("="*70)


if __name__ == "__main__":
    main()
