"""
Translation Speed Test

Tests the latency and throughput of different translation engines.
"""

import time
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.translation.marian_translator import MarianTranslatorProcess
from src.translation.nllb_translator import NLLBTranslatorProcess
from src.translation.google_translator import GoogleTranslatorProcess


# Test sentences
TEST_SENTENCES = [
    "Hello, how are you today?",
    "The weather is beautiful outside.",
    "I am learning to use translation software.",
    "This is a test of the translation system.",
    "Machine learning is fascinating.",
    "Let's see how fast this translator can work.",
    "Context awareness improves translation quality.",
    "Real-time translation requires low latency.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is changing the world.",
]


def test_translator(translator_class, name, num_runs=3):
    """Test a translator's speed and accuracy"""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    try:
        # Initialize translator
        print(f"Initializing {name}...")
        translator = translator_class()
        translator.setup()
        
        latencies = []
        
        # Warm-up run
        print("Warming up...")
        for sentence in TEST_SENTENCES[:2]:
            translator.translate_with_context(sentence)
        
        # Actual test runs
        print(f"\nRunning {num_runs} test iterations...")
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}:")
            run_latencies = []
            
            for i, sentence in enumerate(TEST_SENTENCES):
                start = time.time()
                translation = translator.translate_with_context(sentence)
                latency = (time.time() - start) * 1000
                
                run_latencies.append(latency)
                latencies.append(latency)
                
                print(f"  [{i+1:2d}] {latency:6.1f}ms | {sentence[:40]:40s} -> {translation[:40]}")
            
            avg_latency = sum(run_latencies) / len(run_latencies)
            print(f"  Run average: {avg_latency:.1f}ms")
        
        # Statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        throughput = 1000 / avg_latency  # sentences per second
        
        print(f"\n{name} Results:")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  Min latency:     {min_latency:.1f}ms")
        print(f"  Max latency:     {max_latency:.1f}ms")
        print(f"  Throughput:      {throughput:.2f} sentences/sec")
        
        # Check metrics if available
        if hasattr(translator, 'metrics'):
            metrics = translator.metrics
            if metrics['total_translations'] > 0:
                cache_hit_rate = (metrics['cache_hits'] / metrics['total_translations']) * 100
                print(f"\n  Cache Statistics:")
                print(f"    Total translations: {metrics['total_translations']}")
                print(f"    Cache hits:         {metrics['cache_hits']}")
                print(f"    Cache hit rate:     {cache_hit_rate:.1f}%")
        
        return {
            'name': name,
            'avg_latency': avg_latency,
            'min_latency': min_latency,
            'max_latency': max_latency,
            'throughput': throughput,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': name,
            'success': False,
            'error': str(e)
        }


def main():
    print("="*60)
    print("Translation Speed Test")
    print("="*60)
    print(f"Testing {len(TEST_SENTENCES)} sentences with 3 runs each")
    
    results = []
    
    # Test MarianMT
    results.append(test_translator(MarianTranslatorProcess, "MarianMT", num_runs=3))
    
    # Test NLLB (optional - comment out if too slow)
    # results.append(test_translator(NLLBTranslatorProcess, "NLLB-200", num_runs=3))
    
    # Test Google Translate (requires internet)
    # results.append(test_translator(GoogleTranslatorProcess, "Google Translate", num_runs=3))
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        print(f"\n{'Engine':<20} {'Avg Latency':<15} {'Throughput':<15}")
        print("-" * 50)
        for r in successful_results:
            print(f"{r['name']:<20} {r['avg_latency']:>6.1f}ms        {r['throughput']:>6.2f} sent/sec")
        
        # Find fastest
        fastest = min(successful_results, key=lambda x: x['avg_latency'])
        print(f"\n🏆 Fastest: {fastest['name']} ({fastest['avg_latency']:.1f}ms)")
    
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n❌ Failed engines:")
        for r in failed_results:
            print(f"  - {r['name']}: {r['error']}")


if __name__ == "__main__":
    main()
