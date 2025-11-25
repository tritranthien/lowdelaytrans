"""
System Verification Script
Checks all installed dependencies and GPU compatibility
"""

import sys
import json
from pathlib import Path

def print_section(title):
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")

def check_package(package_name, import_name=None):
    """Check if a package is installed and return version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:20s} {version}")
        return True, version
    except ImportError:
        print(f"✗ {package_name:20s} NOT INSTALLED")
        return False, None

def main():
    print_section("Low-Latency Voice Translation - System Verification")
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}\n")
    
    # Check core packages
    print_section("Core Dependencies")
    
    packages = [
        ("PyTorch", "torch"),
        ("TensorRT", "tensorrt"),
        ("Transformers", "transformers"),
        ("SoundDevice", "sounddevice"),
        ("PySide6", "PySide6"),
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("PyYAML", "yaml"),
    ]
    
    results = {}
    for pkg_name, import_name in packages:
        success, version = check_package(pkg_name, import_name)
        results[pkg_name] = {"installed": success, "version": version}
    
    # Check GPU
    print_section("GPU Information")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Test GPU
            print("\nTesting GPU computation...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            print("✓ GPU computation successful!")
            
            results["GPU"] = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        else:
            print("✗ CUDA not available!")
            results["GPU"] = {"available": False}
            
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        results["GPU"] = {"available": False, "error": str(e)}
    
    # Check ASR
    print_section("ASR (Speech Recognition)")
    
    try:
        from faster_whisper import WhisperModel
        print("✓ faster-whisper installed")
        
        # Try to load a small model
        print("Testing model loading...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✓ Whisper model loaded successfully!")
        results["ASR"] = {"engine": "faster-whisper", "status": "OK"}
    except Exception as e:
        print(f"✗ Error with faster-whisper: {e}")
        results["ASR"] = {"engine": "faster-whisper", "status": "ERROR", "error": str(e)}
    
    # Check Translation
    print_section("Translation")
    
    try:
        from transformers import MarianMTModel, MarianTokenizer
        print("✓ MarianMT available")
        results["Translation"] = {"engine": "MarianMT", "status": "OK"}
    except Exception as e:
        print(f"✗ Error with MarianMT: {e}")
        results["Translation"] = {"engine": "MarianMT", "status": "ERROR", "error": str(e)}
    
    # Check TTS
    print_section("Text-to-Speech")
    
    try:
        import edge_tts
        print("✓ Edge TTS installed")
        results["TTS"] = {"engine": "edge-tts", "status": "OK"}
    except:
        print("⚠ Edge TTS not installed (optional)")
        results["TTS"] = {"engine": "edge-tts", "status": "NOT_INSTALLED"}
    
    # Check Audio
    print_section("Audio I/O")
    
    try:
        import sounddevice as sd
        print("✓ sounddevice installed")
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        results["Audio"] = {"status": "OK"}
    except Exception as e:
        print(f"✗ Error with audio: {e}")
        results["Audio"] = {"status": "ERROR", "error": str(e)}
    
    # Summary
    print_section("Verification Summary")
    
    all_ok = True
    critical_components = ["PyTorch", "GPU", "ASR", "Translation", "Audio"]
    
    for component in critical_components:
        if component in results:
            status = results[component].get("installed", results[component].get("available", results[component].get("status")))
            if status in [True, "OK"]:
                print(f"✓ {component:20s} READY")
            else:
                print(f"✗ {component:20s} FAILED")
                all_ok = False
    
    print("\n" + "="*70)
    if all_ok:
        print("🎉 System verification PASSED! Ready for development.")
    else:
        print("⚠ System verification FAILED. Please fix the issues above.")
    print("="*70 + "\n")
    
    # Save results
    report_file = Path(__file__).parent / "verification_report.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed report saved to: {report_file}")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
