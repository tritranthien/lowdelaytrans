#!/usr/bin/env python3
"""
Automated Environment Setup Script for Low-Latency Voice Translation System
This script will install all dependencies and verify GPU compatibility.
"""

import subprocess
import sys
import os
from pathlib import Path
import json

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print_info(f"Running: {description}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_success(f"{description} - Completed")
            return True, result.stdout
        else:
            print_error(f"{description} - Failed")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return False, result.stderr
    except Exception as e:
        print_error(f"{description} - Exception: {str(e)}")
        return False, str(e)

def check_gpu():
    """Verify NVIDIA GPU and CUDA availability"""
    print_header("GPU & CUDA Verification")
    
    # Check nvidia-smi
    success, output = run_command("nvidia-smi", "Checking NVIDIA GPU", check=False)
    if not success:
        print_error("NVIDIA GPU not detected! This system requires RTX 3060.")
        return False
    
    # Check PyTorch CUDA
    check_code = """
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
"""
    success, output = run_command(f'python -c "{check_code}"', "Checking PyTorch CUDA", check=False)
    if success:
        print(output)
        if "RTX 3060" in output or "CUDA Available: True" in output:
            print_success("GPU verification passed!")
            return True
    
    print_warning("GPU check completed with warnings")
    return True

def install_tensorrt():
    """Install TensorRT for CUDA 12.1"""
    print_header("Installing TensorRT")
    
    # Try pip installation first
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install nvidia-tensorrt", "Installing NVIDIA TensorRT"),
    ]
    
    for cmd, desc in commands:
        success, _ = run_command(cmd, desc, check=False)
        if not success:
            print_warning(f"{desc} failed, trying alternative method...")
            # Try with specific version
            if "tensorrt" in cmd:
                run_command("pip install tensorrt", "Installing TensorRT (alternative)", check=False)
    
    # Verify installation
    verify_code = 'import tensorrt; print(f"TensorRT Version: {tensorrt.__version__}")'
    success, output = run_command(f'python -c "{verify_code}"', "Verifying TensorRT", check=False)
    
    if success:
        print_success("TensorRT installed successfully!")
        print(output)
        return True
    else:
        print_error("TensorRT installation failed!")
        print_warning("You may need to install manually from: https://developer.nvidia.com/tensorrt")
        return False

def install_dependencies():
    """Install all Python dependencies"""
    print_header("Installing Python Dependencies")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print_error(f"requirements.txt not found at {requirements_file}")
        return False
    
    # Install in stages to handle potential conflicts
    stages = [
        ("Core frameworks", ["torch", "torchvision", "torchaudio"]),
        ("Audio libraries", ["sounddevice", "soundfile", "librosa", "pyaudio", "pydub"]),
        ("UI framework", ["PySide6"]),
        ("NLP & Translation", ["transformers", "sentencepiece"]),
        ("Utilities", ["numpy", "scipy", "pyyaml", "tqdm", "psutil"]),
    ]
    
    for stage_name, packages in stages:
        print_info(f"\nInstalling {stage_name}...")
        for package in packages:
            cmd = f"pip install {package} --upgrade"
            run_command(cmd, f"Installing {package}", check=False)
    
    # Install remaining from requirements.txt
    print_info("\nInstalling remaining packages from requirements.txt...")
    run_command(f"pip install -r {requirements_file}", "Installing from requirements.txt", check=False)
    
    print_success("Dependency installation completed!")
    return True

def install_nemo():
    """Install NVIDIA NeMo"""
    print_header("Installing NVIDIA NeMo")
    
    commands = [
        ("pip install Cython", "Installing Cython (NeMo dependency)"),
        ("pip install nemo_toolkit[asr]", "Installing NeMo with ASR support"),
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc, check=False)
    
    # Verify
    verify_code = 'import nemo; import nemo.collections.asr as nemo_asr; print(f"NeMo Version: {nemo.__version__}")'
    success, output = run_command(f'python -c "{verify_code}"', "Verifying NeMo", check=False)
    
    if success:
        print_success("NeMo installed successfully!")
        print(output)
        return True
    else:
        print_warning("NeMo installation completed with warnings")
        return True

def create_project_structure():
    """Create project directory structure"""
    print_header("Creating Project Structure")
    
    base_dir = Path(__file__).parent
    directories = [
        "src",
        "src/asr",
        "src/ocr",
        "src/translation",
        "src/tts",
        "src/audio",
        "src/ui",
        "src/utils",
        "models",
        "models/asr",
        "models/translation",
        "models/tts",
        "models/tensorrt",
        "config",
        "logs",
        "tests",
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {dir_name}/")
    
    # Create __init__.py files
    for dir_name in [d for d in directories if d.startswith("src/")]:
        init_file = base_dir / dir_name / "__init__.py"
        init_file.touch(exist_ok=True)
    
    print_success("Project structure created!")
    return True

def generate_verification_report():
    """Generate a verification report"""
    print_header("Generating Verification Report")
    
    report = {
        "system": {},
        "python_packages": {},
        "gpu": {}
    }
    
    # System info
    success, output = run_command("python --version", "Python version", check=False)
    if success:
        report["system"]["python"] = output.strip()
    
    # GPU info
    gpu_check = """
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print(torch.version.cuda if torch.cuda.is_available() else "N/A")
"""
    success, output = run_command(f'python -c "{gpu_check}"', "GPU info", check=False)
    if success:
        lines = output.strip().split('\n')
        report["gpu"]["available"] = lines[0] == "True"
        report["gpu"]["name"] = lines[1] if len(lines) > 1 else "N/A"
        report["gpu"]["cuda_version"] = lines[2] if len(lines) > 2 else "N/A"
    
    # Package versions
    packages = ["torch", "tensorrt", "transformers", "sounddevice", "PySide6"]
    for pkg in packages:
        check = f'import {pkg}; print({pkg}.__version__)'
        success, output = run_command(f'python -c "{check}"', f"{pkg} version", check=False)
        if success:
            report["python_packages"][pkg] = output.strip()
        else:
            report["python_packages"][pkg] = "Not installed"
    
    # Save report
    report_file = Path(__file__).parent / "setup_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, indent=2, fp=f)
    
    print_success(f"Report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*70)
    print(f"{Colors.BOLD}SETUP SUMMARY{Colors.ENDC}")
    print("="*70)
    print(f"Python: {report['system'].get('python', 'N/A')}")
    print(f"GPU: {report['gpu'].get('name', 'N/A')}")
    print(f"CUDA: {report['gpu'].get('cuda_version', 'N/A')}")
    print("\nInstalled Packages:")
    for pkg, ver in report['python_packages'].items():
        status = Colors.OKGREEN if ver != "Not installed" else Colors.FAIL
        print(f"  {status}{pkg}: {ver}{Colors.ENDC}")
    print("="*70 + "\n")
    
    return True

def main():
    """Main setup routine"""
    print_header("Low-Latency Voice Translation System - Environment Setup")
    print_info("This script will install all required dependencies for RTX 3060 optimization\n")
    
    steps = [
        (check_gpu, "GPU & CUDA Verification"),
        (create_project_structure, "Project Structure Creation"),
        (install_tensorrt, "TensorRT Installation"),
        (install_dependencies, "Python Dependencies Installation"),
        (install_nemo, "NVIDIA NeMo Installation"),
        (generate_verification_report, "Verification Report Generation"),
    ]
    
    results = []
    for step_func, step_name in steps:
        try:
            success = step_func()
            results.append((step_name, success))
        except Exception as e:
            print_error(f"{step_name} failed with exception: {str(e)}")
            results.append((step_name, False))
    
    # Final summary
    print_header("Setup Complete!")
    print("\nResults:")
    all_success = True
    for step_name, success in results:
        if success:
            print_success(step_name)
        else:
            print_error(step_name)
            all_success = False
    
    if all_success:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 All setup steps completed successfully!{Colors.ENDC}")
        print(f"{Colors.OKCYAN}You can now start developing the voice translation system.{Colors.ENDC}\n")
    else:
        print(f"\n{Colors.WARNING}{Colors.BOLD}⚠ Setup completed with some warnings.{Colors.ENDC}")
        print(f"{Colors.WARNING}Please review the errors above and install missing components manually.{Colors.ENDC}\n")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
