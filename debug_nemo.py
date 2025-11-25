import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.utils.config import Config
from src.asr.nemo_asr import NeMoASRProcess
import traceback

def main():
    print("Initializing Config...")
    Config()
    
    print("Creating NeMoASRProcess...")
    try:
        asr = NeMoASRProcess()
        print("Calling setup()...")
        asr.setup()
        print("Setup complete!")
    except Exception:
        print("CRASHED!")
        traceback.print_exc()

if __name__ == "__main__":
    main()
