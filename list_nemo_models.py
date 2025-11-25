import sys
import os

# Add src to path
sys.path.append(os.getcwd())

import nemo.collections.asr as nemo_asr

print("Listing available models for EncDecCTCModel...")
try:
    models = nemo_asr.models.EncDecCTCModel.list_available_models()
    for model in models:
        print(f"- {model.pretrained_model_name}")
except Exception as e:
    print(f"Error listing models: {e}")
