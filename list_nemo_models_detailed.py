import sys
import os
sys.path.append(os.getcwd())

import nemo.collections.asr as nemo_asr

print("Available NeMo ASR models:")
print("=" * 60)

models = nemo_asr.models.EncDecCTCModel.list_available_models()
for i, model in enumerate(models, 1):
    print(f"{i}. {model.pretrained_model_name}")
    if hasattr(model, 'description'):
        print(f"   Description: {model.description}")
    print()
