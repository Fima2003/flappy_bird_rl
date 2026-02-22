import torch
import math


def check_gpu():
    # Check specifically for "MPS" (Metal Performance Shaders)
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print("✅ SUCCESS: Apple Metal (MPS) is active!")
        print(f"   Device: {mps_device}")
    else:
        print("❌ WARNING: Running on CPU. Something is wrong.")
