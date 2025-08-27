import torch
import torchvision
import platform
import sys
import os

print("=== Python & System Info ===")
print("Python version:", sys.version)
print("OS:", platform.system(), platform.release(), platform.version())
print("Machine:", platform.machine())

print("\n=== PyTorch & CUDA Info ===")
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("GPU(s) detected:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print("     Memory Allocated:", round(torch.cuda.memory_allocated(i)/1024**2, 1), "MB")
        print("     Memory Cached:", round(torch.cuda.memory_reserved(i)/1024**2, 1), "MB")

print("\n=== Installed Packages (important ones) ===")
try:
    import diffusers
    print("diffusers version:", diffusers.__version__)
except ImportError:
    print("diffusers not installed")

try:
    import transformers
    print("transformers version:", transformers.__version__)
except ImportError:
    print("transformers not installed")

try:
    import timm
    print("timm (ViT models) version:", timm.__version__)
except ImportError:
    print("timm not installed")