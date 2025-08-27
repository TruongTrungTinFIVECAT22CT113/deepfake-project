import torch
import torchvision
import torchaudio
import xformers

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "Available:", torch.cuda.is_available())
print("Torchvision:", torchvision.__version__)
print("Torchaudio:", torchaudio.__version__)
print("Xformers:", xformers.__version__)