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
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("GPU(s) detected:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print("     Memory Allocated:", round(torch.cuda.memory_allocated(i)/1024**2, 1), "MB")
        print("     Memory Cached:", round(torch.cuda.memory_reserved(i)/1024**2, 1), "MB")

print("\n=== Installed Packages (important ones) ===")

from importlib import metadata as _md

_PKGS = [
    "torch","torchvision","torchaudio","timm",
    "opencv-python","mediapipe","gradio","tqdm",
    "numpy","pillow","pandas","scikit-image","scikit-learn",
    "onnxruntime","tflite-runtime"
]

def _ver(name):
    try:
        return _md.version(name)
    except Exception:
        try:
            mod = __import__(name.replace("-", "_"))
            return getattr(mod, "__version__", "installed (no __version__)")
        except Exception:
            return "NOT INSTALLED"

def print_versions(save_path="deepfake-project/reports/env_report.txt"):
    lines = []
    lines.append("Python      : " + sys.version.split()[0])
    try:
        import torch
        lines.append(f"PyTorch     : {torch.__version__} | CUDA={torch.cuda.is_available()} | device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA device : {torch.cuda.get_device_name(0)}")
            lines.append(f"cuDNN       : enabled={torch.backends.cudnn.enabled} | version={torch.backends.cudnn.version()}")
    except Exception as e:
        lines.append(f"PyTorch     : not available ({e})")
    lines.append("")
    lines.append("Packages:")
    for p in _PKGS:
        lines.append(f"- {p:15} { _ver(p) }")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n✔ Saved to {save_path}")

if __name__ == "__main__":
    print_versions()
