import numpy as np
import cv2
from PIL import Image

DEFAULT_FILTERS = {
    "Deepfakes": ["deblock_light"],
    "Face2Face": ["unsharp_light"],
    "FaceShifter": ["unsharp_light"],
    "FaceSwap": ["unsharp_mid"],
    "NeuralTextures": ["denoise_verylight", "unsharp_verylight"],
    "ImprovedDeepfake": ["unsharp_light", "deblock_light"],  # Tùy chỉnh cho method mới, dựa trên đặc tính (có thể test và chỉnh)
    "Transformer": ["unsharp_mid", "denoise_verylight"],   # Tùy chỉnh
    "Diffusion": ["deblock_light"]                         # Tùy chỉnh
}

def _unsharp_np(img_rgb, amount=0.6, radius=3):
    blur = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=radius)
    sharp = cv2.addWeighted(img_rgb, 1 + amount, blur, -amount, 0)
    return sharp

def _deblock_np(img_rgb, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, h, hColor, templateWindowSize, searchWindowSize)

def _denoise_np(img_rgb, strength=5):
    return cv2.bilateralFilter(img_rgb, d=0, sigmaColor=strength, sigmaSpace=strength)

def apply_method_filters(pil_img, method_name):
    img = np.array(pil_img.convert("RGB"))
    ops = DEFAULT_FILTERS.get(method_name, [])
    out = img
    for op in ops:
        if op == "unsharp_verylight": out = _unsharp_np(out, amount=0.25, radius=1.5)
        elif op == "unsharp_light": out = _unsharp_np(out, amount=0.45, radius=2.0)
        elif op == "unsharp_mid": out = _unsharp_np(out, amount=0.7, radius=2.0)
        elif op == "deblock_light": out = _deblock_np(out, h=2, hColor=2, templateWindowSize=7, searchWindowSize=21)
        elif op =="denoise_verylight": out = _denoise_np(out, strength=4)
    return Image.fromarray(out)