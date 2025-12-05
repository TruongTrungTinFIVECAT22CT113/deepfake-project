import json
import os
import numpy as np
import cv2
from PIL import Image

# mapping động: { method_name: [ops...] }
_FILTERS_MAP = {}

def set_filters_map(m: dict | None):
    """Gán map filter theo method từ bên ngoài (app.py sẽ gọi lúc startup)."""
    global _FILTERS_MAP
    _FILTERS_MAP = dict(m or {})

def try_load_filters_json(base_dir: str):
    """
    Tìm filters.json trong base_dir (cùng nơi đặt checkpoint).
    Định dạng:
    {
      "Deepfakes": ["deblock_light"],
      "Face2Face": ["unsharp_light"],
      ...
    }
    """
    path = os.path.join(base_dir, "filters.json")
    if os.path.isfile(path):
        try:
            js = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(js, dict):
                set_filters_map(js)
        except Exception:
            pass

def _unsharp_np(img_rgb, amount=0.6, radius=3):
    blur = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=radius)
    sharp = cv2.addWeighted(img_rgb, 1 + amount, blur, -amount, 0)
    return sharp

def _deblock_np(img_rgb, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, h, hColor, templateWindowSize, searchWindowSize)

def _denoise_np(img_rgb, strength=5):
    return cv2.bilateralFilter(img_rgb, d=0, sigmaColor=strength, sigmaSpace=strength)

def apply_method_filters(pil_img, method_name: str):
    """
    Áp các filter theo method nếu có cấu hình trong _FILTERS_MAP.
    Không có config → trả nguyên ảnh.
    """
    ops = _FILTERS_MAP.get(method_name, [])
    if not ops:
        return pil_img

    img = np.array(pil_img.convert("RGB"))
    out = img
    for op in ops:
        if op == "unsharp_verylight": out = _unsharp_np(out, amount=0.25, radius=1.5)
        elif op == "unsharp_light":   out = _unsharp_np(out, amount=0.45, radius=2.0)
        elif op == "unsharp_mid":     out = _unsharp_np(out, amount=0.7,  radius=2.0)
        elif op == "deblock_light":   out = _deblock_np(out, h=2, hColor=2, templateWindowSize=7, searchWindowSize=21)
        elif op == "denoise_verylight": out = _denoise_np(out, strength=4)
    return Image.fromarray(out)