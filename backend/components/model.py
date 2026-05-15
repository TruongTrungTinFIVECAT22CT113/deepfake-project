# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import timm
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _infer_arch_family(model_name: str) -> str:
    n = model_name.lower()
    if "swin" in n: return "swin"
    if "maxvit" in n: return "maxvit"
    if "coatnet" in n: return "coatnet"
    if any(k in n for k in ["vit", "deit", "beit", "cait"]): return "vit"
    return "cnn"


class MultiHeadViT(nn.Module):
    def __init__(self, model_name: str, img_size: int, num_methods: int,
                 num_face_classes: int, num_head_classes: int, num_full_classes: int,
                 drop_rate: float=0.0, drop_path_rate: float=0.0):
        super().__init__()
        backbone_kwargs = dict(pretrained=False, num_classes=0, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        if any(k in model_name.lower() for k in ["vit", "swin", "beit", "deit", "cait"]):
            backbone_kwargs["img_size"] = img_size

        self.backbone = timm.create_model(model_name, **backbone_kwargs)
        feat = self.backbone.num_features

        def head(n): 
            return nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(feat, n))

        self.head_bin  = head(2)
        self.head_met  = head(num_methods)
        self.head_face = head(max(1, num_face_classes))
        self.head_head = head(max(1, num_head_classes))
        self.head_full = head(max(1, num_full_classes))

    def forward(self, x):
        f = self.backbone(x)
        return self.head_bin(f), self.head_met(f), self.head_face(f), self.head_head(f), self.head_full(f)


def _infer_head_sizes_from_ckpt_state(ckpt_model_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    sizes = {"num_methods": 0, "num_face_classes": 1, "num_head_classes": 1, "num_full_classes": 1}
    if "head_met.1.weight" in ckpt_model_state: sizes["num_methods"] = ckpt_model_state["head_met.1.weight"].shape[0]
    if "head_face.1.weight" in ckpt_model_state: sizes["num_face_classes"] = ckpt_model_state["head_face.1.weight"].shape[0]
    if "head_head.1.weight" in ckpt_model_state: sizes["num_head_classes"] = ckpt_model_state["head_head.1.weight"].shape[0]
    if "head_full.1.weight" in ckpt_model_state: sizes["num_full_classes"] = ckpt_model_state["head_full.1.weight"].shape[0]
    return sizes


def _filter_state_dict_by_shape(dst_state: Dict[str, torch.Tensor], src_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in src_state.items() if k in dst_state and dst_state[k].shape == v.shape}


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def discover_checkpoints() -> List[str]:
    roots = ["deepfake_detector/models", "backend/models", "models"]
    found = []
    for root in roots:
        if not os.path.isdir(root): continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".pt"):
                    found.append(os.path.join(dirpath, fn))
    found.sort()
    return found


def load_single_detector(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    model_state = ckpt.get("model", {})
    ema_state = ckpt.get("ema", None)

    head_sizes = _infer_head_sizes_from_ckpt_state(model_state)
    num_methods = head_sizes["num_methods"] or len(meta.get("method_names", [])) or 7

    model_name = meta.get("backbone_model") or meta.get("model_name") or "vit_base_patch16_384"
    img_size = int(meta.get("img_size", 384))

    model = MultiHeadViT(
        model_name, img_size, num_methods,
        head_sizes["num_face_classes"],
        head_sizes["num_head_classes"],
        head_sizes["num_full_classes"]
    ).to(device)

    model.eval()

    # Load weights trước (FP32) — bắt buộc trước khi chuyển FP16
    dst = model.state_dict()
    if ema_state:
        dst.update(_filter_state_dict_by_shape(dst, ema_state))
    dst.update(_filter_state_dict_by_shape(dst, model_state))
    model.load_state_dict(dst, strict=False)

    # FP16 SAU khi load weights xong
    if device.type == "cuda":
        try:
            model = model.half()
        except Exception as e:
            print(f"[WARN] FP16 failed: {e}")

    arch_family = _infer_arch_family(model_name)
    arch_type = "swin" if arch_family == "swin" else "vit" if arch_family == "vit" else "cnn"

    tfm = build_transform(img_size)
    method_names = meta.get("method_names", [f"method_{i}" for i in range(num_methods)])

    return {
        "model": model,
        "transform": tfm,
        "device": device,
        "method_names": method_names,
        "img_size": img_size,
        "best_thr": float(ckpt.get("best_thr", meta.get("threshold", meta.get("best_thr", 0.818)))),
        "ckpt_path": ckpt_path,
        "model_name": model_name,
        "arch_family": _infer_arch_family(model_name),
        "arch_type": arch_type,
    }


def load_multiple_detectors(ckpt_paths: List[str], device_name: Optional[str] = None):
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    return [load_single_detector(p, device) for p in ckpt_paths]