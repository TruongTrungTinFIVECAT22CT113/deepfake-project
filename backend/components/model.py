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

class MultiHeadViT(nn.Module):
    def __init__(self, model_name: str, img_size: int,
                 num_methods: int, num_face_classes: int, num_head_classes: int, num_full_classes: int,
                 drop_rate: float=0.0, drop_path_rate: float=0.0):
        super().__init__()

        # Chuẩn bị kwargs cho timm.create_model
        backbone_kwargs = dict(
            pretrained=False,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Chỉ những backbone kiểu ViT / Swin / BEiT... mới nhận img_size
        if any(k in model_name.lower() for k in ["vit", "swin", "beit", "deit", "cait"]):
            backbone_kwargs["img_size"] = img_size

        # ConvNeXt (vd: convnext_base.fb_in22k_ft_in1k_384) sẽ KHÔNG bị nhồi img_size nữa
        self.backbone = timm.create_model(
            model_name,
            **backbone_kwargs,
        )
        feat = self.backbone.num_features

        def head(n):
            return nn.Sequential(
                nn.Dropout(p=drop_rate if drop_rate > 0 else 0.0),
                nn.Linear(feat, n)
            )

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
    if "head_met.1.weight"  in ckpt_model_state: sizes["num_methods"]      = ckpt_model_state["head_met.1.weight"].shape[0]
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
    roots = [
        os.path.join("deepfake_detector", "models"),
        os.path.join("backend", "models"),
        "models"
    ]
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

    # heads
    head_sizes = _infer_head_sizes_from_ckpt_state(model_state)
    num_methods       = head_sizes["num_methods"] or len(meta.get("method_names", [])) or 7
    num_face_classes  = head_sizes["num_face_classes"]
    num_head_classes  = head_sizes["num_head_classes"]
    num_full_classes  = head_sizes["num_full_classes"]

    model_name = meta.get("backbone_model") or meta.get("model_name") or "vit_base_patch16_384"
    img_size   = int(meta.get("img_size", 384))
    # sau khi load ckpt = torch.load(...)
    ckpt_best_thr = ckpt.get("best_thr", None)
    meta = ckpt.get("meta", {}) or {}

    best_thr = (
        float(ckpt_best_thr)
        if ckpt_best_thr is not None
        else float(
            meta.get("threshold",
                 meta.get("best_thr",
                     os.environ.get("DF_DEFAULT_THR", 0.818)
                 )
            )
        )
    )


    model = MultiHeadViT(model_name, img_size, num_methods, num_face_classes, num_head_classes, num_full_classes).to(device)
    dst = model.state_dict()
    if ema_state:
        dst.update(_filter_state_dict_by_shape(dst, ema_state))
    dst.update(_filter_state_dict_by_shape(dst, model_state))
    model.load_state_dict(dst, strict=False)
    model.eval()

    tfm = build_transform(img_size)
    method_names = meta.get("method_names", [f"method_{i}" for i in range(num_methods)])

    return {
        "model": model,
        "transform": tfm,
        "device": device,
        "method_names": method_names,
        "img_size": img_size,
        "best_thr": best_thr,
        "ckpt_path": ckpt_path,
        "model_name": model_name,
    }

def load_multiple_detectors(ckpt_paths: List[str], device_name: Optional[str]=None):
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    infos = []
    for p in ckpt_paths:
        infos.append(load_single_detector(p, device))
    return infos
