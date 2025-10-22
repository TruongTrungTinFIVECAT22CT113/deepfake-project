# backend/components/model.py
# -*- coding: utf-8 -*-
import os, glob, json, re
from typing import List
import torch
import torch.nn as nn
import timm
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadBackbone(nn.Module):
    def __init__(self, model_name="vit_base_patch16_384", num_methods=8, img_size=512,
                 drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        last_err = None
        backbone = None
        for kwargs in [
            dict(pretrained=False, num_classes=0, drop_rate=drop_rate, drop_path_rate=drop_path_rate, img_size=img_size),
            dict(pretrained=False, num_classes=0, drop_rate=drop_rate, drop_path_rate=drop_path_rate),
            dict(pretrained=False, num_classes=0, drop_rate=drop_rate),
            dict(pretrained=False, num_classes=0),
        ]:
            try:
                backbone = timm.create_model(model_name, **kwargs)
                break
            except TypeError as e:
                last_err = e
        if backbone is None:
            raise RuntimeError(f"Cannot create backbone for {model_name}: {last_err}")

        self.backbone = backbone
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            with torch.no_grad():
                tmp = torch.zeros(1, 3, 224, 224)
                f = self.backbone(tmp)
                feat_dim = f.shape[-1]

        self.dropout = nn.Dropout(p=0.0)
        self.head_bin = nn.Linear(feat_dim, 2)
        self.head_m   = nn.Linear(feat_dim, num_methods)

    def forward(self, x):
        f = self.backbone(x)
        f = self.dropout(f)
        return self.head_bin(f), self.head_m(f)

# --- helpers ---
def _remap_heads(state: dict) -> dict:
    """Đồng bộ key các phiên bản train khác nhau vào head_bin / head_m."""
    out = {}
    for k, v in state.items():
        if k.startswith("head_cls."):
            out[k.replace("head_cls.", "head_bin.")] = v
        elif k.startswith("head_mth."):
            out[k.replace("head_mth.", "head_m.")] = v
        elif k.startswith("head_met."):
            out[k.replace("head_met.", "head_m.")] = v
        elif k.startswith("head_method."):
            out[k.replace("head_method.", "head_m.")] = v
        elif any(k.startswith(p) for p in ("head_face.", "head_head.", "head_full.")):
            # BE không dùng 3 head này
            continue
        else:
            out[k] = v
    return out

def _read_thr_sidecar(ckpt_path: str):
    d = os.path.dirname(ckpt_path)
    cal_json = os.path.join(d, "calibration.json")
    if os.path.isfile(cal_json):
        try:
            js = json.load(open(cal_json, "r", encoding="utf-8"))
            if "threshold" in js:
                return float(js["threshold"])
        except Exception:
            pass
    thr_txt = os.path.join(d, "thr.txt")
    if os.path.isfile(thr_txt):
        try:
            return float(open(thr_txt, "r", encoding="utf-8").read().strip())
        except Exception:
            pass
    return None

def _infer_n_methods_from_state(state: dict) -> int:
    """
    Bắt mọi biến thể weight của head_m:
      - head_m.weight
      - head_m.1.weight
      - module.head_m.1.weight
    Lấy out_features (dim 0).
    """
    pat = re.compile(r"^(module\.)?head_m(\.\d+)?\.weight$")
    for k, v in state.items():
        if pat.match(k) and hasattr(v, "shape"):
            return int(v.shape[0])
    # fallback an toàn nếu checkpoint không có head_m (rất hiếm)
    return 1

def discover_checkpoints():
    patterns = ["**/detector_best.pt", "**/*.pt"]
    roots = []
    cwd = os.getcwd()
    roots += [os.path.join(cwd, "deepfake_detector", "models"), os.path.join(cwd, "models")]
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.abspath(os.path.join(here, os.pardir))
    roots += [os.path.join(proj, "deepfake_detector", "models"), os.path.join(proj, "models")]
    roots = [os.path.normpath(r) for r in roots if os.path.isdir(r)]

    found = []
    for root in roots:
        for pat in patterns:
            found += [os.path.normpath(p) for p in glob.glob(os.path.join(root, pat), recursive=True) if os.path.isfile(p)]
    found_best = [p for p in found if p.endswith(os.path.join("", "detector_best.pt"))]
    others = [p for p in found if p not in found_best]
    return list(dict.fromkeys(found_best + others))

def load_detector(ckpt_path: str):
    ckpt_path = os.path.normpath(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {}) or {}
    state_raw = ck.get("model", ck)
    state = _remap_heads(state_raw)

    # 1) số lớp method từ weight (bắt mọi pattern head_m(.k)?.weight)
    n_methods = _infer_n_methods_from_state(state)

    # 2) tên method: dùng meta nếu KHỚP số lớp, nếu không thì method_0..N-1
    meta_methods = meta.get("method_names", None)
    if isinstance(meta_methods, list) and len(meta_methods) == n_methods:
        method_names = list(meta_methods)
    else:
        method_names = [f"method_{i}" for i in range(n_methods)]

    mean = meta.get("norm_mean", [0.485, 0.456, 0.406])
    std  = meta.get("norm_std",  [0.229, 0.224, 0.225])
    thr  = meta.get("threshold", None)
    if thr is None:
        thr = _read_thr_sidecar(ckpt_path)
    thr = float(thr if thr is not None else 0.5)

    model_name = meta.get("model_name", "vit_base_patch16_384")
    img_size   = int(meta.get("img_size", 512))
    drop_rate  = float(meta.get("drop_rate", 0.0))
    drop_path_rate = float(meta.get("drop_path_rate", 0.0))

    model = MultiHeadBackbone(
        model_name, num_methods=max(1, n_methods),
        img_size=img_size, drop_rate=drop_rate, drop_path_rate=drop_path_rate
    ).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, tfm, DEVICE, ["fake", "real"], method_names, img_size, thr

def load_multiple_detectors(ckpt_paths: List[str]):
    infos = [load_detector(p) for p in ckpt_paths]
    if len(infos) > 1:
        ref_m = infos[0][4]; ref_size = infos[0][5]
        for i, inf in enumerate(infos[1:], 1):
            if inf[4] != ref_m:
                raise ValueError(f"Method set mismatch between models: {inf[4]} != {ref_m}")
            if inf[5] != ref_size:
                raise ValueError(f"img_size mismatch between models")
    return infos
