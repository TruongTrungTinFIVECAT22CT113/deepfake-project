import os
import glob
import json
import torch
import torch.nn as nn
import timm
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Backbone-agnostic multi-head =====
class MultiHeadBackbone(nn.Module):
    def __init__(self, model_name="vit_base_patch16_384", num_methods=8, img_size=512, drop_rate=0.0):
        super().__init__()
        backbone = None
        last_err = None
        # Thử nhiều cấu hình để mọi backbone của timm đều khởi tạo được
        for kwargs in [
            dict(pretrained=False, num_classes=0, drop_rate=drop_rate, img_size=img_size),
            dict(pretrained=False, num_classes=0, drop_rate=drop_rate),
            dict(pretrained=False, num_classes=0),
        ]:
            try:
                backbone = timm.create_model(model_name, **kwargs)
                break
            except TypeError as e:
                last_err = e
                continue
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
        self.head_bin = nn.Linear(feat_dim, 2)            # fake/real
        self.head_m = nn.Linear(feat_dim, num_methods)    # method

    def forward(self, x):
        f = self.backbone(x)
        f = self.dropout(f)
        return self.head_bin(f), self.head_m(f)

def _remap_heads(state):
    out = {}
    for k, v in state.items():
        if k.startswith("head_cls."):
            out[k.replace("head_cls.", "head_bin.")] = v
        elif k.startswith("head_mth."):
            out[k.replace("head_mth.", "head_m.")] = v
        else:
            out[k] = v
    return out

def _read_thr_sidecar(ckpt_path: str):
    """Đọc threshold từ file cạnh checkpoint: calibration.json (key 'threshold') hoặc thr.txt"""
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

# ===== Load 1 checkpoint =====
def load_detector(ckpt_path: str):
    """Nạp đúng checkpoint được chỉ định (không đổi tên/đường dẫn)."""
    ckpt_path = os.path.normpath(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})

    default_methods = [
        "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap",
        "NeuralTextures", "Transformer", "Diffusion", "ImprovedDeepFake"
    ]
    classes = meta.get("classes", ["fake", "real"])
    method_names = meta.get("method_names", default_methods)
    mean = meta.get("norm_mean", [0.485, 0.456, 0.406])
    std = meta.get("norm_std", [0.229, 0.224, 0.225])
    thr_meta = meta.get("threshold", None)
    thr_side = _read_thr_sidecar(ckpt_path)
    thr = float(thr_meta if thr_meta is not None else (thr_side if thr_side is not None else 0.5))
    model_name = meta.get("model_name", "vit_base_patch16_384")
    img_size = int(meta.get("img_size", 512))

    state = ckpt.get("model", ckpt)
    state = _remap_heads(state)

    # Ước lượng img_size nếu meta thiếu
    if img_size <= 0:
        pos = None
        for k in ["backbone.pos_embed", "pos_embed", "module.backbone.pos_embed"]:
            if k in state:
                pos = state[k]; break
        if pos is not None and hasattr(pos, "shape") and len(pos.shape) == 3:
            n_tokens = int(pos.shape[1]); g = int(round((n_tokens - 1) ** 0.5))
            if "patch16" in model_name: patch = 16
            elif "patch14" in model_name: patch = 14
            elif "patch32" in model_name: patch = 32
            else: patch = 16
            img_size = int(g * patch)
        else:
            img_size = 224

    model = MultiHeadBackbone(model_name, img_size=img_size, num_methods=len(method_names), drop_rate=0.0)
    try:
        model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"Không nạp được state_dict: {e}\n"
            f"Kiểm tra: model_name='{model_name}', img_size={img_size}, checkpoint='{ckpt_path}'."
        )

    model.eval().to(DEVICE)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, tfm, DEVICE, classes, method_names, img_size, thr

# ===== Load nhiều checkpoint (ensemble) =====
def load_multiple_detectors(ckpt_paths):
    detectors = []
    for path in ckpt_paths:
        model, tfm, device, classes, method_names, img_size, thr = load_detector(path)
        detectors.append((model, tfm, device, classes, method_names, img_size, thr))
    # Kiểm tra nhất quán
    if len(detectors) > 1:
        ref_methods = detectors[0][4]
        ref_classes = detectors[0][3]
        ref_img_size = detectors[0][5]
        for i, (_, _, _, classes, method_names, img_size, _) in enumerate(detectors[1:], 1):
            if method_names != ref_methods:
                raise ValueError(f"Checkpoint {ckpt_paths[i]} có method_names khác: {method_names} != {ref_methods}")
            if classes != ref_classes:
                raise ValueError(f"Checkpoint {ckpt_paths[i]} có classes khác: {classes} != {ref_classes}")
            if img_size != ref_img_size:
                raise ValueError(f"Checkpoint {ckpt_paths[i]} có img_size khác: {img_size} != {ref_img_size}")
    return detectors

# ===== Auto-discovery =====
def _candidate_roots():
    """CWD + project dir (dù chạy ở đâu cũng quét được)."""
    roots = []
    cwd = os.getcwd()
    roots.append(os.path.join(cwd, "deepfake_detector", "models"))
    roots.append(os.path.join(cwd, "models"))

    here = os.path.dirname(os.path.abspath(__file__))         # components/
    proj = os.path.abspath(os.path.join(here, os.pardir))     # project root
    roots.append(os.path.join(proj, "deepfake_detector", "models"))
    roots.append(os.path.join(proj, "models"))

    roots = [os.path.normpath(r) for r in roots]
    return list(dict.fromkeys([r for r in roots if os.path.isdir(r)]))

def discover_checkpoints():
    """Ưu tiên detector_best.pt; nếu không có thì *.pt."""
    patterns = ["**/detector_best.pt", "**/*.pt"]
    found = []
    for root in _candidate_roots():
        for pat in patterns:
            found += [os.path.normpath(p) for p in glob.glob(os.path.join(root, pat), recursive=True) if os.path.isfile(p)]
    found_best = [p for p in found if p.endswith(os.path.join("", "detector_best.pt"))]
    others = [p for p in found if p not in found_best]
    return list(dict.fromkeys(found_best + others))
