# -*- coding: utf-8 -*-
import os, argparse, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import timm
except Exception:
    timm = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- utils --------------------
def list_images_recursive(root: str) -> List[str]:
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    root_p = Path(root)
    if not root_p.exists(): return []
    return sorted(str(p) for p in root_p.rglob('*') if p.is_file() and p.suffix.lower() in exts)

def crop_face_pil(im):  # placeholder nếu bạn không cần crop
    return im

def build_transform(img_size: int):
    import torchvision.transforms as T
    # Chuẩn ImageNet
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# -------------------- model (khớp train.py) --------------------
class MultiHeadViT(nn.Module):
    """
    Khớp với train.py:
      - backbone = timm ViT (num_classes=0)
      - head_bin = Dropout -> Linear(feat, 2)  # 0=fake, 1=real
      - head_met = Dropout -> Linear(feat, M)
    """
    def __init__(self, model_name: str, img_size: int, num_methods: int,
                 drop_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        if timm is None:
            raise RuntimeError('timm is required')
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, img_size=img_size,
            drop_rate=drop_rate, drop_path_rate=drop_path_rate
        )
        feat_dim = getattr(self.backbone, 'num_features', 768)
        self.head_bin = nn.Sequential(
            nn.Dropout(p=drop_rate if drop_rate > 0 else 0.0),
            nn.Linear(feat_dim, 2)   # 0=fake, 1=real
        )
        self.head_met = nn.Sequential(
            nn.Dropout(p=drop_rate if drop_rate > 0 else 0.0),
            nn.Linear(feat_dim, num_methods)
        )

    def forward(self, x):
        f = self.backbone(x)
        lb = self.head_bin(f)   # [N,2]
        lm = self.head_met(f)   # [N,M]
        return lb, lm

# -------------------- EMA helper --------------------
def extract_ema_shadow(ckpt: dict):
    """
    Trả về dict tên-tham_số cho EMA shadow nếu có.
    Hỗ trợ các dạng:
      - ckpt['ema'] là dict {param_name: tensor}
      - ckpt['ema'] là dict có key 'shadow'
      - ckpt['state_dict_ema'] (một số code lưu vậy)
    """
    if ckpt is None:
        return None
    if 'ema' in ckpt:
        ema_obj = ckpt['ema']
        if isinstance(ema_obj, dict):
            if 'shadow' in ema_obj and isinstance(ema_obj['shadow'], dict):
                return ema_obj['shadow']
            # có thể chính nó là shadow map
            return ema_obj
    if 'state_dict_ema' in ckpt and isinstance(ckpt['state_dict_ema'], dict):
        return ckpt['state_dict_ema']
    return None

def apply_ema_(model: nn.Module, ema_shadow: dict):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and (n in ema_shadow):
                p.data.copy_(ema_shadow[n].to(p.device, dtype=p.dtype))

# -------------------- loader --------------------
@dataclass
class LoadedModel:
    model: nn.Module
    tfm: any
    classes: List[str]
    methods: List[str]
    img_size: int
    thr: float
    ckpt: dict
    meta: dict
    ema_shadow: Optional[dict]

def load_model(ckpt_path: str, override_model: Optional[str], override_img: Optional[int]) -> LoadedModel:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    meta = ckpt.get('meta', {})
    classes = meta.get('classes', ['fake', 'real'])
    methods = meta.get('methods', ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures'])
    img_size = int(override_img or meta.get('img_size', 512))
    thr = float(meta.get('threshold', 0.5))
    model_name = override_model or meta.get('model_name', 'vit_base_patch16_224')

    model = MultiHeadViT(model_name, img_size, len(methods))
    sd = ckpt.get('model', ckpt)  # fallback nếu ckpt lưu "thẳng"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"↩️  load_state_dict | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing or unexpected:
        print("   missing (sample):", missing[:4])
        print("   unexpected (sample):", unexpected[:4])

    ema_shadow = extract_ema_shadow(ckpt)
    model.to(DEVICE).eval()
    tfm = build_transform(img_size)
    return LoadedModel(model, tfm, classes, methods, img_size, thr, ckpt, meta, ema_shadow)

# -------------------- inference --------------------
def tta_forward(model, x, tta: int):
    with torch.no_grad():
        lb, _ = model(x)                         # [N,2]
        pb = F.softmax(lb, dim=1)[:, 0]         # P(fake)
        if tta and tta > 1:
            xr = torch.flip(x, dims=[3])
            lb2, _ = model(xr)
            pb = 0.5 * (pb + F.softmax(lb2, dim=1)[:, 0])
    return pb

class ImageFolderFlat(Dataset):
    def __init__(self, paths, tfm, face_crop=False):
        self.paths = paths
        self.tfm = tfm
        self.face_crop = face_crop
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        try:
            im = Image.open(p).convert('RGB')
            if self.face_crop:
                im = crop_face_pil(im)
        except Exception:
            im = Image.new('RGB', (512, 512), (0, 0, 0))
        return self.tfm(im), p

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tta', type=int, default=1)
    ap.add_argument('--face_crop', action='store_true')
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--data_root', type=str, default='data/processed/faces/val')
    ap.add_argument('--out_ckpt', type=str, default=None)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=0)
    # override meta nếu thiếu/sai
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--img_size', type=int, default=None)
    # EMA toggle
    ap.add_argument('--ema', action='store_true', help='use EMA shadow weights if present in ckpt')
    args = ap.parse_args()

    lm = load_model(args.ckpt, args.model, args.img_size)

    # áp EMA nếu có và được yêu cầu
    if args.ema and lm.ema_shadow is not None:
        print("🟩 Applying EMA shadow weights")
        apply_ema_(lm.model, lm.ema_shadow)
    else:
        print("🟨 Not using EMA (either --ema not set or ckpt has no EMA)")

    base_thr = lm.thr if args.threshold is None else args.threshold

    fake_paths = list_images_recursive(os.path.join(args.data_root, 'fake'))
    real_paths = list_images_recursive(os.path.join(args.data_root, 'real'))
    N = len(fake_paths) + len(real_paths)
    if N == 0:
        print('No images under', args.data_root)
        return

    def run(paths):
        ds = ImageFolderFlat(paths, lm.tfm, face_crop=args.face_crop)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
        outs = []
        for xb, _ in tqdm(dl, total=len(dl), ncols=100, unit='img'):
            xb = xb.to(DEVICE, non_blocking=True)
            pb = tta_forward(lm.model, xb, args.tta)
            outs.append(pb.detach().cpu())
        return torch.cat(outs, dim=0).numpy()

    t0 = time.time()
    fake_pb = run(fake_paths)
    real_pb = run(real_paths)
    dt = time.time() - t0

    def report(thr):
        fake_pred_fake = int((fake_pb >= thr).sum())
        fake_pred_real = len(fake_pb) - fake_pred_fake
        real_pred_real = int((real_pb < thr).sum())
        real_pred_fake = len(real_pb) - real_pred_real
        acc = (fake_pred_fake + real_pred_real) / (len(fake_pb) + len(real_pb))
        rec_fake = fake_pred_fake / len(fake_pb) if len(fake_pb) > 0 else 0
        rec_real = real_pred_real / len(real_pb) if len(real_pb) > 0 else 0
        bacc = 0.5 * (rec_fake + rec_real)
        return acc, bacc, (fake_pred_fake, fake_pred_real, real_pred_fake, real_pred_real)

    acc, bacc, cm = report(base_thr)
    print("\n== EVAL on val ==")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Backbone   : VisionTransformer | img_size={lm.img_size}")
    print(f"Threshold  : {base_thr:.3f} | TTA={args.tta} | face_crop={args.face_crop}")
    print(f"Accuracy   : {acc:.4f} | Balanced Acc: {bacc:.4f}  (N={N})\n")
    print("Confusion Matrix (rows=true, cols=pred)  labels: 0=fake, 1=real")
    print("            pred=0(fake)  pred=1(real)")
    print(f"true=0(fake)      {cm[0]:5d}         {cm[1]:5d}")
    print(f"true=1(real)      {cm[2]:5d}          {cm[3]:5d}\n")
    prec_fake = cm[0] / max(cm[0] + cm[2], 1)
    rec_fake = cm[0] / max(cm[0] + cm[1], 1)
    prec_real = cm[3] / max(cm[1] + cm[3], 1)
    rec_real = cm[3] / max(cm[2] + cm[3], 1)
    print(f"Per-class metrics\nfake: precision={prec_fake:.3f} recall={rec_fake:.3f}\nreal: precision={prec_real:.3f} recall={rec_real:.3f}\n")

    if args.calibrate:
        # quét mịn vùng bạn hay dùng (0.80-0.99)
        thr_grid = np.linspace(0.80, 0.99, 2001)
        best_thr, best_bacc = base_thr, bacc
        for t in thr_grid:
            _, bb, _ = report(t)
            if bb > best_bacc:
                best_bacc = bb
                best_thr = float(t)
        print(f"🔧 Calibrate threshold → best_thr={best_thr:.3f} | best_bacc={best_bacc:.4f}")
        if args.out_ckpt:
            new_ckpt = dict(lm.ckpt)
            new_meta = dict(lm.meta)
            new_meta['threshold'] = best_thr
            new_meta.setdefault('model_name', lm.meta.get('model_name', 'vit_base_patch16_384'))
            new_meta.setdefault('img_size', lm.meta.get('img_size', lm.img_size))
            new_ckpt['meta'] = new_meta
            Path(Path(args.out_ckpt).parent).mkdir(parents=True, exist_ok=True)
            torch.save(new_ckpt, args.out_ckpt)
            print(f"💾 Saved calibrated checkpoint → {args.out_ckpt}")

if __name__ == '__main__':
    main()