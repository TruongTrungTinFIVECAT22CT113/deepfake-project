# eval_by_method.py
# Tính acc theo từng method trên .../val/fake/<Method>/*
# Streaming, AMP, TTA, face-crop (tuỳ chọn) – khớp pipeline train.

import os
import csv
import datetime
import argparse, os, glob, numpy as np, torch, timm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_FACE_DET = None
def _lazy_face_det():
    global _FACE_DET
    if _FACE_DET is None:
        try:
            import mediapipe as mp
            _FACE_DET = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
        except Exception:
            _FACE_DET = False
    return _FACE_DET

def crop_face_pil(img, expand=0.25):
    det = _lazy_face_det()
    if not det:
        return img
    import numpy as np
    w, h = img.size
    arr = np.array(img)
    res = det.process(arr[..., ::-1])
    if not res.detections:
        return img
    best = max(res.detections, key=lambda d: d.location_data.relative_bounding_box.width *
                                             d.location_data.relative_bounding_box.height)
    r = best.location_data.relative_bounding_box
    x0 = int(max(0, (r.xmin - expand) * w))
    y0 = int(max(0, (r.ymin - expand) * h))
    x1 = int(min(w, (r.xmin + r.width  + expand) * w))
    y1 = int(min(h, (r.ymin + r.height + expand) * h))
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0, y0, x1, y1))

class MultiHeadViT(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_224", img_size=224, num_methods=6):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, img_size=img_size)
        feat_dim = self.backbone.num_features
        self.head_cls = nn.Linear(feat_dim, 2)
        self.head_mth = nn.Linear(feat_dim, num_methods)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head_cls(feat), self.head_mth(feat)

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    classes = meta.get("classes", ["fake","real"])
    method_names = meta.get("method_names", ["Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures","Other"])
    mean = meta.get("norm_mean", [0.5,0.5,0.5])
    std  = meta.get("norm_std",  [0.5,0.5,0.5])
    thr  = float(meta.get("threshold", 0.5))
    model_name = meta.get("model_name", "vit_base_patch16_224")
    img_size = int(meta.get("img_size", 224))

    model = MultiHeadViT(model_name, img_size=img_size, num_methods=len(method_names))
    miss, unexp = model.load_state_dict(ckpt["model"], strict=False)
    if miss or unexp:
        print(f"⚠️  state_dict mismatch | missing={len(miss)} unexpected={len(unexp)}")
    model.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return model, tfm, classes, method_names, thr, img_size

class ImageListDataset(Dataset):
    def __init__(self, paths, transform, do_face_crop=False):
        self.paths = paths
        self.tfm = transform
        self.do_face_crop = do_face_crop
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            im = Image.open(p).convert("RGB")
            if self.do_face_crop:
                im = crop_face_pil(im)
        except Exception:
            im = Image.new("RGB", (512,512), (0,0,0))
        return self.tfm(im), p

@torch.no_grad()
def predict_batch(model, xb, amp=True, tta=2):
    if amp and DEVICE.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            lb, lm = model(xb)
            if tta >= 2:
                lb2, lm2 = model(torch.flip(xb, dims=[3]))
                lb = (lb + lb2) / 2
                lm = (lm + lm2) / 2
    else:
        lb, lm = model(xb)
        if tta >= 2:
            lb2, lm2 = model(torch.flip(xb, dims=[3]))
            lb = (lb + lb2) / 2
            lm = (lm + lm2) / 2
    prob_fake = torch.softmax(lb, dim=1)[:, 0]
    pred_meth = torch.argmax(lm, dim=1)
    return prob_fake, pred_meth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="deepfake_detector/checkpoints/detector_best_calib.pt")
    ap.add_argument("--data_root", default="data/processed/faces")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="giới hạn mỗi method (0=all)")
    ap.add_argument("--tta", type=int, default=2)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--face_crop", action="store_true", help="bật face crop khi eval")
    args = ap.parse_args()

    model, tfm, classes, method_names, thr, img_size = load_model(args.ckpt)

    root = os.path.join(args.data_root, args.split, "fake")
    methods = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

    print(f"== EVAL by method on {root} ==")
    total_ok = total_all = 0
    meth_report = {}

    for m in methods:
        mp = os.path.join(root, m)
        paths = sorted(glob.glob(os.path.join(mp, "*.*")))
        if args.limit and args.limit > 0:
            paths = paths[:args.limit]
        if not paths:
            continue

        ds = ImageListDataset(paths, tfm, do_face_crop=args.face_crop)
        dl = DataLoader(
            ds, batch_size=args.batch, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False
        )

        ok_bin = 0; n = 0; ok_m = 0
        for xb, pb in tqdm(dl, desc=f"{m} ({len(paths)} imgs) @ {img_size}px"):
            xb = xb.to(DEVICE, non_blocking=True)
            prob_fake, pred_meth = predict_batch(model, xb, amp=(not args.no_amp), tta=args.tta)
            is_fake_pred = (prob_fake >= thr)
            ok_bin += int(is_fake_pred.sum().item())
            n += len(pb)
            if m in method_names:
                gt_idx = method_names.index(m)
                ok_m += int((pred_meth.cpu().numpy() == gt_idx).sum())

        acc_bin = ok_bin / max(n,1)
        acc_mth = ok_m / max(n,1)
        total_ok += ok_bin; total_all += n
        meth_report[m] = (acc_bin, acc_mth, n)

    print("\n== Summary ==")
    for m,(acc_bin, acc_mth, n) in meth_report.items():
        print(f"{m:15s} | bin_acc={acc_bin:0.4f} | method_acc={acc_mth:0.4f} | N={n}")
    print(f"Overall bin_acc={total_ok/max(total_all,1):0.4f} | N={total_all}")

    # ==== Xuất kết quả ra file CSV vào thư mục reports ====
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"eval_by_method_{ts}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Binary_Accuracy", "Method_Accuracy", "N"])
        for m, (acc_bin, acc_mth, n) in meth_report.items():
            writer.writerow([m, f"{acc_bin:.4f}", f"{acc_mth:.4f}", n])
        writer.writerow([])
        writer.writerow(["Overall", f"{total_ok/max(total_all,1):.4f}", "", total_all])
    print(f"\n[✓] Đã lưu kết quả vào {out_csv}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()