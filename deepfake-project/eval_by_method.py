# eval_by_method.py
# Đo độ chính xác theo từng kiểu giả (DeepFakeDetection, Face2Face, ...), dùng checkpoint detector.
# Ví dụ:
#   py eval_by_method.py
#   py eval_by_method.py --ckpt deepfake_detector/checkpoints/detector_best.pt --data_root data/processed/faces

import os, argparse, numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import torch
from torchvision import transforms, datasets
import timm

METHODS = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

def load_detector(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    model_name = meta.get("model_name", "vit_base_patch16_224")
    img_size   = int(meta.get("img_size", 224))
    mean = meta.get("norm_mean", [0.5,0.5,0.5]); std = meta.get("norm_std", [0.5,0.5,0.5])

    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["model"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    thr_meta = float(meta.get("threshold", 0.5))
    return model, tfm, device, thr_meta

@torch.no_grad()
def predict_batch(model, x):
    logits = model(x)
    logits = (logits + model(torch.flip(x, dims=[3]))) / 2
    probs = torch.softmax(logits, dim=1)
    return probs[:,0]  # fake

def infer_method_from_path(path: str) -> str:
    p = path.replace("\\","/")
    if "/val/real/" in p: return "real"
    for m in METHODS:
        if f"/val/fake/{m}/" in p or f"/fake/{m}/" in p:
            return m
    # preprocess cũ: thử đọc từ tên file
    low = os.path.basename(path).lower()
    for m in METHODS:
        if m.lower() in low: return m
    return "fake_other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed/faces")
    ap.add_argument("--ckpt", type=str, default="deepfake_detector/checkpoints/detector_best.pt")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--thr", type=float, default=-1.0, help="ngưỡng cố định; <0 = tự tìm tối ưu theo bacc")
    args = ap.parse_args()

    model, tfm, device, thr_meta = load_detector(args.ckpt)
    ds = datasets.ImageFolder(os.path.join(args.data_root, "val"),
                              transform=tfm)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers,
                                     pin_memory=True, persistent_workers=(args.workers>0))
    paths = [p for (p, _) in ds.samples]
    labels = np.array([y for (_, y) in ds.samples])  # 0=fake,1=real
    probs = []

    for x,_ in dl:
        x = x.to(device, non_blocking=True)
        probs.append(predict_batch(model, x).detach().cpu().numpy())
    probs = np.concatenate(probs, axis=0)  # p_fake

    # tìm threshold tốt nhất (balanced acc) nếu không chỉ định
    if args.thr < 0:
        best_thr, best_bacc = 0.5, 0.0
        for thr in np.linspace(0.40, 0.80, 81):
            pred_fake = (probs >= thr).astype(int)
            y_fake = (labels == 0).astype(int)
            tp = ((pred_fake==1) & (y_fake==1)).sum()
            tn = ((pred_fake==0) & (y_fake==0)).sum()
            fp = ((pred_fake==1) & (y_fake==0)).sum()
            fn = ((pred_fake==0) & (y_fake==1)).sum()
            tpr = tp / max(tp+fn, 1); tnr = tn / max(tn+fp, 1)
            bacc = 0.5*(tpr+tnr)
            if bacc > best_bacc:
                best_bacc, best_thr = bacc, thr
        thr = best_thr
    else:
        thr = float(args.thr)

    pred_fake = (probs >= thr).astype(int)
    y_fake    = (labels == 0).astype(int)

    # Tổng quan
    tp = int(((pred_fake==1) & (y_fake==1)).sum())
    tn = int(((pred_fake==0) & (y_fake==0)).sum())
    fp = int(((pred_fake==1) & (y_fake==0)).sum())
    fn = int(((pred_fake==0) & (y_fake==1)).sum())

    acc = (tp+tn)/max(len(labels),1)
    tpr = tp/max(tp+fn,1); tnr = tn/max(tn+fp,1); bacc = 0.5*(tpr+tnr)

    print(f"=== Overall ===")
    print(f"Threshold: {thr:.3f} (ckpt meta: {thr_meta:.3f})")
    print(f"Acc: {acc:.4f} | BAcc: {bacc:.4f} | TPR(fake): {tpr:.4f} | TNR(real): {tnr:.4f}")
    print(f"CM: tp={tp} tn={tn} fp={fp} fn={fn}")
    print()

    # Theo method
    methods = [infer_method_from_path(p) for p in paths]
    method_idx = defaultdict(list)
    for i, m in enumerate(methods):
        method_idx[m].append(i)

    real_idx = method_idx.get("real", [])
    tnr_real = float((pred_fake[real_idx]==0).sum()) / max(len(real_idx),1) if real_idx else float('nan')

    print("=== Per-method (TPR trên ảnh fake của method đó) ===")
    for m in METHODS:
        idx = [i for i in method_idx.get(m, []) if labels[i]==0]
        if not idx: 
            print(f"{m:18s}: n=0")
            continue
        tpr_m = float((pred_fake[idx]==1).sum())/len(idx)
        bacc_m = 0.5*(tpr_m + (tnr_real if not np.isnan(tnr_real) else 0.0))
        print(f"{m:18s}: n={len(idx):5d} | TPR={tpr_m:6.3f} | BAcc*={bacc_m:6.3f}")

    if real_idx:
        print(f"\n(real): n={len(real_idx)} | TNR={tnr_real:.3f}")

if __name__ == "__main__":
    main()