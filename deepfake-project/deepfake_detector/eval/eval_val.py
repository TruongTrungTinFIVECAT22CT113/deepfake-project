# eval_val.py
# Đánh giá trên split (mặc định: val) với Acc, Balanced-Acc, Confusion Matrix
# Streaming DataLoader (ít RAM), TTA, AMP. Có tùy chọn face-crop và calibrate threshold.

import argparse, os, glob, numpy as np, torch, timm, math
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== (tuỳ chọn) Face crop dùng MediaPipe ======
_FACE_DET = None
def _lazy_face_det():
    global _FACE_DET
    if _FACE_DET is None:
        try:
            import mediapipe as mp
            _FACE_DET = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.3
            )
        except Exception:
            _FACE_DET = False
    return _FACE_DET

def crop_face_pil(img: Image.Image, expand=0.25):
    det = _lazy_face_det()
    if not det:
        return img  # không có mediapipe thì trả ảnh gốc
    import numpy as np
    w, h = img.size
    arr = np.array(img)
    res = det.process(arr[..., ::-1])  # BGR/RGB đều ok
    if not res.detections:
        return img
    # lấy mặt lớn nhất
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

# ====== Model multi-head giống train.py ======
class MultiHeadViT(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_224", img_size=224, num_methods=6):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, img_size=img_size)
        feat_dim = self.backbone.num_features
        self.head_cls = nn.Linear(feat_dim, 2)            # fake/real
        self.head_mth = nn.Linear(feat_dim, num_methods)  # method
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
    return model, tfm, classes, method_names, thr, img_size, ckpt, meta

class ImageListDataset(Dataset):
    def __init__(self, paths, labels, transform, do_face_crop=False):
        self.paths = paths
        self.labels = labels
        self.tfm = transform
        self.do_face_crop = do_face_crop
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self.labels[idx]
        try:
            im = Image.open(p).convert("RGB")
            if self.do_face_crop:
                im = crop_face_pil(im)
        except Exception:
            im = Image.new("RGB", (512,512), (0,0,0))
        return self.tfm(im), y

@torch.no_grad()
def predict_prob_fake(model, xb, amp=True, tta=2):
    if amp and DEVICE.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            lg1, _ = model(xb)
            if tta >= 2:
                lg2, _ = model(torch.flip(xb, dims=[3]))
                lg = (lg1 + lg2) / 2
            else:
                lg = lg1
    else:
        lg1, _ = model(xb)
        if tta >= 2:
            lg2, _ = model(torch.flip(xb, dims=[3]))
            lg = (lg1 + lg2) / 2
        else:
            lg = lg1
    prob_fake = torch.softmax(lg, dim=1)[:, 0]
    return prob_fake

def find_best_threshold(all_probs, all_labels):
    # maximize Balanced Accuracy
    labels = np.array(all_labels).astype(np.int32)  # 0=fake,1=real
    probs = np.array(all_probs, dtype=np.float32)   # prob(fake)
    # sắp xếp uniq probs để quét
    cand = np.unique(probs)
    if len(cand) > 2000:  # downsample để nhanh hơn
        idx = np.linspace(0, len(cand)-1, 2000).astype(int)
        cand = cand[idx]
    best_bacc, best_thr = -1.0, 0.5
    for t in cand:
        pred_fake = (probs >= t).astype(np.int32)      # 1=fake, 0=real
        y_pred = np.where(pred_fake==1, 0, 1)          # 0=fake,1=real
        tp = ((y_pred==0) & (labels==0)).sum()
        tn = ((y_pred==1) & (labels==1)).sum()
        fp = ((y_pred==0) & (labels==1)).sum()
        fn = ((y_pred==1) & (labels==0)).sum()
        rec_fake = tp / max(tp+fn,1)
        rec_real = tn / max(tn+fp,1)
        bacc = 0.5*(rec_fake + rec_real)
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, float(t)
    return best_thr, best_bacc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="deepfake_detector/checkpoints/detector_best.pt")
    ap.add_argument("--data_root", default="data/processed/faces")
    ap.add_argument("--split", choices=["val","train"], default="val")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số ảnh mỗi class (0 = all)")
    ap.add_argument("--tta", type=int, default=2, help="1=off, 2=flip")
    ap.add_argument("--no_amp", action="store_true", help="tắt AMP")
    ap.add_argument("--face_crop", action="store_true", help="bật face crop khi eval (khuyên dùng nếu train có --face_crop)")
    ap.add_argument("--calibrate", action="store_true", help="tìm threshold tối ưu theo BACC")
    ap.add_argument("--out_ckpt", default="", help="nếu muốn lưu ckpt mới với threshold mới")
    args = ap.parse_args()

    model, tfm, classes, method_names, thr, img_size, ckpt, meta = load_model(args.ckpt)

    real_paths = glob.glob(os.path.join(args.data_root, args.split, "real", "*.*"))
    # fake có subfolder theo method
    fake_paths = glob.glob(os.path.join(args.data_root, args.split, "fake", "*", "*.*"))

    if args.limit and args.limit > 0:
        real_paths = real_paths[:args.limit]
        fake_paths = fake_paths[:args.limit]

    paths = fake_paths + real_paths
    labels = [0]*len(fake_paths) + [1]*len(real_paths)  # 0=fake,1=real

    ds = ImageListDataset(paths, labels, tfm, do_face_crop=args.face_crop)
    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    n = len(ds)
    tp = tn = fp = fn = 0
    correct = 0
    all_probs, all_labels = [], []

    for xb, yb in tqdm(dl, desc=f"EVAL {args.split} ({n} imgs) @ {img_size}px"):
        xb = xb.to(DEVICE, non_blocking=True)
        probs = predict_prob_fake(model, xb, amp=(not args.no_amp), tta=args.tta).cpu().numpy()
        yb_np = yb.numpy()
        all_probs.append(probs)
        all_labels.append(yb_np)

        pred_fake_flag = (probs >= thr).astype(np.int32)      # 1=fake, 0=real
        y_pred = np.where(pred_fake_flag==1, 0, 1)            # về 0=fake,1=real

        tp += int(((y_pred==0) & (yb_np==0)).sum())
        tn += int(((y_pred==1) & (yb_np==1)).sum())
        fp += int(((y_pred==0) & (yb_np==1)).sum())
        fn += int(((y_pred==1) & (yb_np==0)).sum())
        correct += int((y_pred == yb_np).sum())

    all_probs = np.concatenate(all_probs) if len(all_probs) else np.array([])
    all_labels = np.concatenate(all_labels) if len(all_labels) else np.array([])

    acc = correct / max(n,1)
    prec_fake = tp / max(tp+fp,1)
    rec_fake  = tp / max(tp+fn,1)
    prec_real = tn / max(tn+fn,1)
    rec_real  = tn / max(tn+fp,1)
    bacc = 0.5*(rec_fake+rec_real)

    print(f"\n== EVAL on {args.split} ==")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Backbone   : {model.backbone.__class__.__name__} | img_size={img_size}")
    print(f"Threshold  : {thr:.3f} | TTA={args.tta} | AMP={'off' if args.no_amp else 'on'} | face_crop={args.face_crop}")
    print(f"Accuracy   : {acc:.4f} | Balanced Acc: {bacc:.4f}  (N={n})\n")
    print("Confusion Matrix (rows=true, cols=pred)  labels: 0=fake, 1=real")
    print(f"            pred=0(fake)  pred=1(real)")
    print(f"true=0(fake)   {tp:8d}      {fn:8d}")
    print(f"true=1(real)   {fp:8d}      {tn:8d}\n")
    print("Per-class metrics")
    print(f"fake: precision={prec_fake:.3f} recall={rec_fake:.3f}")
    print(f"real: precision={prec_real:.3f} recall={rec_real:.3f}")

    if args.calibrate and len(all_probs):
        best_thr, best_bacc = find_best_threshold(all_probs, all_labels)
        print(f"\n🔧 Calibrate threshold → best_thr={best_thr:.3f} | best_bacc={best_bacc:.4f}")
        if args.out_ckpt:
            meta["threshold"] = float(best_thr)
            ckpt["meta"] = meta
            os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
            torch.save(ckpt, args.out_ckpt)
            print(f"💾 Saved calibrated checkpoint → {args.out_ckpt}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()