# eval_val.py  — đo Acc + Confusion Matrix cho detector (auto lấy img_size từ checkpoint)
import argparse, glob, numpy as np
from PIL import Image
import torch
from torchvision import transforms
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    classes = meta.get("classes", ["fake","real"])
    mean = meta.get("norm_mean", [0.5,0.5,0.5])
    std  = meta.get("norm_std",  [0.5,0.5,0.5])
    thr  = float(meta.get("threshold", 0.5))
    model_name = meta.get("model_name", "vit_base_patch16_224")
    img_size = int(meta.get("img_size", 224))  # 👈 đọc size đã train

    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 👈 resize đúng size
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return model, tfm, classes, thr

@torch.no_grad()
def predict_probs(model, x):
    p1 = model(x)
    p2 = model(torch.flip(x, dims=[3]))  # TTA flip
    p = torch.softmax((p1+p2)/2, dim=1)[:,0]  # prob(fake) = index 0
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="deepfake_detector/checkpoints/detector_best.pt")
    ap.add_argument("--data_root", default="data/processed/faces")
    ap.add_argument("--split", choices=["val","train"], default="val")
    ap.add_argument("--max_per_class", type=int, default=5000, help="giới hạn mỗi lớp (0=không giới hạn)")
    ap.add_argument("--batch", type=int, default=64)  # 384x384 nên để batch eval vừa phải
    args = ap.parse_args()

    model, tfm, classes, thr = load_model(args.ckpt)

    real = glob.glob(f"{args.data_root}/{args.split}/real/*.*")
    fake = glob.glob(f"{args.data_root}/{args.split}/fake/*.*")
    if args.max_per_class > 0:
        real = real[:args.max_per_class]
        fake = fake[:args.max_per_class]

    X, y = [], []
    for p in real: X.append(tfm(Image.open(p).convert("RGB"))); y.append(1)  # 1=real
    for p in fake: X.append(tfm(Image.open(p).convert("RGB"))); y.append(0)  # 0=fake
    X = torch.stack(X); y = np.array(y)

    probs = []
    for i in range(0, len(X), args.batch):
        probs.append(predict_probs(model, X[i:i+args.batch].to(DEVICE)).cpu().numpy())
    probs = np.concatenate(probs)

    # Quyết định theo threshold (prob_fake >= thr -> dự đoán fake)
    pred_fake_flag = (probs >= thr).astype(int)  # 1=fake, 0=real
    # Đổi về nhãn 0=fake,1=real:
    y_pred = np.where(pred_fake_flag==1, 0, 1)

    acc = (y_pred == y).mean()

    # Confusion Matrix theo NHÃN GỐC (0=fake, 1=real)
    tp = ((y_pred==0) & (y==0)).sum()  # đúng: fake
    tn = ((y_pred==1) & (y==1)).sum()  # đúng: real
    fp = ((y_pred==0) & (y==1)).sum()  # báo fake nhầm
    fn = ((y_pred==1) & (y==0)).sum()  # bỏ sót fake

    # Precision/Recall cho từng lớp
    prec_fake = tp / max(tp+fp,1)
    rec_fake  = tp / max(tp+fn,1)
    prec_real = tn / max(tn+fn,1)
    rec_real  = tn / max(tn+fp,1)

    print(f"\n== EVAL on {args.split} ==")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Classes    : {classes}")
    print(f"Threshold  : {thr:.3f}")
    print(f"Accuracy   : {acc:.4f}  (N={len(y)})\n")
    print("Confusion Matrix (rows=true, cols=pred)  labels: 0=fake, 1=real")
    print(f"            pred=0(fake)  pred=1(real)")
    print(f"true=0(fake)   {tp:8d}      {fn:8d}")
    print(f"true=1(real)   {fp:8d}      {tn:8d}\n")
    print("Per-class metrics")
    print(f"fake: precision={prec_fake:.3f} recall={rec_fake:.3f}")
    print(f"real: precision={prec_real:.3f} recall={rec_real:.3f}")

if __name__ == "__main__":
    main()