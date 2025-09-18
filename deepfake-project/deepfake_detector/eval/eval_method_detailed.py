# -*- coding: utf-8 -*-
import os, time, argparse, math, csv, sys
from glob import glob
from PIL import Image
import numpy as np
import torch, torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

# ---------- utils: load ckpt ----------
def _remap_heads(state):
    out = {}
    for k,v in state.items():
        if k.startswith("head_cls."):
            out[k.replace("head_cls.","head_bin.")] = v
        elif k.startswith("head_mth."):
            out[k.replace("head_mth.","head_m.")] = v
        else:
            out[k] = v
    return out

class MultiHeadViT(nn.Module):
    def __init__(self, backbone, feat_dim, num_methods):
        super().__init__()
        self.backbone = backbone
        self.head_bin = nn.Linear(feat_dim, 2)
        self.head_m   = nn.Linear(feat_dim, num_methods)
    def forward(self, x):
        f = self.backbone(x)
        return self.head_bin(f), self.head_m(f)

def load_backbone(model_name="vit_base_patch16_224", img_size=512, drop_rate=0.0):
    import timm
    last_err = None
    for kw in [
        dict(pretrained=False, num_classes=0, drop_rate=drop_rate, img_size=img_size),
        dict(pretrained=False, num_classes=0, drop_rate=drop_rate),
        dict(pretrained=False, num_classes=0),
    ]:
        try:
            m = timm.create_model(model_name, **kw)
            return m, m.num_features
        except TypeError as e:
            last_err = e
    raise RuntimeError(f"Cannot create backbone: {last_err}")

def load_detector(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    classes = meta.get("classes", ["fake","real"])
    methods = meta.get("method_names", ["Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures","Other"])
    mean = meta.get("norm_mean", [0.5,0.5,0.5]); std = meta.get("norm_std", [0.5,0.5,0.5])
    thr  = float(meta.get("threshold", 0.5))
    model_name = meta.get("model_name", "vit_base_patch16_224")
    img_size   = int(meta.get("img_size", 224))

    bb, feat = load_backbone(model_name, img_size)
    model = MultiHeadViT(bb, feat, len(methods))
    state = _remap_heads(ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, tfm, classes, methods, img_size, thr

@torch.no_grad()
def predict_p(img: Image.Image, tfm, model, device, tta=2, idx_fake=0, idx_real=1):
    x = tfm(img).unsqueeze(0).to(device)
    use_amp = (device.type=="cuda")
    if use_amp:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            lb, lm = model(x)
            if tta and tta>=2:
                lb2, lm2 = model(torch.flip(x, dims=[3]))
                lb = (lb+lb2)/2; lm=(lm+lm2)/2
    else:
        lb, lm = model(x)
        if tta and tta>=2:
            lb2, lm2 = model(torch.flip(x, dims=[3]))
            lb = (lb+lb2)/2; lm=(lm+lm2)/2
    pbin = torch.softmax(lb, dim=1).squeeze(0).cpu().numpy()
    pm   = torch.softmax(lm, dim=1).squeeze(0).cpu().numpy()
    p_fake = float(pbin[idx_fake])
    p_real = float(pbin[idx_real])
    return p_fake, p_real, pm

def parse_thr_map(s: str, methods):
    out = {}
    if not s: return out
    for it in s.split(","):
        it = it.strip()
        if not it or "=" not in it: continue
        k,v = it.split("=",1)
        k=k.strip(); v=v.strip()
        try: v=float(v)
        except: continue
        for name in methods:
            if name.lower()==k.lower():
                out[name]=v; break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True, help="data/processed/faces/val/fake")
    ap.add_argument("--tta", type=int, default=2)
    ap.add_argument("--face_crop", action="store_true")  # giữ để tương thích, nhưng thư mục đã là mặt
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--per_method_thr", type=str, default="")
    ap.add_argument("--method_gate", type=float, default=0.55)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_log", type=str, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0) # chưa dùng; đọc ảnh tuần tự để ổn định
    ap.add_argument("--log_every", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tfm, classes, methods, img_size, thr_meta = load_detector(args.ckpt, device)
    assert "fake" in classes and "real" in classes, f"classes={classes}"
    idx_fake = classes.index("fake"); idx_real = classes.index("real")

    thr_global = float(args.threshold) if args.threshold is not None else float(thr_meta)
    thr_map = parse_thr_map(args.per_method_thr, methods)
    gate = float(args.method_gate)

    method_dirs = [(m, os.path.join(args.root, m)) for m in methods if os.path.isdir(os.path.join(args.root, m))]
    if not method_dirs:
        # fallback: mọi thư mục con
        method_dirs = [(os.path.basename(p), p) for p in sorted(glob(os.path.join(args.root,"*"))) if os.path.isdir(p)]

    total_imgs = 0
    for _,p in method_dirs: total_imgs += len(glob(os.path.join(p,"*.png"))) + len(glob(os.path.join(p,"*.jpg"))) + len(glob(os.path.join(p,"*.jpeg")))

    t0 = time.time()
    rows = []
    log_lines = []
    for mname,mdir in method_dirs:
        paths = []
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.webp"):
            paths += glob(os.path.join(mdir, ext))
        n = len(paths)
        if n==0: continue

        t_m0 = time.time()
        det_fake = 0; correct = 0; wrong = 0; missed = 0
        confmix = {}
        pbar = tqdm(total=n, desc=mname, ncols=100)
        for i, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
            except:
                missed += 1; pbar.update(1); continue
            p_fake, p_real, pm = predict_p(img, tfm, model, device, tta=args.tta, idx_fake=idx_fake, idx_real=idx_real)
            m_idx = int(np.argmax(pm)); m_conf = float(pm[m_idx]); m_pred = methods[m_idx]
            thr_eff = thr_map.get(m_pred, thr_global) if m_conf >= gate else thr_global
            is_fake = (p_fake >= thr_eff)
            if is_fake:
                det_fake += 1
                if m_pred == mname:
                    correct += 1
                else:
                    wrong += 1
                    confmix[m_pred] = confmix.get(m_pred, 0) + 1
            else:
                missed += 1
            pbar.update(1)
        pbar.close()
        dt = time.time() - t_m0
        ips = n / dt if dt>0 else 0.0
        top_conf = sorted(confmix.items(), key=lambda x: -x[1])[:4]
        log_lines.append(f"{mdir} | N={n:6d} | det_fake={det_fake:6d} ({det_fake/n*100:5.1f}%) | correct={correct:6d} ({correct/n*100:5.1f}%) | wrong={wrong:6d} | missed={missed:6d} | time={dt/60:.2f} min | {ips:.1f} img/s")
        if top_conf:
            top_s = ", ".join([f"{k}:{v}" for k,v in top_conf])
            log_lines.append(f"   → nhầm nhiều sang: {top_s}")

        rows.append([mname, n,
                     round(det_fake/n,4), round(correct/n,4), wrong, missed,
                     round(ips,2)])

    total_dt = time.time() - t0
    # print summary
    for ln in log_lines: print(ln)
    print(f"\n[Done] total_time={total_dt/60:.2f} min")

    if args.out_csv is None:
        args.out_csv = os.path.join("reports", f"method_eval_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method","N","det_fake_ratio","correct_ratio","wrong_count","missed_count","imgs_per_sec"])
        for r in rows: w.writerow(r)

    if args.out_log is None:
        args.out_log = args.out_csv.replace(".csv",".log")
    with open(args.out_log, "w", encoding="utf-8") as f:
        for ln in log_lines: f.write(ln+"\n")
        f.write(f"[Done] total_time={total_dt/60:.2f} min\n")

if __name__ == "__main__":
    main()
