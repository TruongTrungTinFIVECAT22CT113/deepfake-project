# -*- coding: utf-8 -*-
import os, time, argparse, csv
from glob import glob
from PIL import Image
import numpy as np
import torch, torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

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
def predict_p(img, tfm, model, device, tta=2, idx_fake=0, idx_real=1):
    x = tfm(img).unsqueeze(0).to(device)
    use_amp = (device.type=="cuda")
    if use_amp:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            lb, lm = model(x)
            if tta and tta>=2:
                lb2, lm2 = model(torch.flip(x, dims=[3]))
                lb=(lb+lb2)/2; lm=(lm+lm2)/2
    else:
        lb, lm = model(x)
        if tta and tta>=2:
            lb2, lm2 = model(torch.flip(x, dims=[3]))
            lb=(lb+lb2)/2; lm=(lm+lm2)/2
    pbin = torch.softmax(lb, dim=1).squeeze(0).cpu().numpy()
    pm   = torch.softmax(lm, dim=1).squeeze(0).cpu().numpy()
    return float(pbin[idx_fake]), float(pbin[idx_real]), pm

def parse_thr_map(s, methods):
    out={}
    if not s: return out
    for it in s.split(","):
        it=it.strip()
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
    ap.add_argument("--tta", type=int, default=2)
    ap.add_argument("--face_crop", action="store_true")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--per_method_thr", type=str, default="")
    ap.add_argument("--method_gate", type=float, default=0.55)
    ap.add_argument("--save_fp", type=str, default=None)
    ap.add_argument("--out_log", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=2000)
    args = ap.parse_args()

    # REAL set path:
    root = os.path.join("data","processed","faces","val","real")
    paths=[]
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.webp"):
        paths += glob(os.path.join(root, ext))
    n=len(paths)
    if n==0:
        print("No images in REAL set."); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tfm, classes, methods, img_size, thr_meta = load_detector(args.ckpt, device)
    idx_fake = classes.index("fake"); idx_real = classes.index("real")
    thr_global = args.threshold if args.threshold is not None else float(thr_meta)
    thr_map = parse_thr_map(args.per_method_thr, methods)
    gate = float(args.method_gate)

    if args.save_fp:
        os.makedirs(args.save_fp, exist_ok=True)

    t0=time.time()
    pred_real=0; pred_fake=0
    wrong_by_method={}
    pbar=tqdm(total=n, desc="REAL", ncols=100)
    for i,p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except:
            pbar.update(1); continue
        pf, pr, pm = predict_p(img, tfm, model, device, args.tta, idx_fake, idx_real)
        m_idx = int(np.argmax(pm)); m_pred=methods[m_idx]; m_conf=float(pm[m_idx])
        thr_eff = thr_map.get(m_pred, thr_global) if m_conf >= gate else thr_global
        is_fake = (pf >= thr_eff)
        if is_fake:
            pred_fake += 1
            wrong_by_method[m_pred] = wrong_by_method.get(m_pred, 0) + 1
            if args.save_fp:
                fname = os.path.basename(p)
                img.save(os.path.join(args.save_fp, f"{m_pred}_{fname}"))
        else:
            pred_real += 1
        pbar.update(1)
    pbar.close()
    dt=time.time()-t0
    ips=n/dt if dt>0 else 0.0

    print("\n== Summary (REAL set) ==")
    print(f"N={n} | pred_real={pred_real} ({pred_real/n*100:0.2f}%) | pred_fake={pred_fake} ({pred_fake/n*100:0.2f}%) | time={dt/60:0.2f} min | {ips:0.1f} img/s")
    if wrong_by_method:
        print("Dự đoán nhầm sang phương pháp (top-count):")
        top = sorted(wrong_by_method.items(), key=lambda x:-x[1])[:5]
        for k,v in top:
            print(f"  - {k}: {v}")

    if args.out_log is None:
        args.out_log = os.path.join("reports", f"real_eval_{time.strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(os.path.dirname(args.out_log), exist_ok=True)
    with open(args.out_log,"w",encoding="utf-8") as f:
        f.write(f"CKPT: {args.ckpt} | img={img_size} | Thr={thr_global:.3f} | TTA={args.tta} | face_crop={args.face_crop}\n\n")
        f.write(f"N={n} | pred_real={pred_real} ({pred_real/n*100:.2f}%) | pred_fake={pred_fake} ({pred_fake/n*100:.2f}%) | time={dt/60:.2f} min | {ips:.1f} img/s\n")
        if wrong_by_method:
            f.write("Misclassified as methods (top):\n")
            for k,v in top: f.write(f"  - {k}: {v}\n")

if __name__ == "__main__":
    main()
