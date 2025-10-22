# -*- coding: utf-8 -*-
"""
Calibration thresholds cho Multi-head ViT (5 head) trên processed_multi.
Tính đồng thời:
  - Global threshold (toàn bộ)
  - Per-branch thresholds (face/head/full)
Cả 2 biến thể: unconstrained & constrained (rec_real >= cons_rec_real_min)

Usage ví dụ:
python -m eval_val.py \
  --ckpt deepfake_detector/models/vitb384_512/checkpoints/detector_best.pt \
  --data_root data/processed_multi --split val \
  --batch 72 --workers 8 --val_tta hflip --ema \
  --thr_min 0.0 --thr_max 1.0 --thr_steps 1001 \
  --cons_rec_real_min 0.90 \
  --out_report deepfake_detector/models/vitb384_512/checkpoints/calib_val.json \
  --out_ckpt   deepfake_detector/models/vitb384_512/checkpoints/detector_best_calib.pt
"""
import os, json, argparse, time, re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from tqdm import tqdm

try:
    import timm
except Exception:
    timm = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

BRANCH2ID = {"face": 0, "head": 1, "full": 2}
ID2BRANCH = {v:k for k,v in BRANCH2ID.items()}


# ---------------- utils ----------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(root: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[str]:
    if not os.path.isdir(root):
        return []
    files=[]
    for e in exts:
        files += glob(os.path.join(root, f"**/*{e}"), recursive=True)
    files.sort(key=natural_key)
    return files

def build_transform(img_size: int):
    from torchvision import transforms as T
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_label_map(label_map_path: Optional[str], data_root: Optional[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    # 1) ưu tiên file label_map.json
    if label_map_path and os.path.isfile(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            lm = json.load(f)
        return lm.get("method_names", []), lm.get("branch_classes", {})
    # 2) nếu cung cấp đường dẫn bất kỳ, thử file cạnh nó
    if label_map_path:
        cand = os.path.join(os.path.dirname(label_map_path), "label_map.json")
        if os.path.isfile(cand):
            with open(cand, "r", encoding="utf-8") as f:
                lm = json.load(f)
            return lm.get("method_names", []), lm.get("branch_classes", {})
    # 3) fallback: quét từ data_root
    assert data_root and os.path.isdir(data_root), "Need data_root to infer label map"
    def scan_branch(br: str) -> List[str]:
        names = ["real_"+br]
        fake_dir = os.path.join(data_root, br, "val", f"fake_{br}")
        if not os.path.isdir(fake_dir):
            fake_dir = os.path.join(data_root, br, "train", f"fake_{br}")
        if os.path.isdir(fake_dir):
            for d in sorted(os.listdir(fake_dir)):
                if os.path.isdir(os.path.join(fake_dir, d)):
                    names.append(d)
        return names
    branch_classes = {br: scan_branch(br) for br in ["face","head","full"]}
    method_names = sorted({m for br in ["face","head","full"]
                           for m in branch_classes[br] if not m.startswith("real_")})
    return method_names, branch_classes


# ---------------- model ----------------
class MultiHeadViT(nn.Module):
    def __init__(self, model_name: str, img_size: int,
                 num_methods: int, num_face_classes: int, num_head_classes: int, num_full_classes: int,
                 drop_rate: float=0.0, drop_path_rate: float=0.0):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required")
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, img_size=img_size,
            drop_rate=drop_rate, drop_path_rate=drop_path_rate
        )
        feat = getattr(self.backbone, "num_features", 768)
        def head(n): 
            return nn.Sequential(nn.Dropout(p=drop_rate if drop_rate>0 else 0.0),
                                 nn.Linear(feat, n))
        self.head_bin  = head(2)
        self.head_met  = head(num_methods)
        self.head_face = head(num_face_classes)
        self.head_head = head(num_head_classes)
        self.head_full = head(num_full_classes)
    def forward(self, x):
        f = self.backbone(x)
        return (self.head_bin(f), self.head_met(f), self.head_face(f), self.head_head(f), self.head_full(f))


# --------------- EMA helpers ---------------
def extract_ema_shadow(ckpt: dict):
    if ckpt is None: return None
    if 'ema' in ckpt:
        ema_obj = ckpt['ema']
        if isinstance(ema_obj, dict):
            if 'shadow' in ema_obj and isinstance(ema_obj['shadow'], dict):
                return ema_obj['shadow']
            return ema_obj
    if 'state_dict_ema' in ckpt and isinstance(ckpt['state_dict_ema'], dict):
        return ckpt['state_dict_ema']
    return None

@torch.no_grad()
def apply_ema_(model: nn.Module, ema_shadow: dict):
    for n, p in model.named_parameters():
        if p.requires_grad and (n in ema_shadow):
            p.data.copy_(ema_shadow[n].to(p.device, dtype=p.dtype))


# --------------- dataset for calib ---------------
class FlatImageSet(Dataset):
    def __init__(self, items: List[Tuple[str, int, int]], tfm):
        self.items = items  # (path, y_bin, y_branch)
        self.tfm = tfm
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, yb, ybr = self.items[i]
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            im = Image.new("RGB", (512, 512), (0, 0, 0))
        return self.tfm(im), yb, ybr

def collect_items_processed_multi(data_root: str, split: str) -> List[Tuple[str, int, int]]:
    items=[]
    for br in ["face","head","full"]:
        real_dir = os.path.join(data_root, br, split, f"real_{br}")
        for p in list_images(real_dir):
            items.append((p, 1, BRANCH2ID[br]))
        fake_root = os.path.join(data_root, br, split, f"fake_{br}")
        if os.path.isdir(fake_root):
            for m in sorted(os.listdir(fake_root)):
                mdir = os.path.join(fake_root, m)
                if os.path.isdir(mdir):
                    for p in list_images(mdir):
                        items.append((p, 0, BRANCH2ID[br]))
    return items


# --------------- evaluation core ---------------
@torch.no_grad()
def forward_probs(model: nn.Module, loader: DataLoader, val_tta: str="none"):
    all_pb=[]; all_y=[]; all_br=[]
    for xb, yb, ybr in tqdm(loader, ncols=100, desc="[Infer]"):
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        ybr = ybr.to(DEVICE, non_blocking=True)
        if val_tta=="hflip":
            lb1, *_ = model(xb)
            lb2, *_ = model(torch.flip(xb, dims=[-1]))
            lb = 0.5*(lb1+lb2)
        else:
            lb, *_ = model(xb)
        pb = F.softmax(lb, dim=1)[:,0]  # P(fake)
        all_pb.append(pb.detach().cpu())
        all_y.append(yb.detach().cpu())
        all_br.append(ybr.detach().cpu())
    return torch.cat(all_pb,0), torch.cat(all_y,0), torch.cat(all_br,0)

def sweep_threshold(pb: torch.Tensor, y: torch.Tensor, thrs: torch.Tensor):
    best={"thr":0.5,"acc":0.0,"bacc":0.0,"rec_fake":0.0,"rec_real":0.0}
    accs=[]; baccs=[]; recf=[]; recr=[]
    m_fake=(y==0); m_real=(y==1)
    for t in thrs:
        pred=(pb>=t).long()
        pred_bin=torch.where(pred==1, torch.zeros_like(pred), torch.ones_like(pred))
        acc=(pred_bin==y).float().mean().item()
        rf=(pred_bin[m_fake]==0).float().mean().item() if m_fake.any() else 0.0
        rr=(pred_bin[m_real]==1).float().mean().item() if m_real.any() else 0.0
        bacc=0.5*(rf+rr)
        accs.append(acc); baccs.append(bacc); recf.append(rf); recr.append(rr)
        if (bacc>best["bacc"]) or (abs(bacc-best["bacc"])<1e-12 and acc>best["acc"]):
            best.update({"thr":float(t.item()),"acc":acc,"bacc":bacc,"rec_fake":rf,"rec_real":rr})
    return best, accs, baccs, recf, recr

def sweep_threshold_constrained(pb: torch.Tensor, y: torch.Tensor, thrs: torch.Tensor, rec_real_min: float):
    # chọn theo: rec_real >= rec_real_min, tối đa hóa BACC (tie-break ACC)
    best=None
    m_fake=(y==0); m_real=(y==1)
    for t in thrs:
        pred=(pb>=t).long()
        pred_bin=torch.where(pred==1, torch.zeros_like(pred), torch.ones_like(pred))
        acc=(pred_bin==y).float().mean().item()
        rf=(pred_bin[m_fake]==0).float().mean().item() if m_fake.any() else 0.0
        rr=(pred_bin[m_real]==1).float().mean().item() if m_real.any() else 0.0
        if rr+1e-9 < rec_real_min:
            continue
        bacc=0.5*(rf+rr)
        cand={"thr":float(t.item()),"acc":acc,"bacc":bacc,"rec_fake":rf,"rec_real":rr}
        if (best is None) or (cand["bacc"]>best["bacc"]) or (abs(cand["bacc"]-best["bacc"])<1e-12 and cand["acc"]>best["acc"]):
            best=cand
    return best  # có thể None nếu constraint quá cao


# --------------- helpers (printing) ---------------
def fmt_metrics(d: Optional[Dict[str, float]]) -> str:
    if d is None:
        return "None"
    return (
        f"thr*={d['thr']:.4f} | BACC={d['bacc']:.4f} | ACC={d['acc']:.4f} "
        f"| rf={d['rec_fake']:.4f} | rr={d['rec_real']:.4f}"
    )


# --------------- main ---------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", choices=["val","test"], default="val")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--model", type=str, default=None)  # override nếu cần
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--drop_rate", type=float, default=0.0)
    ap.add_argument("--drop_path_rate", type=float, default=0.0)

    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--val_tta", choices=["none","hflip"], default="none")

    ap.add_argument("--thr_min", type=float, default=0.0)
    ap.add_argument("--thr_max", type=float, default=1.0)
    ap.add_argument("--thr_steps", type=int, default=1001)
    ap.add_argument("--cons_rec_real_min", type=float, default=0.0, help="ràng buộc recall Real tối thiểu (vd 0.90)")

    ap.add_argument("--label_map", type=str, default=None)
    ap.add_argument("--out_report", type=str, default=None)
    ap.add_argument("--out_ckpt", type=str, default=None)
    args=ap.parse_args()

    # label map
    lm_try = args.label_map or os.path.join(os.path.dirname(args.ckpt), "label_map.json")
    method_names, branch_classes = load_label_map(
        label_map_path=(lm_try if os.path.isfile(lm_try) else None),
        data_root=args.data_root
    )

    # model & ckpt
    ck=torch.load(args.ckpt, map_location="cpu")
    meta=ck.get("meta", {})
    model_name = args.model or meta.get("model_name", "vit_base_patch16_384")
    img_size   = int(meta.get("img_size", args.img_size))

    if timm is None:
        raise RuntimeError("timm is required")

    model=MultiHeadViT(
        model_name, img_size,
        num_methods=len(method_names),
        num_face_classes=len(branch_classes['face']),
        num_head_classes=len(branch_classes['head']),
        num_full_classes=len(branch_classes['full']),
        drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate
    ).to(DEVICE)

    sd = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"↩️  load_state_dict | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing:   print("    missing(sample):", missing[:6])
    if unexpected:print("    unexpected(sample):", unexpected[:6])

    ema_shadow = None
    if args.ema:
        ema_shadow = ck.get("ema", None)
        if isinstance(ema_shadow, dict) and "shadow" in ema_shadow:
            ema_shadow = ema_shadow["shadow"]
        if ema_shadow is None and "state_dict_ema" in ck:
            ema_shadow = ck["state_dict_ema"]
        if ema_shadow:
            print("🟩 Applying EMA")
            apply_ema_(model, ema_shadow)
        else:
            print("🟨 EMA requested but not found in ckpt")

    # data
    items = collect_items_processed_multi(args.data_root, args.split)
    if len(items)==0:
        raise RuntimeError("Không thấy ảnh nào trong split.")
    tfm=build_transform(img_size)
    y_branch = torch.tensor([b for _,_,b in items], dtype=torch.long)
    ds=FlatImageSet(items, tfm)
    dl=DataLoader(ds, batch_size=args.batch, shuffle=False,
                  num_workers=args.workers, pin_memory=True, drop_last=False)

    # inference
    t0=time.time()
    pb, y, br = forward_probs(model, dl, args.val_tta)  # pb=P(fake), y∈{0,1}, br∈{0,1,2}
    thrs=torch.linspace(args.thr_min, args.thr_max, steps=args.thr_steps)

    # ---- Global sweep (unconstrained + constrained) ----
    best_global, accs, baccs, recf, recr = sweep_threshold(pb, y, thrs)
    best_global_cons = sweep_threshold_constrained(pb, y, thrs, args.cons_rec_real_min)

    # ---- Per-branch sweep ----
    per_branch = {}
    per_branch_cons = {}
    for bid in (0,1,2):
        mask = (br==bid)
        if mask.any():
            best_b, *_ = sweep_threshold(pb[mask], y[mask], thrs)
            per_branch[ID2BRANCH[bid]] = best_b
            best_bc = sweep_threshold_constrained(pb[mask], y[mask], thrs, args.cons_rec_real_min)
            per_branch_cons[ID2BRANCH[bid]] = best_bc
        else:
            per_branch[ID2BRANCH[bid]] = None
            per_branch_cons[ID2BRANCH[bid]] = None

    dt=time.time()-t0

    # ----- print summary -----
    print("\n== CALIB RESULT ==")
    ema_flag = "on" if ema_shadow else "off"
    print(f"Split     : {args.split} | N={len(y)} frames | TTA={args.val_tta} | EMA={ema_flag}")
    print(f"Global    : {fmt_metrics(best_global)}")
    if best_global_cons:
        print(f"Global(c) : {fmt_metrics(best_global_cons)} | cons_rec_real_min={args.cons_rec_real_min}")
    else:
        print(f"Global(c) : None | cons_rec_real_min={args.cons_rec_real_min}")
    for bname in ["face","head","full"]:
        print(f"{bname:>5}     : {fmt_metrics(per_branch[bname])}")
    for bname in ["face","head","full"]:
        line = fmt_metrics(per_branch_cons[bname])
        print(f"{bname:>5}(c)  : {line} | cons_rec_real_min={args.cons_rec_real_min}")
    print(f"Time      : {dt/60.0:.2f} min")

    # ----- build report -----
    report = {
        "ckpt": args.ckpt,
        "data_root": args.data_root,
        "split": args.split,
        "model_name": meta.get("model_name", model_name),
        "img_size": img_size,
        "val_tta": args.val_tta,
        "ema_used": bool(ema_shadow),
        "thr_range": [args.thr_min, args.thr_max, args.thr_steps],
        "cons_rec_real_min": args.cons_rec_real_min,

        "global": best_global,
        "global_constrained": best_global_cons,   # có thể None
        "per_branch": per_branch,                # dict face/head/full
        "per_branch_constrained": per_branch_cons,  # dict, có thể None theo branch
        "N": int(len(y)),
    }

    # ----- save report -----
    if args.out_report:
        Path(os.path.dirname(args.out_report) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved report → {args.out_report}")

    # ----- save calibrated ckpt (embed thresholds) -----
    if args.out_ckpt:
        new_ck = dict(ck)
        new_meta = dict(ck.get("meta", {}))

        # ghi global và per-branch
        new_meta["threshold"] = float(best_global["thr"])
        new_meta["branch_thresholds"] = {b: (None if per_branch[b] is None else float(per_branch[b]["thr"]))
                                         for b in ["face","head","full"]}

        # nếu có bản constrained thì ghi thêm
        if best_global_cons:
            new_meta["threshold_constrained"] = float(best_global_cons["thr"])
        new_meta["branch_thresholds_constrained"] = {
            b: (None if per_branch_cons[b] is None else float(per_branch_cons[b]["thr"]))
            for b in ["face","head","full"]
        }

        new_meta.setdefault("model_name", model_name)
        new_meta.setdefault("img_size", img_size)
        new_ck["meta"] = new_meta

        Path(os.path.dirname(args.out_ckpt) or ".").mkdir(parents=True, exist_ok=True)
        torch.save(new_ck, args.out_ckpt)
        print(f"💾 Saved calibrated ckpt → {args.out_ckpt}")

if __name__ == "__main__":
    main()
