import os, json, argparse, re
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
ID2BRANCH = {v: k for k, v in BRANCH2ID.items()}

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

# --------------- model ----------------
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

# --------------- dataset ---------------
class EvalItem:
    __slots__=("path","y_bin","y_branch","y_method_idx")
    def __init__(self, path, y_bin, y_branch, y_method_idx):
        self.path=path; self.y_bin=y_bin; self.y_branch=y_branch; self.y_method_idx=y_method_idx

class EvalSet(Dataset):
    def __init__(self, items: List[EvalItem], tfm):
        self.items=items; self.tfm=tfm
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it=self.items[i]
        try:
            im=Image.open(it.path).convert("RGB")
        except Exception:
            im=Image.new("RGB",(512,512),(0,0,0))
        return self.tfm(im), it.y_bin, it.y_branch, ( -1 if it.y_method_idx is None else it.y_method_idx )

def collect_items(data_root: str, split: str,
                  method_to_idx: Dict[str,int]) -> List[EvalItem]:
    items=[]
    for br in ["face","head","full"]:
        # real
        real_dir=os.path.join(data_root, br, split, f"real_{br}")
        for p in list_images(real_dir):
            items.append(EvalItem(p, 1, BRANCH2ID[br], None))
        # fake
        fake_root=os.path.join(data_root, br, split, f"fake_{br}")
        if os.path.isdir(fake_root):
            for m in sorted(os.listdir(fake_root)):
                mdir=os.path.join(fake_root, m)
                if os.path.isdir(mdir) and m in method_to_idx:
                    midx=method_to_idx[m]
                    for p in list_images(mdir):
                        items.append(EvalItem(p, 0, BRANCH2ID[br], midx))
    return items

# --------------- metrics ---------------
def bin_metrics_from_probs(pb: torch.Tensor, y: torch.Tensor, thr: float):
    # pb = P(fake), y∈{0,1} (0=fake, 1=real)
    pred = (pb >= thr).long()
    # map: pred==1 -> fake → pred_bin(0=fake,1=real)
    pred_bin = torch.where(pred==1, torch.zeros_like(pred), torch.ones_like(pred))
    acc = (pred_bin==y).float().mean().item()
    m_fake=(y==0); m_real=(y==1)
    rec_fake=(pred_bin[m_fake]==0).float().mean().item() if m_fake.any() else 0.0
    rec_real=(pred_bin[m_real]==1).float().mean().item() if m_real.any() else 0.0
    bacc=0.5*(rec_fake+rec_real)
    return {"acc":acc, "bacc":bacc, "rec_fake":rec_fake, "rec_real":rec_real}

def per_branch_bin_metrics(pb: torch.Tensor, y: torch.Tensor, br: torch.Tensor,
                           thr_global: float, thr_branch: Optional[Dict[str,float]]):
    out={}
    for bid in (0,1,2):
        mask=(br==bid)
        if not mask.any():
            out[ID2BRANCH[bid]] = None
            continue
        thr = thr_global
        if thr_branch and ID2BRANCH[bid] in thr_branch and thr_branch[ID2BRANCH[bid]] is not None:
            thr = float(thr_branch[ID2BRANCH[bid]])
        out[ID2BRANCH[bid]] = bin_metrics_from_probs(pb[mask], y[mask], thr)
    return out

def per_method_accuracy(method_logits: torch.Tensor, y_method_idx: torch.Tensor,
                        method_names: List[str]) -> Dict[str, Optional[float]]:
    # chỉ tính trên mẫu fake có y_method_idx >= 0
    mask = (y_method_idx >= 0)
    if not mask.any():
        return {m: None for m in method_names}
    pred = method_logits[mask].argmax(dim=1)
    y_gt = y_method_idx[mask]
    # accuracy tổng thể từng method
    res={}
    for midx, mname in enumerate(method_names):
        mm = (y_gt==midx)
        if mm.any():
            acc = (pred[mm]==y_gt[mm]).float().mean().item()
            res[mname]=acc
        else:
            res[mname]=None
    # thêm accuracy trung bình (macro) vào khoá đặc biệt
    vals=[v for v in res.values() if isinstance(v, float)]
    res["_macro_avg"]= float(sum(vals)/len(vals)) if vals else None
    return res

# --------------- evaluation core ---------------
@torch.no_grad()
def infer_collect(model: nn.Module, loader: DataLoader, val_tta: str="none"):
    all_pb=[]; all_y=[]; all_br=[]
    all_met_logits=[]
    all_y_method=[]

    for xb, yb, ybr, ym in tqdm(loader, ncols=100, desc="[EVAL]"):
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        ybr= ybr.to(DEVICE, non_blocking=True)

        if val_tta=="hflip":
            lb1, lm1, *_ = model(xb)
            lb2, lm2, *_ = model(torch.flip(xb, dims=[-1]))
            lb = 0.5*(lb1+lb2)
            lm = 0.5*(lm1+lm2)
        else:
            lb, lm, *_ = model(xb)

        pb = F.softmax(lb, dim=1)[:,0]  # P(fake)

        all_pb.append(pb.detach().cpu())
        all_y.append(yb.detach().cpu())
        all_br.append(ybr.detach().cpu())
        all_met_logits.append(lm.detach().cpu())
        all_y_method.append(ym.detach().cpu())

    return ( torch.cat(all_pb,0), torch.cat(all_y,0), torch.cat(all_br,0),
             torch.cat(all_met_logits,0), torch.cat(all_y_method,0) )

# --------------- main ---------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--splits", nargs="+", default=["val","test"], choices=["val","test"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--model", type=str, default=None)  # override nếu cần
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--drop_rate", type=float, default=0.0)
    ap.add_argument("--drop_path_rate", type=float, default=0.0)

    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--val_tta", choices=["none","hflip"], default="none")

    ap.add_argument("--label_map", type=str, default=None)
    ap.add_argument("--out_report", type=str, default=None)
    ap.add_argument("--out_method_csv", type=str, default=None)
    args=ap.parse_args()

    # ----- load ckpt & meta (thresholds) -----
    ck=torch.load(args.ckpt, map_location="cpu")
    meta=ck.get("meta", {})
    model_name = args.model or meta.get("model_name", "vit_base_patch16_384")
    img_size   = int(meta.get("img_size", args.img_size))

    thr_global = float(meta.get("threshold", 0.5))
    thr_branch = meta.get("branch_thresholds", None)  # dict {face/head/full: thr or None}

    # ----- label map -----
    lm_try = args.label_map or os.path.join(os.path.dirname(args.ckpt), "label_map.json")
    method_names, branch_classes = load_label_map(
        label_map_path=(lm_try if os.path.isfile(lm_try) else None),
        data_root=args.data_root
    )
    method_to_idx = {m:i for i,m in enumerate(method_names)}

    # ----- model -----
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

    if args.ema:
        ema_shadow = extract_ema_shadow(ck)
        if ema_shadow:
            print("🟩 Applying EMA")
            apply_ema_(model, ema_shadow)
        else:
            print("🟨 EMA requested but not found in ckpt")

    tfm=build_transform(img_size)

    report={"ckpt": args.ckpt,
            "data_root": args.data_root,
            "model_name": model_name,
            "img_size": img_size,
            "val_tta": args.val_tta,
            "threshold_global": thr_global,
            "thresholds_branch": thr_branch,
            "splits": {}}

    all_method_rows=[]

    for split in args.splits:
        print(f"\n=== SPLIT: {split.upper()} ===")
        items = collect_items(args.data_root, split, method_to_idx)
        if len(items)==0:
            print("!! Không tìm thấy ảnh trong split này.")
            report["splits"][split] = None
            continue

        ds = EvalSet(items, tfm)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True, drop_last=False)

        pb, y, br, met_logits, y_method_idx = infer_collect(model, dl, args.val_tta)

        # Binary metrics: global threshold + per-branch thresholds (nếu có)
        bin_global = bin_metrics_from_probs(pb, y, thr_global)
        bin_branch = per_branch_bin_metrics(pb, y, br, thr_global, thr_branch)

        # Per-method accuracy (head_method)
        method_acc = per_method_accuracy(met_logits, y_method_idx, method_names)

        # Ghi vào report
        split_res = {
            "N": int(y.numel()),
            "binary_global": bin_global,
            "binary_per_branch": bin_branch,
            "method_accuracy": method_acc
        }
        report["splits"][split] = split_res

        # lưu CSV row
        for m in method_names:
            v = method_acc.get(m, None)
            all_method_rows.append({"split": split, "method": m, "acc": ("" if v is None else v)})

        # In tóm tắt
        print(f"[BIN global] ACC={bin_global['acc']:.4f} | BACC={bin_global['bacc']:.4f} "
              f"| rec_fake={bin_global['rec_fake']:.4f} | rec_real={bin_global['rec_real']:.4f}")
        for b in ["face","head","full"]:
            mb = bin_branch[b]
            if mb is None:
                print(f"[BIN {b}] None")
            else:
                print(f"[BIN {b}] ACC={mb['acc']:.4f} | BACC={mb['bacc']:.4f} | rf={mb['rec_fake']:.4f} | rr={mb['rec_real']:.4f}")
        macro = method_acc.get("_macro_avg", None)
        if macro is not None:
            print(f"[METHOD] macro_avg={macro:.4f}")
        else:
            print("[METHOD] macro_avg=None")

    # save JSON
    if args.out_report:
        Path(os.path.dirname(args.out_report) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved JSON report → {args.out_report}")

    # save CSV per-method
    if args.out_method_csv:
        try:
            import csv
            Path(os.path.dirname(args.out_method_csv) or ".").mkdir(parents=True, exist_ok=True)
            with open(args.out_method_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["split","method","acc"])
                w.writeheader()
                for row in all_method_rows:
                    w.writerow(row)
            print(f"💾 Saved method CSV → {args.out_method_csv}")
        except Exception as e:
            print(f"CSV save failed: {e}")

if __name__ == "__main__":
    main()
