# deepfake_detector/src/train_vit.py  — face-only dataset loader (no head/full scanning), EMA như train_gru.py
import os, re, time, json, math, random, signal, argparse, csv, gc
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from PIL import Image
from glob import glob
from tqdm import tqdm
import timm
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

BRANCH2ID = {"face": 0, "head": 1, "full": 2}
ID2BRANCH = {v:k for k,v in BRANCH2ID.items()}

# -------------- utils --------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(root: str, exts=(".jpg",".jpeg",".png",".bmp",".webp")):
    files=[]
    for e in exts:
        files += glob(os.path.join(root, f"**/*{e}"), recursive=True)
    files.sort(key=natural_key)
    return files

def parse_method_boost(s: str) -> Dict[str, float]:
    out={}
    if not s: return out
    for p in s.split(","):
        p=p.strip()
        if not p: continue
        if "=" in p:
            k,v=p.split("=",1)
            try: out[k.strip()]=float(v.strip())
            except: pass
    return out

# -------------- dataset (FACE-ONLY, new layout) --------------
class MultiBranchDataset(Dataset):
    """
    New layout (face-only):
      data_root/
        face/
          train|val|test/
            real/<DatasetName>/<video-id>/000000.jpg ...
            fake/<MethodName>/<video-id>/000000.jpg ...

    - Binary label (yb): real=1, fake=0
    - Method label (ym): index in union of fake methods (for fake only), else -1
    - Branch id (ybr): always 0 ('face') for all samples
    - Branch-class (ybcls): multi-class over ["real_face"] + sorted(face methods) for face-branch
    """
    def __init__(self, data_root: str, split: str, tfm):
        self.data_root = data_root
        self.split = split
        self.tfm = tfm

        self.samples: List[Tuple[str,int,int,int,int]] = []  # (path, y_bin, y_met, y_branch, y_branch_cls)
        self.branch_methods = { 'face': set(), 'head': set(), 'full': set() }

        split_dir = os.path.join(data_root, "face", split)

        # 1) REAL (face/…/real/<DatasetName>/<video-id>/*.jpg)
        real_root = os.path.join(split_dir, "real")
        if os.path.isdir(real_root):
            for ds in sorted([d for d in os.listdir(real_root) if os.path.isdir(os.path.join(real_root, d))]):
                ds_dir = os.path.join(real_root, ds)
                for img in list_images(ds_dir):
                    # real sample → yb=1, ym=-1, branch=face
                    self.samples.append((img, 1, -1, BRANCH2ID["face"], 0))  # ybcls tạm set 0, cập nhật sau

        # 2) FAKE (face/…/fake/<Method>/<video-id>/*.jpg)
        fake_root = os.path.join(split_dir, "fake")
        face_methods = []
        if os.path.isdir(fake_root):
            face_methods = sorted([d for d in os.listdir(fake_root) if os.path.isdir(os.path.join(fake_root, d))])
            for mname in face_methods:
                self.branch_methods['face'].add(mname)
                m_dir = os.path.join(fake_root, mname)
                for img in list_images(m_dir):
                    self.samples.append((img, 0, mname, BRANCH2ID["face"], -1))  # ybcls cập nhật sau

        # union method set
        self.method_names = sorted(self.branch_methods['face'])
        self.method_to_idx = {m:i for i,m in enumerate(self.method_names)}

        # branch classes (face-only); head/full để rỗng để giữ tương thích code khác
        self.branch_classnames = {
            'face': ["real_face"] + self.method_names,
            'head': [],
            'full': [],
        }
        self.branch_class_to_idx = {
            'face': {n:i for i,n in enumerate(self.branch_classnames['face'])},
            'head': {},
            'full': {},
        }

        # finalize y_branch_cls
        finalized=[]
        for p, yb, ym_name, ybr, _ybcls in self.samples:
            if yb == 1:
                ybcls = self.branch_class_to_idx['face']["real_face"]
                finalized.append((p, 1, -1, ybr, ybcls))
            else:
                ym = self.method_to_idx[str(ym_name)]
                ybcls = self.branch_class_to_idx['face'][str(ym_name)]
                finalized.append((p, 0, ym, ybr, ybcls))
        self.samples = sorted(finalized, key=lambda t: natural_key(t[0]))

        # stats
        self.per_class = {"real_face":0}
        for m in self.method_names: self.per_class[m]=0
        for _, yb, ym, ybr, ybcls in self.samples:
            if yb==1: self.per_class["real_face"] += 1
            else:     self.per_class[self.method_names[ym]] += 1

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, yb, ym, ybr, ybcls = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.tfm(im)
        return (
            x,
            torch.tensor(yb, dtype=torch.long),
            torch.tensor(ym, dtype=torch.long),
            torch.tensor(ybr, dtype=torch.long),
            torch.tensor(ybcls, dtype=torch.long),
        )

# -------------- model --------------
class MultiHeadViT(nn.Module):
    def __init__(self, model_name: str, img_size: int, num_methods: int,
                 num_face_classes: int, num_head_classes: int, num_full_classes: int,
                 drop_rate: float=0.0, drop_path_rate: float=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=img_size,
            drop_rate=drop_rate, drop_path_rate=drop_path_rate
        )
        feat = self.backbone.num_features
        def head(n):
            return nn.Sequential(nn.Dropout(p=drop_rate if drop_rate>0 else 0.0),
                                 nn.Linear(feat, n))
        self.head_bin  = head(2)
        self.head_met  = head(num_methods)
        # face head is used; head/full kept for API compatibility (will be unused if empty)
        self.head_face = head(max(1, num_face_classes))
        self.head_head = head(max(1, num_head_classes))
        self.head_full = head(max(1, num_full_classes))

    def forward(self, x):
        f = self.backbone(x)
        return ( self.head_bin(f),
                 self.head_met(f),
                 self.head_face(f),
                 self.head_head(f),
                 self.head_full(f) )

# -------------- EMA (giống train_gru.py) --------------
class EMA:
    def __init__(self, model: nn.Module, decay=0.9999):
        self.decay = decay
        # shadow khởi tạo cùng device với tham số model
        self.shadow = {n: p.data.detach().clone()
                       for n,p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n,p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.shadow:
                self.shadow[n] = p.data.detach().clone()
            else:
                # shadow = decay*shadow + (1-decay)*param
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        # Backup weights gốc của model về CPU để khôi phục sau
        self._backup_cpu = {}
        for n,p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._backup_cpu[n] = p.data.detach().cpu()
            # copy shadow (dù shadow đang ở CPU hay GPU) sang đúng device của param
            p.data.copy_(self.shadow[n].to(p.device, non_blocking=True))

    def restore(self, model: nn.Module):
        # Khôi phục lại weights gốc của model sau khi eval xong
        if not hasattr(self, "_backup_cpu"):
            return
        for n,p in model.named_parameters():
            if n in self._backup_cpu:
                p.data.copy_(self._backup_cpu[n].to(p.device, non_blocking=True))
        del self._backup_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_shadow_to_cpu(self):
        # Dời toàn bộ shadow về RAM để giải phóng VRAM (dùng trước khi eval nếu --val_use_ema)
        for n in list(self.shadow.keys()):
            self.shadow[n] = self.shadow[n].cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload_shadow_to_gpu(self, device):
        # Nạp shadow trở lại GPU sau khi eval
        for n in list(self.shadow.keys()):
            self.shadow[n] = self.shadow[n].to(device, non_blocking=True)

# -------------- loss helpers --------------
def focal_ce(logits, targets, weight=None, gamma=0.0, label_smoothing=0.0):
    if gamma<=0.0:
        return F.cross_entropy(logits, targets, weight=weight, label_smoothing=label_smoothing)
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    tgt_logp = logp.gather(1, targets.view(-1,1)).squeeze(1)
    pt = p.gather(1, targets.view(-1,1)).squeeze(1).clamp_min(1e-8)
    focal = ((1-pt)**gamma) * (-tgt_logp)
    if weight is not None:
        w = weight.gather(0, targets)
        focal = focal * w
    if label_smoothing>0.0:
        focal = (1-label_smoothing)*focal + label_smoothing*(-logp.mean(dim=1))
    return focal.mean()

def one_hot(y, n): return F.one_hot(y, num_classes=n).float()

def apply_mix(x, y, mixup_alpha, cutmix_alpha, prob):
    if prob<=0 or (mixup_alpha<=0 and cutmix_alpha<=0): return x, one_hot(y,2), False
    if random.random()>=prob: return x, one_hot(y,2), False
    B,C,H,W = x.shape
    idx = torch.randperm(B, device=x.device)
    if cutmix_alpha>0 and random.random()<0.5:
        lam = random.betavariate(cutmix_alpha, cutmix_alpha); lam=max(0,min(1,lam))
        r = math.sqrt(1-lam)
        cw, ch = int(W*r), int(H*r)
        cx, cy = random.randint(0,W-1), random.randint(0,H-1)
        x1, y1 = max(0,cx-cw//2), max(0,cy-ch//2)
        x2, y2 = min(W, x1+cw), min(H, y1+ch)
        xmix = x.clone()
        xmix[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
        lam2 = 1 - ( (x2-x1)*(y2-y1) / float(W*H) )
        t = lam2*one_hot(y,2)+(1-lam2)*one_hot(y[idx],2)
        return xmix, t, True
    else:
        lam = random.betavariate(mixup_alpha, mixup_alpha); lam=max(0,min(1,lam))
        xmix = lam*x + (1-lam)*x[idx]
        t = lam*one_hot(y,2) + (1-lam)*one_hot(y[idx],2)
        return xmix, t, True

# -------------- eval --------------
@torch.no_grad()
def evaluate(model, loader, device, method_names: List[str],
             thr_min, thr_max, thr_steps,
             method_loss_weight=0.2, label_smoothing=0.0,
             use_ema=False, ema_obj=None, val_tta="none", val_repeat=1,
             cons_bacc_min=0.90, cons_rec_real_min=0.90,
             phase_name="VAL"):
    # Áp EMA như train_gru.py (không dùng backup state_dict nặng VRAM)
    if use_ema and ema_obj is not None:
        ema_obj.apply_to(model)

    model.eval()
    t0=time.time()

    K=thr_steps
    thrs=torch.linspace(thr_min,thr_max,steps=K,device=device)
    correct_fake=torch.zeros(K,dtype=torch.float64,device=device)
    correct_real=torch.zeros(K,dtype=torch.float64,device=device)
    tot_fake=0; tot_real=0

    br_corr={0:torch.zeros(K,dtype=torch.float64,device=device),
             1:torch.zeros(K,dtype=torch.float64,device=device),
             2:torch.zeros(K,dtype=torch.float64,device=device)}
    br_tot={0:0,1:0,2:0}

    method_correct=0; method_total=0
    per_m_correct={m:0 for m in method_names}
    per_m_total  ={m:0 for m in method_names}
    
    for r in range(val_repeat):
        for x, yb, ym, ybr, _ in tqdm(loader, desc=f'[{phase_name} r{r+1}]', dynamic_ncols=True):
            x = x.to(device); yb = yb.to(device); ym = ym.to(device); ybr = ybr.to(device)

            # ---- TTA ----
            if val_tta == "hflip":
                x2 = torch.flip(x, [-1])
                lb1, lm1, _, _, _ = model(x)
                lb2, lm2, _, _, _ = model(x2)
                lb = (lb1 + lb2) * 0.5
                lm = (lm1 + lm2) * 0.5
            elif val_tta == "scale":
                scales = [0.9, 1.0, 1.1]
                outs_b, outs_m = [], []
                for s in scales:
                    hs = int(x.shape[2] * s)
                    ws = int(x.shape[3] * s)
                    xs = F.interpolate(x, size=(hs, ws), mode='bilinear', align_corners=False)
                    if xs.shape[-1] != 384 or xs.shape[-2] != 384:
                        xs = F.interpolate(xs, size=(384, 384), mode='bilinear', align_corners=False)
                    lb_s, lm_s, _, _, _ = model(xs)
                    outs_b.append(lb_s); outs_m.append(lm_s)
                lb = sum(outs_b) / len(outs_b)
                lm = sum(outs_m) / len(outs_m)
            else:
                lb, lm, _, _, _ = model(x)

            # ---- cập nhật metrics ----
            mask_fake = (yb==0) & (ym>=0)
            if mask_fake.any():
                pred_m = lm[mask_fake].argmax(1)
                mt = ym[mask_fake]
                method_correct += int((pred_m==mt).sum().item())
                method_total   += int(mask_fake.sum().item())
                for i in range(mt.numel()):
                    t = int(mt[i]); p = int(pred_m[i])
                    per_m_total[method_names[t]] += 1
                    if p == t: per_m_correct[method_names[t]] += 1

            p_fake = torch.softmax(lb.float(), dim=1)[:,0]
            comp = (p_fake.unsqueeze(1) >= thrs.unsqueeze(0))
            pred_bin = torch.where(comp, torch.zeros_like(yb).unsqueeze(1),
                                   torch.ones_like(yb).unsqueeze(1))
            yexp = yb.unsqueeze(1).expand_as(pred_bin)
            is_fake = (yexp==0); is_real = (yexp==1)
            correct_fake += (pred_bin==0).logical_and(is_fake).sum(0).to(torch.float64)
            correct_real += (pred_bin==1).logical_and(is_real).sum(0).to(torch.float64)
            tot_fake += int((yb==0).sum().item()); tot_real += int((yb==1).sum().item())

            for bid in (0,1,2):
                mask = (ybr==bid)
                if mask.any():
                    br_tot[bid] += int(mask.sum().item())
                    pb = p_fake[mask]
                    compb = (pb.unsqueeze(1) >= thrs.unsqueeze(0))
                    predb = torch.where(compb, torch.zeros_like(yb[mask]).unsqueeze(1),
                                        torch.ones_like(yb[mask]).unsqueeze(1))
                    br_corr[bid] += (predb==yb[mask].unsqueeze(1)).sum(0).to(torch.float64)

    rec_fake=(correct_fake/max(1,tot_fake)).cpu()
    rec_real=(correct_real/max(1,tot_real)).cpu()
    bacc=0.5*(rec_fake+rec_real)
    acc =(correct_fake+correct_real).cpu()/float(tot_fake+tot_real+1e-12)
    best_idx=int(torch.argmax(bacc))
    res={
        "acc": float(acc[best_idx]),
        "bacc": float(bacc[best_idx]),
        "best_thr": float(thrs[best_idx].cpu()),
        "rec_fake": float(rec_fake[best_idx]),
        "rec_real": float(rec_real[best_idx]),
        "method_acc_fake": (method_correct/method_total if method_total>0 else 0.0),
        "acc_face_bin": (float((br_corr[0][best_idx]/br_tot[0]).cpu()) if br_tot[0]>0 else None),
        "acc_head_bin": (float((br_corr[1][best_idx]/br_tot[1]).cpu()) if br_tot[1]>0 else None),
        "acc_full_bin": (float((br_corr[2][best_idx]/br_tot[2]).cpu()) if br_tot[2]>0 else None),
        "per_method_acc": {m:(per_m_correct[m]/per_m_total[m] if per_m_total[m]>0 else None) for m in method_names}
    }

    mask = (bacc>=cons_bacc_min) & (rec_real>=cons_rec_real_min)
    if mask.any():
        idxs=torch.nonzero(mask).squeeze(1)
        best=idxs[torch.argmax(rec_fake[idxs])]
        res.update({
            "cons_thr": float(thrs[best].cpu()),
            "cons_acc": float(acc[best]),
            "cons_bacc": float(bacc[best]),
            "cons_rec_fake": float(rec_fake[best]),
            "cons_rec_real": float(rec_real[best]),
        })

    # Khôi phục weights gốc sau khi eval EMA
    if use_ema and ema_obj is not None:
        ema_obj.restore(model)

    print(f"[KQ {phase_name}] acc={res['acc']:.4f} | bacc={res['bacc']:.4f} | "
          f"rec_fake={res['rec_fake']:.4f} | rec_real={res['rec_real']:.4f} | "
          f"thr*={res['best_thr']:.3f} | m_acc(fake)={res['method_acc_fake']:.4f}")
    return res

# -------------- main --------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--micro_batch", type=int, default=16)
    ap.add_argument("--val_batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--model", default="vit_base_patch16_384") # chạy file check_timm_support.py để xem các model hỗ trợ của timm

    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--val_use_ema", action="store_true")

    # cân bằng
    ap.add_argument("--balance_by_method", action="store_true")
    ap.add_argument("--method_boost", type=str, default="")
    ap.add_argument("--bin_balance_sampler", action="store_true")

    # loss weights
    ap.add_argument("--method_loss_weight", type=float, default=0.2)
    ap.add_argument("--face_head_loss_weight", type=float, default=0.25)
    ap.add_argument("--head_head_loss_weight", type=float, default=0.25)
    ap.add_argument("--full_head_loss_weight", type=float, default=0.25)

    ap.add_argument("--bin_weight_real", type=float, default=1.0)
    ap.add_argument("--focal_gamma", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    # augment
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--mixup_prob", type=float, default=0.0)
    ap.add_argument("--color_jitter", type=float, default=0.3)
    ap.add_argument("--rand_erase_p", type=float, default=0.25)
    ap.add_argument("--random_resized_crop", action="store_true")

    # drops
    ap.add_argument("--drop_rate", type=float, default=0.1)
    ap.add_argument("--drop_path_rate", type=float, default=0.1)

    # optim
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--warmup_steps", type=int, default=800)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    # NEW: cosine decay (disabled if <= 0)
    ap.add_argument("--cosine_min_lr", type=float, default=0.0)

    # threshold sweep
    ap.add_argument("--thr_min", type=float, default=0.55)
    ap.add_argument("--thr_max", type=float, default=0.90)
    ap.add_argument("--thr_steps", type=int, default=101)

    # run control
    ap.add_argument("--resume", default="")
    ap.add_argument("--freeze_epochs", type=int, default=0)
    ap.add_argument("--method_warmup_epochs", type=int, default=0)
    ap.add_argument("--early_stop_patience", type=int, default=3)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--val_tta", choices=["none","hflip","scale"], default="none")
    ap.add_argument("--cons_bacc_min", type=float, default=0.90)
    ap.add_argument("--cons_rec_real_min", type=float, default=0.90)
    # NEW: during freeze phase, keep last N ViT blocks trainable
    ap.add_argument("--unfreeze_last_blocks", type=int, default=0)

    # eval-only
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_split", choices=["val","test"], default="val")
    ap.add_argument("--val_repeat", type=int, default=1, help="Số lần lặp lại validation để lấy trung bình (VD: 2 hoặc 3).")

    args=ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    try: torch.set_float32_matmul_precision("high")
    except: pass

    os.makedirs(args.out_dir, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    tfml=[]
    if args.random_resized_crop:
        tfml.append(transforms.RandomResizedCrop(args.img_size, scale=(0.7,1.0)))
    else:
        tfml.append(transforms.Resize((args.img_size,args.img_size)))
    tfml.append(transforms.RandomHorizontalFlip(0.5))
    if args.color_jitter>0:
        cj=args.color_jitter
        tfml.append(transforms.ColorJitter(cj,cj,cj,min(0.1,cj*0.5)))
    tfml += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if args.rand_erase_p>0:
        tfml.append(transforms.RandomErasing(p=args.rand_erase_p, scale=(0.02,0.2), ratio=(0.3,3.3), value='random'))
    tfm_train=transforms.Compose(tfml)
    tfm_eval =transforms.Compose([transforms.Resize((args.img_size,args.img_size)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    # datasets (now face-only loader)
    ds_train=MultiBranchDataset(args.data_root,'train',tfm_train)
    ds_val  =MultiBranchDataset(args.data_root,'val',  tfm_eval)

    has_any_test = os.path.isdir(os.path.join(args.data_root, 'face', 'test'))
    ds_test = MultiBranchDataset(args.data_root,'test', tfm_eval) if has_any_test else None

    method_names = ds_train.method_names
    print(f"📊 Method union: {method_names}")
    print(f"📚 Branch classes: {{'face': {ds_train.branch_classnames['face']}, 'head': [], 'full': []}}")

    def stat(ds):
        return f"{len(ds)} | " + ", ".join([f"{k}:{v}" for k,v in ds.per_class.items()])
    print(f"🔎 TRAIN: {stat(ds_train)}")
    print(f"🔎 VAL  : {stat(ds_val)}")
    if ds_test is not None: print(f"🔎 TEST : {stat(ds_test)}")

    # sampler
    sampler=None
    if args.balance_by_method:
        boost=parse_method_boost(args.method_boost)
        print(f"⚖️  Balance-by-method ON | boost={boost}")
        w=[]
        for _, yb, ym, ybr, _ in ds_train.samples:
            if yb==1:
                w.append(boost.get("real",1.0))
            else:
                w.append(boost.get(method_names[int(ym)], 1.0))
        sampler=WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    elif args.bin_balance_sampler:
        n_fake=sum(1 for _,yb,_,_,_ in ds_train.samples if yb==0)
        n_real=sum(1 for _,yb,_,_,_ in ds_train.samples if yb==1)
        ratio=max(1.0, (n_fake/float(n_real)) if n_real>0 else 1.0)
        print(f"⚖️  Balance binary ON ×{ratio:.2f} (oversample real)")
        w=[ (ratio if yb==1 else 1.0) for _,yb,_,_,_ in ds_train.samples ]
        sampler=WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    else:
        print("⚖️  No balancing")

    accum_steps=max(1, args.batch_size//args.micro_batch)
    print(f"🧮 Effective batch={args.batch_size} (micro={args.micro_batch}, accum={accum_steps})")

    dl_train=DataLoader(ds_train, batch_size=args.micro_batch, sampler=sampler,
                        shuffle=(sampler is None), num_workers=args.workers,
                        pin_memory=True, drop_last=True, prefetch_factor=4,
                        persistent_workers=(args.workers>0))
    dl_val  =DataLoader(ds_val, batch_size=args.val_batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True, drop_last=False,
                        prefetch_factor=2, persistent_workers=(args.workers>0))
    dl_test=None
    if ds_test is not None and len(ds_test)>0:
        dl_test=DataLoader(ds_test, batch_size=args.val_batch, shuffle=False,
                           num_workers=args.workers, pin_memory=True, drop_last=False,
                           prefetch_factor=2, persistent_workers=(args.workers>0))

    model=MultiHeadViT(
        args.model, args.img_size, num_methods=len(method_names),
        num_face_classes=len(ds_train.branch_classnames['face']),
        num_head_classes=len(ds_train.branch_classnames['head']),
        num_full_classes=len(ds_train.branch_classnames['full']),
        drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate
    ).to(device)

    if args.grad_ckpt and hasattr(model.backbone,"set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)

    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
    ema = EMA(model, 0.9999) if args.ema else None

    # ---- FREEZE helpers (support --unfreeze_last_blocks) ----
    def _set_requires_grad(module: nn.Module, req: bool):
        for p in module.parameters():
            p.requires_grad = req

    def set_backbone_grad(req: bool):
        """
        If req=True: unfreeze ALL backbone params.
        If req=False: freeze backbone, except keep last N transformer blocks (and final norm) trainable
        when args.unfreeze_last_blocks > 0.
        """
        if req:
            _set_requires_grad(model.backbone, True)
            return
        # freeze all first
        _set_requires_grad(model.backbone, False)
        n_keep = max(0, int(args.unfreeze_last_blocks))
        if n_keep <= 0:
            return
        # try to unfreeze last N blocks of ViT
        blocks = getattr(model.backbone, "blocks", None)
        if isinstance(blocks, (list, nn.ModuleList)):
            for b in list(blocks)[-n_keep:]:
                _set_requires_grad(b, True)
        # unfreeze final norm if present
        if hasattr(model.backbone, "norm"):
            _set_requires_grad(model.backbone.norm, True)

    # freeze backbone đầu
    if args.freeze_epochs>0:
        print(f"🧊 Freeze backbone {args.freeze_epochs} epoch đầu (giữ mở {args.unfreeze_last_blocks} block cuối)")
        set_backbone_grad(False)

    start_epoch=1; best_metric=-1.0; epochs_no_improve=0
    if args.resume and os.path.isfile(args.resume):
        ck=torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ck.get('model', ck), strict=False)
        if 'optimizer' in ck:
            try: opt.load_state_dict(ck['optimizer'])
            except: pass
        if 'scaler' in ck:
            try: scaler.load_state_dict(ck['scaler'])
            except: pass
        if 'ema' in ck and ema is not None and isinstance(ck['ema'], dict):
            ema.shadow={k:v.to(device) for k,v in ck['ema'].items()}
        if 'epoch' in ck: start_epoch=int(ck['epoch'])+1
        if 'best_metric' in ck: best_metric=float(ck['best_metric'])
        print(f"↩️  Resume từ {args.resume} | epoch tiếp theo: {start_epoch}")

    # lưu label map cho inference
    with open(os.path.join(args.out_dir,"label_map.json"),"w",encoding="utf-8") as f:
        json.dump({"method_names": method_names,
                   "branch_classes": {'face': ds_train.branch_classnames['face'], 'head': [], 'full': []}},
                  f, ensure_ascii=False, indent=2)

    # eval-only
    if args.eval_only:
        loader = dl_val if args.eval_split=="val" else dl_test
        if loader is None: raise RuntimeError(f"Không có dữ liệu cho split '{args.eval_split}'.")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if ema is not None and args.val_use_ema: ema.offload_shadow_to_cpu()

        evaluate(model, loader, device, method_names,
                 args.thr_min, args.thr_max, args.thr_steps,
                 method_loss_weight=args.method_loss_weight,
                 label_smoothing=args.label_smoothing,
                 use_ema=(args.val_use_ema and ema is not None), ema_obj=ema,
                 val_tta=args.val_tta, val_repeat=args.val_repeat,
                 cons_bacc_min=args.cons_bacc_min, cons_rec_real_min=args.cons_rec_real_min,
                 phase_name=args.eval_split.upper())

        if ema is not None and args.val_use_ema: ema.reload_shadow_to_gpu(device)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return

    # losses
    w_bin=torch.tensor([1.0, float(args.bin_weight_real)], dtype=torch.float, device=device)
    def ce_bin(logits, targets):
        return focal_ce(logits, targets, weight=w_bin, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    def ce_met(logits, targets):
        return F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)

    # ---- LR schedule (warmup → cosine if enabled) ----
    global_step=0
    # compute effective optimizer steps per epoch (after grad accumulation)
    steps_per_epoch = max(1, len(dl_train) // max(1, args.batch_size//args.micro_batch))
    total_steps = steps_per_epoch * args.epochs
    cosine_steps = max(1, total_steps - max(0, args.warmup_steps))

    def set_lr(step):
        # warmup
        if args.warmup_steps>0 and step<args.warmup_steps:
            lr=args.lr*float(step+1)/float(args.warmup_steps)
        else:
            if args.cosine_min_lr and args.cosine_min_lr>0.0:
                # cosine decay from lr -> cosine_min_lr over cosine_steps
                s = min(step - args.warmup_steps, cosine_steps)
                cos_t = 0.5*(1+math.cos(math.pi*float(s)/float(cosine_steps)))
                lr = args.cosine_min_lr + (args.lr - args.cosine_min_lr)*cos_t
            else:
                lr=args.lr
        for pg in opt.param_groups: pg['lr']=lr
        return lr

    # log file init
    csv_path=os.path.join(args.out_dir,"metrics_epoch.csv")
    jsonl_path=os.path.join(args.out_dir,"metrics_epoch.jsonl")
    if not os.path.exists(csv_path):
        with open(csv_path,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow([
                "epoch","time_min",
                # train
                "train_acc","train_bacc","train_rec_fake","train_rec_real","train_thr_star",
                "train_acc_face_bin","train_acc_head_bin","train_acc_full_bin",
                # val
                "val_acc","val_bacc","val_rec_fake","val_rec_real","val_thr_star",
                "val_method_acc_fake","val_acc_face_bin","val_acc_head_bin","val_acc_full_bin",
                # test
                "test_acc","test_bacc","test_rec_fake","test_rec_real","test_thr_star",
                "test_method_acc_fake","test_acc_face_bin","test_acc_head_bin","test_acc_full_bin"
            ])

    interrupted={'flag':False}
    def on_sigint(sig,frame): interrupted['flag']=True
    signal.signal(signal.SIGINT, on_sigint)

    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        if args.freeze_epochs>0 and epoch>args.freeze_epochs:
            set_backbone_grad(True)

        pbar=tqdm(dl_train, dynamic_ncols=True, desc=f"[Epoch {epoch}/{args.epochs}]")
        opt.zero_grad(set_to_none=True)
        t_train0=time.time()

        eff_w_met = 0.0 if epoch<=args.method_warmup_epochs else args.method_loss_weight

        # số liệu train để tóm tắt sau epoch
        logits_b_list=[]; labels_b_list=[]
        logits_face_list=[]; labels_face_list=[]
        logits_head_list=[]; labels_head_list=[]
        logits_full_list=[]; labels_full_list=[]
        logits_met_list=[]; labels_met_list=[]

        for step, batch in enumerate(pbar):
            if interrupted['flag']:
                ck=os.path.join(args.out_dir, f"detector_interrupt_e{epoch}_s{step}.pt")
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                            'scaler':scaler.state_dict(),'ema':(ema.shadow if ema else None),
                            'epoch':epoch,'best_metric':best_metric}, ck)
                print(f"\n⛔ Dừng thủ công. Đã lưu {ck}")
                return

            x,yb,ym,ybr,ybcls = batch
            x=x.to(device); yb=yb.to(device); ym=ym.to(device); ybr=ybr.to(device); ybcls=ybcls.to(device)

            x_in=x; soft_y=None; mixed=False
            if args.mixup_prob>0 and (args.mixup>0 or args.cutmix>0):
                x_in, soft_y, mixed = apply_mix(x,yb,args.mixup,args.cutmix,args.mixup_prob)

            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                lb,lm,lfa,lhe,lfu = model(x_in)

                # binary
                loss_b = ( F.kl_div(F.log_softmax(lb.float(),1), soft_y, reduction='batchmean')
                           if mixed else ce_bin(lb.float(), yb) )

                # method (fake-only)
                loss_m=torch.tensor(0.0, device=device)
                mask_fake=(yb==0) & (ym>=0)
                if mask_fake.any() and eff_w_met>0:
                    loss_m = ce_met(lm[mask_fake].float(), ym[mask_fake])

                # branch CE (only face has classes; head/full empty)
                loss_face=torch.tensor(0.0, device=device)
                loss_head=torch.tensor(0.0, device=device)
                loss_full=torch.tensor(0.0, device=device)

                m_face=(ybr==BRANCH2ID['face'])
                if m_face.any():
                    loss_face = F.cross_entropy(lfa[m_face].float(), ybcls[m_face],
                                                label_smoothing=args.label_smoothing)

                # no samples for head/full -> these masks will be empty
                m_head=(ybr==BRANCH2ID['head'])
                if m_head.any():
                    loss_head = F.cross_entropy(lhe[m_head].float(), ybcls[m_head],
                                                label_smoothing=args.label_smoothing)
                m_full=(ybr==BRANCH2ID['full'])
                if m_full.any():
                    loss_full = F.cross_entropy(lfu[m_full].float(), ybcls[m_full],
                                                label_smoothing=args.label_smoothing)

                loss = ( loss_b
                         + eff_w_met*loss_m
                         + args.face_head_loss_weight*loss_face
                         + args.head_head_loss_weight*loss_head
                         + args.full_head_loss_weight*loss_full )

            # collect for train summary
            with torch.no_grad():
                logits_b_list.append(lb.detach().float().cpu())
                labels_b_list.append(yb.detach().cpu())
                if m_face.any():
                    logits_face_list.append(lfa[m_face].detach().float().cpu())
                    labels_face_list.append(yb[m_face].detach().cpu())
                if m_head.any():
                    logits_head_list.append(lhe[m_head].detach().float().cpu())
                    labels_head_list.append(yb[m_head].detach().cpu())
                if m_full.any():
                    logits_full_list.append(lfu[m_full].detach().float().cpu())
                    labels_full_list.append(yb[m_full].detach().cpu())
                if mask_fake.any():
                    logits_met_list.append(lm[mask_fake].detach().float().cpu())
                    labels_met_list.append(ym[mask_fake].detach().cpu())

            # grad step
            loss = loss / max(1, args.batch_size//args.micro_batch)
            scaler.scale(loss).backward()
            do_step = ((step+1) % max(1,args.batch_size//args.micro_batch))==0
            if do_step:
                if args.clip_grad and args.clip_grad>0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                global_step+=1; set_lr(global_step)
                if ema is not None: ema.update(model)

            if (step%10)==0:
                p_fake=torch.softmax(lb.detach().float(),dim=1)[:,0]
                tmid=(args.thr_min+args.thr_max)/2
                pred=(p_fake>=tmid).long()
                pred=torch.where(pred==1, torch.zeros_like(pred), torch.ones_like(pred))
                acc=(pred==yb).float().mean().item()
                pbar.set_postfix_str(f"loss={loss.item():.4f} acc~={acc:.4f} lr={opt.param_groups[0]['lr']:.2e}")

        # ----- TRAIN summary -----
        with torch.no_grad():
            lb_tr=torch.cat(logits_b_list,0); yb_tr=torch.cat(labels_b_list,0)
            thrs_tr=torch.linspace(args.thr_min,args.thr_max,steps=101)
            probs=torch.softmax(lb_tr,dim=1)[:,0]
            best_bacc=-1; best_acc=0; best_thr=float((args.thr_min+args.thr_max)/2); rf=rr=0
            m_fake=(yb_tr==0); m_real=(yb_tr==1)
            for t in thrs_tr:
                pred=torch.where(probs>=t, torch.zeros_like(yb_tr), torch.ones_like(yb_tr))
                acc=(pred==yb_tr).float().mean().item()
                recf=(pred[m_fake]==0).float().mean().item() if m_fake.any() else 0.0
                recr=(pred[m_real]==1).float().mean().item() if m_real.any() else 0.0
                bacc=0.5*(recf+recr)
                if bacc>best_bacc:
                    best_bacc, best_acc, best_thr, rf, rr = bacc, acc, float(t.item()), recf, recr

            def acc_from_lists(logits_list, labels_list):
                if len(logits_list)==0: return None
                l=torch.cat(logits_list,0); y=torch.cat(labels_list,0)
                pred=l.argmax(1)
                return float((pred==y).float().mean().item())

            acc_face_bin = acc_from_lists(logits_face_list, labels_face_list)
            acc_head_bin = acc_from_lists(logits_head_list, labels_head_list)
            acc_full_bin = acc_from_lists(logits_full_list, labels_full_list)

            macc = None
            if len(logits_met_list)>0:
                lm_tr=torch.cat(logits_met_list,0); ym_tr=torch.cat(labels_met_list,0)
                macc=float((lm_tr.argmax(1)==ym_tr).float().mean().item())

        print(f"[KQ TRAIN] acc={best_acc:.4f} | bacc={best_bacc:.4f} | rec_fake={rf:.4f} | rec_real={rr:.4f} | thr*={best_thr:.3f} | m_acc(fake)={macc}")

        # ----- VAL -----
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if ema is not None and args.val_use_ema: ema.offload_shadow_to_cpu()

        val_res = evaluate(
            model, dl_val, device, method_names,
            args.thr_min, args.thr_max, args.thr_steps,
            method_loss_weight=args.method_loss_weight,
            label_smoothing=args.label_smoothing,
            use_ema=(args.val_use_ema and ema is not None), ema_obj=ema,
            val_tta=args.val_tta, val_repeat=args.val_repeat,
            cons_bacc_min=args.cons_bacc_min, cons_rec_real_min=args.cons_rec_real_min
            , phase_name="VAL"
        )

        # ----- TEST (nếu có) -----
        test_res=None
        if dl_test is not None:
            print("\n[ĐÁNH GIÁ TEST] ——")
            test_res = evaluate(
                model, dl_test, device, method_names,
                args.thr_min, args.thr_max, args.thr_steps,
                method_loss_weight=args.method_loss_weight,
                label_smoothing=args.label_smoothing,
                use_ema=(args.val_use_ema and ema is not None), ema_obj=ema,
                val_tta=args.val_tta, val_repeat=args.val_repeat,
                cons_bacc_min=args.cons_bacc_min, cons_rec_real_min=args.cons_rec_real_min
                , phase_name="TEST"
            )

        # Sau khi xong tất cả eval → nạp EMA trở lại GPU để train tiếp
        if ema is not None and args.val_use_ema: ema.reload_shadow_to_gpu(device)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ----- LOGGING (CSV + JSONL) -----
        row = {
            "epoch": epoch,
            "time_min": round((time.time()-t_train0)/60.0, 3),

            "train_acc": best_acc, "train_bacc": best_bacc,
            "train_rec_fake": rf, "train_rec_real": rr,
            "train_thr_star": best_thr,
            "train_acc_face_bin": acc_face_bin,
            "train_acc_head_bin": acc_head_bin,
            "train_acc_full_bin": acc_full_bin,

            "val_acc": val_res["acc"], "val_bacc": val_res["bacc"],
            "val_rec_fake": val_res["rec_fake"], "val_rec_real": val_res["rec_real"],
            "val_thr_star": val_res["best_thr"],
            "val_method_acc_fake": val_res["method_acc_fake"],
            "val_acc_face_bin": val_res["acc_face_bin"],
            "val_acc_head_bin": val_res["acc_head_bin"],
            "val_acc_full_bin": val_res["acc_full_bin"],

            "test_acc": (None if test_res is None else test_res["acc"]),
            "test_bacc": (None if test_res is None else test_res["bacc"]),
            "test_rec_fake": (None if test_res is None else test_res["rec_fake"]),
            "test_rec_real": (None if test_res is None else test_res["rec_real"]),
            "test_thr_star": (None if test_res is None else test_res["best_thr"]),
            "test_method_acc_fake": (None if test_res is None else test_res["method_acc_fake"]),
            "test_acc_face_bin": (None if test_res is None else test_res["acc_face_bin"]),
            "test_acc_head_bin": (None if test_res is None else test_res["acc_head_bin"]),
            "test_acc_full_bin": (None if test_res is None else test_res["acc_full_bin"]),
        }
        # CSV
        with open(csv_path,"a",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow([row[k] for k in [
                "epoch","time_min",
                "train_acc","train_bacc","train_rec_fake","train_rec_real","train_thr_star",
                "train_acc_face_bin","train_acc_head_bin","train_acc_full_bin",
                "val_acc","val_bacc","val_rec_fake","val_rec_real","val_thr_star",
                "val_method_acc_fake","val_acc_face_bin","val_acc_head_bin","val_acc_full_bin",
                "test_acc","test_bacc","test_rec_fake","test_rec_real","test_thr_star",
                "test_method_acc_fake","test_acc_face_bin","test_acc_head_bin","test_acc_full_bin"
            ]])
        # JSONL
        with open(jsonl_path,"a",encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")

        # ----- SAVE -----
        base_meta = {
            "model_name": args.model,
            "img_size": args.img_size,
            "method_names": method_names,            # giữ nguyên cho FE đọc
        }

        improved = val_res['bacc'] > best_metric
        if improved:
            best_metric = val_res['bacc']; epochs_no_improve = 0
            meta_best = dict(base_meta)
            meta_best.update({
                "threshold": val_res.get("best_thr", None),
                "branch_thresholds": None,
                "threshold_constrained": val_res.get("cons_thr", None),
                "branch_thresholds_constrained": None,
            })
            best_path = os.path.join(args.out_dir, "detector_best.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": (ema.shadow if ema else None),
                "epoch": epoch,
                "best_metric": best_metric,
                "best_thr": val_res.get("best_thr", None),
                "meta": meta_best,
            }, best_path)
            print(f"💾 Lưu BEST → {best_path} (thr*={val_res.get('best_thr', None)})")
        else:
            epochs_no_improve += 1

        meta_ep = dict(base_meta)
        meta_ep.update({
            "threshold": val_res.get("best_thr", None),
            "branch_thresholds": None,
            "threshold_constrained": val_res.get("cons_thr", None),
            "branch_thresholds_constrained": None,
        })
        ep_path = os.path.join(args.out_dir, f"detector_epoch{epoch}.pt")
        torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": (ema.shadow if ema else None),
                "epoch": epoch,
                "best_metric": best_metric,
                "val_summary": val_res,
                "test_summary": test_res,
                "meta": meta_ep,
        }, ep_path)

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"🛑 Early stopping sau {args.early_stop_patience} epoch không cải thiện BACC.")
            break

    print("✅ Train xong.")

if __name__ == "__main__":
    main()
