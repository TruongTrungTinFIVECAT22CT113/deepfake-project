# deepfake_detector/src/train_cnn.py
import os, re, time, json, math, random, signal, argparse
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

def natural_key(path: str):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', path)]

def list_images(root: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    out = []
    for e in exts:
        out.extend(glob(os.path.join(root, f"**/*{e}"), recursive=True))
    out.sort(key=natural_key)
    return out

def parse_method_boost(s: str) -> Dict[str, float]:
    out = {}
    if not s:
        return out
    for p in [p.strip() for p in s.split(",") if p.strip()]:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                out[k.strip()] = float(v.strip())
            except:
                pass
    return out

class FacesDataset(Dataset):
    def __init__(self, data_root: str, split: str, methods: List[str], tfm, face_crop: bool = False):
        self.data_root = data_root
        self.split = split
        self.methods = methods
        self.tfm = tfm
        self.face_crop = face_crop

        self.samples: List[Tuple[str, int, int]] = []
        for mi, m in enumerate(methods):
            root_m = os.path.join(data_root, split, "fake", m)
            imgs = list_images(root_m)
            self.samples.extend([(p, 0, mi) for p in imgs])  # 0=fake
        root_r = os.path.join(data_root, split, "real")
        imgs = list_images(root_r)
        self.samples.extend([(p, 1, -1) for p in imgs])     # 1=real
        self.samples.sort(key=lambda t: natural_key(t[0]))

        self.per_class = {"real": 0}
        for m in methods:
            self.per_class[m] = 0
        for _, yb, ym in self.samples:
            if yb == 1:
                self.per_class["real"] += 1
            else:
                self.per_class[methods[ym]] += 1

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, yb, ym = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.tfm(im)
        return x, torch.tensor(yb, dtype=torch.long), torch.tensor(ym, dtype=torch.long)

class MultiHeadCNN(nn.Module):
    """
    CNN backbone từ timm (ConvNeXt/ResNet/…): xuất đặc trưng và gắn 2 head:
      - head_bin: 2 lớp (fake/real)
      - head_met: num_methods lớp (chỉ tính loss trên mẫu fake)
    """
    def __init__(self, model_name: str, img_size: int, num_methods: int,
                 drop_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,           # feature extractor
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            with torch.no_grad():
                tmp = torch.zeros(1, 3, 224, 224)
                f = self.backbone(tmp)
                feat_dim = f.shape[-1]
        self.head_bin = nn.Sequential(nn.Dropout(p=drop_rate if drop_rate>0 else 0.0), nn.Linear(feat_dim, 2))
        self.head_met = nn.Sequential(nn.Dropout(p=drop_rate if drop_rate>0 else 0.0), nn.Linear(feat_dim, num_methods))

    def forward(self, x):
        f = self.backbone(x)
        lb = self.head_bin(f)
        lm = self.head_met(f)
        return lb, lm

class EMA:
    def __init__(self, model: nn.Module, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1.0 - self.decay) * p.data + self.decay * self.shadow[n]

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

def focal_ce(logits: torch.Tensor, targets: torch.Tensor, weight=None, gamma: float = 0.0, label_smoothing: float = 0.0):
    if gamma <= 0.0:
        return F.cross_entropy(logits, targets, weight=weight, label_smoothing=label_smoothing)
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    tgt_logp = logp.gather(1, targets.view(-1, 1)).squeeze(1)
    pt = p.gather(1, targets.view(-1, 1)).squeeze(1).clamp_min(1e-8)
    focal = ((1 - pt) ** gamma) * (-tgt_logp)
    if weight is not None:
        w = weight.gather(0, targets)
        focal = focal * w
    if label_smoothing > 0.0:
        focal = (1 - label_smoothing) * focal + label_smoothing * (-logp.mean(dim=1))
    return focal.mean()

def soft_ce(logits: torch.Tensor, soft_targets: torch.Tensor):
    logp = F.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1).mean()

def rand_bbox(W, H, lam):
    r = math.sqrt(1 - lam)
    cut_w = int(W * r); cut_h = int(H * r)
    cx = random.randint(0, W - 1); cy = random.randint(0, H - 1)
    x1 = max(cx - cut_w // 2, 0); y1 = max(cy - cut_h // 2, 0)
    x2 = min(x1 + cut_w, W); y2 = min(y1 + cut_h, H)
    return x1, y1, x2, y2

def one_hot(labels: torch.Tensor, num_classes: int):
    return F.one_hot(labels, num_classes=num_classes).float()

def apply_mix(x, yb, mixup_alpha: float, cutmix_alpha: float, mix_prob: float):
    if mix_prob <= 0 or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        return x, one_hot(yb, 2), False
    if random.random() >= mix_prob:
        return x, one_hot(yb, 2), False
    B, C, H, W = x.size()
    idx = torch.randperm(B, device=x.device)
    yb2 = yb[idx]
    if cutmix_alpha > 0 and random.random() < 0.5:
        lam = random.betavariate(cutmix_alpha, cutmix_alpha)
        lam = max(0.0, min(1.0, lam))
        x1, y1, x2, y2 = rand_bbox(W, H, lam)
        x_m = x.clone()
        x_m[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        area = abs((x2 - x1) * (y2 - y1))
        lam_adj = 1.0 - float(area) / float(W * H)
        t1 = one_hot(yb, 2); t2 = one_hot(yb2, 2)
        soft_t = lam_adj * t1 + (1 - lam_adj) * t2
        return x_m, soft_t, True
    else:
        lam = random.betavariate(mixup_alpha, mixup_alpha)
        lam = max(0.0, min(1.0, lam))
        x_m = lam * x + (1 - lam) * x[idx]
        t1 = one_hot(yb, 2); t2 = one_hot(yb2, 2)
        soft_t = lam * t1 + (1 - lam) * t2
        return x_m, soft_t, True

@torch.no_grad()
def evaluate(model, loader, device, methods: List[str],
             thr_min: float, thr_max: float,
             method_loss_weight: float = 0.2,
             label_smoothing: float = 0.0,
             use_ema: bool = False,
             ema_obj=None,
             val_tta: str = "none",
             thr_steps: int = 101,
             cons_bacc_min: float = 0.90,
             cons_rec_real_min: float = 0.90):
    backup = None
    if use_ema and (ema_obj is not None) and getattr(ema_obj, "shadow", None):
        backup = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        ema_obj.apply_to(model)

    model.eval()
    t0 = time.time()

    thrs = torch.linspace(thr_min, thr_max, steps=thr_steps, device=device)
    correct_fake = torch.zeros_like(thrs, dtype=torch.float64)
    correct_real = torch.zeros_like(thrs, dtype=torch.float64)
    total_fake = 0; total_real = 0
    sum_loss_bin = 0.0; n_loss = 0
    method_correct = 0; method_total = 0

    pbar = tqdm(loader, desc='[VAL]', dynamic_ncols=True)
    for x, yb, ym in pbar:
        x = x.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True); ym = ym.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            if val_tta == "hflip":
                x_flip = torch.flip(x, dims=[-1])
                lb1, lm1 = model(x);      lb2, lm2 = model(x_flip)
                lb = (lb1 + lb2) * 0.5;   lm = (lm1 + lm2) * 0.5
            else:
                lb, lm = model(x)
            loss_bin = F.cross_entropy(lb, yb, label_smoothing=label_smoothing, reduction='mean')
        B = yb.size(0); sum_loss_bin += float(loss_bin.item()) * B; n_loss += B

        fake_mask = (yb == 0) & (ym >= 0)
        if fake_mask.any():
            pred_m = lm[fake_mask].argmax(1)
            method_correct += int((pred_m == ym[fake_mask]).sum().item())
            method_total += int(fake_mask.sum().item())

        probs_fake = torch.softmax(lb.float(), dim=1)[:, 0]
        comp = (probs_fake.unsqueeze(1) >= thrs.unsqueeze(0))
        pred_bin = torch.where(comp, torch.zeros_like(yb).unsqueeze(1), torch.ones_like(yb).unsqueeze(1))
        yb_expand = yb.unsqueeze(1).expand_as(pred_bin)

        is_fake = (yb_expand == 0); correct_fake += (pred_bin == 0).logical_and(is_fake).sum(dim=0).to(torch.float64)
        total_fake += int((yb == 0).sum().item())
        is_real = (yb_expand == 1); correct_real += (pred_bin == 1).logical_and(is_real).sum(dim=0).to(torch.float64)
        total_real += int((yb == 1).sum().item())

    if backup is not None: model.load_state_dict(backup, strict=False)

    eps = 1e-12
    rec_fake = (correct_fake / max(total_fake, 1)).clamp_(0, 1)
    rec_real = (correct_real / max(total_real, 1)).clamp_(0, 1)
    bacc_all = 0.5 * (rec_fake + rec_real)
    acc_all = (correct_fake + correct_real) / float(total_fake + total_real + eps)

    best_idx = int(torch.argmax(bacc_all).item())
    best_bacc = float(bacc_all[best_idx].item())
    best_acc  = float(acc_all[best_idx].item())
    best_thr  = float(thrs[best_idx].item())
    best_rec_fake = float(rec_fake[best_idx].item())
    best_rec_real = float(rec_real[best_idx].item())

    cons_idx = None; cons_thr = None; cons_vals = None
    mask = (bacc_all >= cons_bacc_min) & (rec_real >= cons_rec_real_min)
    if mask.any():
        idxs = torch.nonzero(mask).squeeze(1)
        score = rec_fake[idxs] + 1e-6 * acc_all[idxs]
        cand = idxs[torch.argmax(score)]
        cons_idx = int(cand.item()); cons_thr = float(thrs[cons_idx].item())
        cons_vals = (float(acc_all[cons_idx].item()), float(bacc_all[cons_idx].item()),
                     float(rec_fake[cons_idx].item()), float(rec_real[cons_idx].item()))
        print(f"[KQ VAL CONSTRAINT] acc={cons_vals[0]:.4f} | bacc={cons_vals[1]:.4f} | rec_fake={cons_vals[2]:.4f} | rec_real={cons_vals[3]:.4f} | thr@cons={cons_thr:.3f}")
    else:
        print("[KQ VAL CONSTRAINT] Không có ngưỡng thỏa điều kiện.")

    val_loss = (sum_loss_bin / max(n_loss, 1))
    method_acc = (method_correct / method_total) if method_total > 0 else 0.0
    elapsed = time.time() - t0
    print(f"[KQ VAL] loss={val_loss:.4f} | acc={best_acc:.4f} | bacc={best_bacc:.4f} | rec_fake={best_rec_fake:.4f} | rec_real={best_rec_real:.4f} | ngưỡng tốt={best_thr:.3f} | acc phương pháp (fake)={method_acc:.4f} | thời gian val={elapsed/60:.2f} phút")
    return {
        'loss': val_loss, 'acc': best_acc, 'bacc': best_bacc, 'best_thr': best_thr,
        'rec_fake': best_rec_fake, 'rec_real': best_rec_real, 'method_acc_fake': float(method_acc),
        'cons_thr': cons_thr, 'cons_acc': (cons_vals[0] if cons_vals else None),
        'cons_bacc': (cons_vals[1] if cons_vals else None),
        'cons_rec_fake': (cons_vals[2] if cons_vals else None),
        'cons_rec_real': (cons_vals[3] if cons_vals else None),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--micro_batch", type=int, default=32)
    p.add_argument("--val_batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--img_size", type=int, default=512)

    # CNN backbone
    p.add_argument("--model", type=str, default="convnextv2_base.fcmae_ft_in22k_in1k_384")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--face_crop", action="store_true")

    p.add_argument("--balance_by_method", action="store_true")
    p.add_argument("--method_boost", type=str, default="")
    p.add_argument("--bin_balance_sampler", action="store_true")

    p.add_argument("--method_loss_weight", type=float, default=0.2)
    p.add_argument("--bin_weight_real", type=float, default=1.0)
    p.add_argument("--focal_gamma", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--cutmix", type=float, default=0.0)
    p.add_argument("--mixup_prob", type=float, default=0.0)
    p.add_argument("--color_jitter", type=float, default=0.3)
    p.add_argument("--rand_erase_p", type=float, default=0.25)
    p.add_argument("--random_resized_crop", action="store_true")

    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--drop_path_rate", type=float, default=0.1)

    p.add_argument("--lr", type=float, default=8e-6)
    p.add_argument("--warmup_steps", type=int, default=800)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.05)

    p.add_argument("--thr_min", type=float, default=0.55)
    p.add_argument("--thr_max", type=float, default=0.90)
    p.add_argument("--thr_steps", type=int, default=101)

    p.add_argument("--resume", type=str, default="")
    p.add_argument("--freeze_epochs", type=int, default=0)
    p.add_argument("--method_warmup_epochs", type=int, default=0)
    p.add_argument("--early_stop_patience", type=int, default=3)

    p.add_argument("--val_use_ema", action="store_true")
    p.add_argument("--val_tta", type=str, default="none", choices=["none", "hflip"])
    p.add_argument("--grad_ckpt", action="store_true")

    p.add_argument("--cons_bacc_min", type=float, default=0.90)
    p.add_argument("--cons_rec_real_min", type=float, default=0.90)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_split", default="val", choices=["val", "test"])
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lấy danh sách phương pháp từ train/fake
    fake_root = os.path.join(args.data_root, "train", "fake")
    method_names = []
    if os.path.isdir(fake_root):
        for d in sorted(os.listdir(fake_root)):
            if os.path.isdir(os.path.join(fake_root, d)):
                method_names.append(d)
    print(f"📊 Phương pháp: {method_names if method_names else '[]'}")

    # Augment/Transform
    tfm_train_list = []
    if args.random_resized_crop:
        tfm_train_list.append(transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)))
    else:
        tfm_train_list.append(transforms.Resize((args.img_size, args.img_size)))
    tfm_train_list.extend([transforms.RandomHorizontalFlip(p=0.5)])
    if args.color_jitter > 0:
        cj = args.color_jitter
        tfm_train_list.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=min(0.1, cj*0.5)))
    tfm_train_list.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if args.rand_erase_p > 0:
        tfm_train_list.append(transforms.RandomErasing(p=args.rand_erase_p, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'))
    tfm_train = transforms.Compose(tfm_train_list)

    tfm_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Datasets
    ds_train = FacesDataset(args.data_root, 'train', method_names, tfm_train, face_crop=args.face_crop)
    ds_val   = FacesDataset(args.data_root, 'val',   method_names, tfm_eval, face_crop=args.face_crop)
    ds_test  = FacesDataset(args.data_root, 'test',  method_names, tfm_eval, face_crop=args.face_crop) \
               if os.path.isdir(os.path.join(args.data_root, 'test')) else None

    print(f"🔎 Tổng ảnh TRAIN: {len(ds_train)} | theo lớp: {ds_train.per_class}")
    print(f"🔎 Tổng ảnh VAL  : {len(ds_val)} | theo lớp: {ds_val.per_class}")
    if ds_test is not None:
        print(f"🔎 Tổng ảnh TEST : {len(ds_test)} | theo lớp: {ds_test.per_class}")
    else:
        print("ℹ️  Không tìm thấy split 'test' → sẽ bỏ qua đánh giá TEST sau mỗi epoch.")

    # Sampler
    sampler = None
    if args.balance_by_method:
        boost = parse_method_boost(args.method_boost)
        print(f"⚖️  Cân bằng theo method: BẬT | boost: {boost}")
        weights = []
        for _, yb, ym in ds_train.samples:
            if yb == 1:
                w = boost.get("real", 1.0)
            else:
                mname = method_names[int(ym)]
                w = boost.get(mname, 1.0)
            weights.append(w)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    if sampler is None and args.bin_balance_sampler:
        n_fake = sum(1 for _, yb, _ in ds_train.samples if yb == 0)
        n_real = sum(1 for _, yb, _ in ds_train.samples if yb == 1)
        ratio = max(1.0, n_fake / float(n_real)) if n_real > 0 else 1.0
        print(f"⚖️  Cân bằng nhị phân: BẬT | oversample real ×{ratio:.2f}")
        weights = [ (ratio if yb == 1 else 1.0) for _, yb, _ in ds_train.samples ]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    else:
        if not args.balance_by_method:
            print("⚖️  Cân bằng theo method: TẮT")
        if not args.bin_balance_sampler:
            print("⚖️  Cân bằng nhị phân: TẮT")

    accum_steps = max(1, args.batch_size // args.micro_batch)
    print(f"🧮 Batch hiệu dụng = {args.batch_size} (micro_batch={args.micro_batch}, accum_steps={accum_steps})")

    # Dataloaders
    dl_train = DataLoader(
        ds_train, batch_size=args.micro_batch, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.workers, pin_memory=True, drop_last=True, prefetch_factor=4, persistent_workers=(args.workers>0)
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.val_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=(args.workers>0)
    )
    dl_test = None
    if ds_test is not None and len(ds_test) > 0:
        dl_test = DataLoader(
            ds_test, batch_size=args.val_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=(args.workers>0)
        )

    # Model / Opt
    model = MultiHeadCNN(
        args.model, args.img_size, num_methods=len(method_names),
        drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate
    ).to(device)

    if args.grad_ckpt and hasattr(model.backbone, "set_grad_checkpointing"):
        try: model.backbone.set_grad_checkpointing(True)
        except Exception: pass

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    ema = EMA(model, decay=0.9999) if args.ema else None

    def set_backbone_requires_grad(req: bool):
        for _, p in model.backbone.named_parameters():
            p.requires_grad = req

    if args.freeze_epochs > 0:
        print(f"🧊 Đóng băng backbone trong {args.freeze_epochs} epoch đầu")
        set_backbone_requires_grad(False)

    # Resume
    start_epoch = 1
    best_metric = -1.0
    epochs_no_improve = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck.get('model', ck), strict=False)
        if 'optimizer' in ck:
            try: opt.load_state_dict(ck['optimizer']); print("↩️  Khôi phục optimizer.")
            except: pass
        if 'scaler' in ck:
            try: scaler.load_state_dict(ck['scaler']); print("↩️  Khôi phục scaler.")
            except: pass
        if 'ema' in ck and ema is not None and isinstance(ck['ema'], dict):
            ema.shadow = {k: v.to(device) for k, v in ck['ema'].items()}
        if 'epoch' in ck: start_epoch = int(ck['epoch']) + 1
        if 'best_metric' in ck: best_metric = float(ck['best_metric'])
        print(f"↩️  Nạp trọng số từ {args.resume}")

    print(f"📂 Lưu model vào: {args.out_dir}")
    print(f"🖼  Kích thước ảnh={args.img_size} | mô hình(CNN)={args.model} | epochs={args.epochs}")

    # Eval-only: chọn split theo --eval_split
    if args.eval_only:
        eval_loader = dl_val if args.eval_split == "val" else dl_test
        if eval_loader is None:
            raise RuntimeError(f"Không có dữ liệu cho split '{args.eval_split}'.")
        _ = evaluate(
            model, eval_loader, device, method_names,
            args.thr_min, args.thr_max,
            method_loss_weight=args.method_loss_weight,
            label_smoothing=args.label_smoothing,
            use_ema=args.val_use_ema and (ema is not None),
            ema_obj=ema, val_tta=args.val_tta, thr_steps=args.thr_steps,
            cons_bacc_min=args.cons_bacc_min, cons_rec_real_min=args.cons_rec_real_min
        )
        return

    # Loss helpers
    w_bin = torch.tensor([1.0, float(args.bin_weight_real)], dtype=torch.float, device=device)
    def ce_bin_fn(logits, targets): return focal_ce(logits, targets, weight=w_bin, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    def ce_met_fn(logits, targets): return F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)

    global_step = 0
    def set_lr(optim, base_lr, step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            lr = base_lr * float(step + 1) / float(args.warmup_steps)
        else:
            lr = base_lr
        for pg in optim.param_groups: pg['lr'] = lr
        return lr

    interrupted = {'flag': False}
    def handle_sigint(signum, frame): interrupted['flag'] = True
    signal.signal(signal.SIGINT, handle_sigint)

    # Train loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if args.freeze_epochs > 0 and epoch > args.freeze_epochs:
            set_backbone_requires_grad(True)

        pbar = tqdm(dl_train, total=len(dl_train), dynamic_ncols=True, desc=f"[Epoch {epoch}/{args.epochs}]")
        opt.zero_grad(set_to_none=True)
        effective_method_w = (0.0 if epoch <= args.method_warmup_epochs else args.method_loss_weight)

        tr_bin_logits, tr_bin_labels = [], []
        tr_met_logits, tr_met_labels = [], []
        t_train0 = time.time()

        for step, (x, yb, ym) in enumerate(pbar):
            if interrupted['flag']:
                inter_path = os.path.join(args.out_dir, f"detector_interrupt_epoch{epoch}_step{step}.pt")
                torch.save({'model': model.state_dict(),'optimizer': opt.state_dict(),'scaler': scaler.state_dict(),'ema': (ema.shadow if ema is not None else None),'epoch': epoch,'best_metric': best_metric}, inter_path)
                print(f"\n⛔ Dừng thủ công. Đã lưu checkpoint → {inter_path}"); return

            x  = x.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True); ym = ym.to(device, non_blocking=True)

            x_in = x; soft_yb = None; applied_mix = False
            if args.mixup_prob > 0.0 and (args.mixup > 0.0 or args.cutmix > 0.0):
                x_in, soft_yb, applied_mix = apply_mix(x, yb, args.mixup, args.cutmix, args.mixup_prob)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                lb, lm = model(x_in)
                loss_b = soft_ce(lb.float(), soft_yb) if applied_mix else ce_bin_fn(lb.float(), yb)
                fake_mask = (yb == 0) & (ym >= 0)
                loss_m_raw = ce_met_fn(lm[fake_mask].float(), ym[fake_mask]) if (fake_mask.any() and effective_method_w > 0) else torch.tensor(0.0, device=device)
                loss = loss_b + effective_method_w * loss_m_raw

            with torch.no_grad():
                tr_bin_logits.append(lb.detach().float().cpu()); tr_bin_labels.append(yb.detach().cpu())
                tr_met_logits.append(lm.detach().float().cpu());  tr_met_labels.append(ym.detach().cpu())

            loss = loss / max(1, args.batch_size // args.micro_batch)
            scaler.scale(loss).backward()

            do_step = ((step + 1) % max(1, args.batch_size // args.micro_batch)) == 0
            if do_step:
                if args.clip_grad and args.clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                global_step += 1; set_lr(opt, args.lr, global_step)
                if ema is not None: ema.update(model)

            if (step % 2) == 0:
                with torch.no_grad():
                    probs_fake = torch.softmax(lb.float(), dim=1)[:, 0]
                    thrs = torch.linspace(args.thr_min, args.thr_max, steps=41, device=probs_fake.device)
                    best_acc_b, best_bacc_b, best_thr_b = 0.0, 0.0, float((args.thr_min + args.thr_max) / 2)
                    m_fake = (yb == 0); m_real = (yb == 1)
                    for t in thrs:
                        pred_bin = torch.where(probs_fake >= t, torch.zeros_like(yb), torch.ones_like(yb))
                        acc_b = (pred_bin == yb).float().mean().item()
                        rec_fake = (pred_bin[m_fake] == 0).float().mean().item() if m_fake.any() else 0.0
                        rec_real = (pred_bin[m_real] == 1).float().mean().item() if m_real.any() else 0.0
                        bacc_b = 0.5 * (rec_fake + rec_real)
                        if bacc_b > best_bacc_b:
                            best_bacc_b, best_acc_b, best_thr_b = bacc_b, acc_b, float(t.item())
                    fake_mask_b = (yb == 0) & (ym >= 0)
                    method_acc_b = (lm[fake_mask_b].argmax(1) == ym[fake_mask_b]).float().mean().item() if fake_mask_b.any() else 0.0
                pbar.set_postfix_str(f"loss={loss.item():.4f}, acc={best_acc_b:.4f}, bacc={best_bacc_b:.4f}, thr={best_thr_b:.3f}, m_acc={method_acc_b:.4f}, lr={opt.param_groups[0]['lr']:.2e}")

        # ===== KQ TRAIN (tổng hợp) =====
        with torch.no_grad():
            lb_tr = torch.cat(tr_bin_logits, 0); yb_tr = torch.cat(tr_bin_labels, 0)
            lm_tr = torch.cat(tr_met_logits, 0); ym_tr = torch.cat(tr_met_labels, 0)
            loss_bin_tr = F.cross_entropy(lb_tr, yb_tr, label_smoothing=args.label_smoothing)
            fake_mask_tr = (yb_tr == 0) & (ym_tr >= 0)
            loss_met_tr_raw = F.cross_entropy(lm_tr[fake_mask_tr], ym_tr[fake_mask_tr], label_smoothing=args.label_smoothing) if (fake_mask_tr.any() and args.method_loss_weight > 0) else torch.tensor(0.0)
            train_loss = (loss_bin_tr + args.method_loss_weight * loss_met_tr_raw).item()

            probs_fake_tr = torch.softmax(lb_tr, dim=1)[:, 0]
            thrs_tr = torch.linspace(args.thr_min, args.thr_max, steps=101)
            best_acc_tr, best_bacc_tr, best_thr_tr = 0.0, 0.0, float((args.thr_min + args.thr_max) / 2)
            best_rec_fake_tr, best_rec_real_tr = 0.0, 0.0
            for t in thrs_tr:
                pred_bin_tr = torch.where(probs_fake_tr >= t, torch.zeros_like(yb_tr), torch.ones_like(yb_tr))
                acc_tr = (pred_bin_tr == yb_tr).float().mean().item()
                m_fake_tr = (yb_tr == 0); m_real_tr = (yb_tr == 1)
                rec_fake_tr = (pred_bin_tr[m_fake_tr] == 0).float().mean().item() if m_fake_tr.any() else 0.0
                rec_real_tr = (pred_bin_tr[m_real_tr] == 1).float().mean().item() if m_real_tr.any() else 0.0
                bacc_tr = 0.5 * (rec_fake_tr + rec_real_tr)
                if bacc_tr > best_bacc_tr:
                    best_bacc_tr, best_acc_tr, best_thr_tr = bacc_tr, acc_tr, float(t.item())
                    best_rec_fake_tr, best_rec_real_tr = rec_fake_tr, rec_real_tr
            method_acc_tr = (lm_tr[fake_mask_tr].argmax(1) == ym_tr[fake_mask_tr]).float().mean().item() if fake_mask_tr.any() else 0.0
            elapsed_train = time.time() - t_train0
            print(f"[KQ TRAIN] loss={train_loss:.4f} | loss_bin={loss_bin_tr.item():.4f} | loss_met={loss_met_tr_raw.item():.4f} (x{args.method_loss_weight:.2f}) | acc={best_acc_tr:.4f} | bacc={best_bacc_tr:.4f} | rec_fake={best_rec_fake_tr:.4f} | rec_real={best_rec_real_tr:.4f} | ngưỡng tốt={best_thr_tr:.3f} | acc phương pháp (fake)={method_acc_tr:.4f} | thời gian train={elapsed_train/60:.2f} phút")

        # ===== VAL mỗi epoch =====
        val_res = evaluate(
            model, dl_val, device, method_names,
            args.thr_min, args.thr_max,
            method_loss_weight=args.method_loss_weight,
            label_smoothing=args.label_smoothing,
            use_ema=args.val_use_ema and (ema is not None),
            ema_obj=ema, val_tta=args.val_tta, thr_steps=args.thr_steps,
            cons_bacc_min=args.cons_bacc_min, cons_rec_real_min=args.cons_rec_real_min
        )

        # ===== TEST mỗi epoch (nếu có) =====
        test_res = None
        if dl_test is not None:
            print("\n[ĐÁNH GIÁ TEST] ——")
            test_res = evaluate(
                model, dl_test, device, method_names,
                args.thr_min, args.thr_max,
                method_loss_weight=args.method_loss_weight,
                label_smoothing=args.label_smoothing,
                use_ema=args.val_use_ema and (ema is not None),
                ema_obj=ema, val_tta=args.val_tta, thr_steps=args.thr_steps,
                cons_bacc_min=args.cons_bacc_min, cons_rec_real_min=args.cons_rec_real_min
            )
            print(f"[KQ TEST] acc={test_res['acc']:.4f} | bacc={test_res['bacc']:.4f} | rec_fake={test_res['rec_fake']:.4f} | rec_real={test_res['rec_real']:.4f} | ngưỡng tốt={test_res['best_thr']:.3f} | acc phương pháp (fake)={test_res['method_acc_fake']:.4f}")

        # ===== Lưu best (theo VAL) + checkpoint epoch có kèm summary =====
        improved = val_res['bacc'] > best_metric
        if improved:
            best_metric = val_res['bacc']; epochs_no_improve = 0
            best_path = os.path.join(args.out_dir, "detector_best.pt")
            torch.save({'model': model.state_dict(),'optimizer': opt.state_dict(),'scaler': scaler.state_dict(),'ema': (ema.shadow if ema is not None else None),'epoch': epoch,'best_metric': best_metric,'best_thr': val_res.get('best_thr', None)}, best_path)
            print(f"💾 Đã lưu mô hình BEST → {best_path} (best_thr={val_res.get('best_thr', None)})")
        else:
            epochs_no_improve += 1

        ep_path = os.path.join(args.out_dir, f"detector_epoch{epoch}.pt")
        torch.save({'model': model.state_dict(),'optimizer': opt.state_dict(),'scaler': scaler.state_dict(),'ema': (ema.shadow if ema is not None else None),'epoch': epoch,'best_metric': best_metric,'val_summary': val_res,'test_summary': test_res}, ep_path)

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"🛑 Early stopping vì không cải thiện BACC trong {args.early_stop_patience} epoch."); break

    print("✅ Train xong.")

if __name__ == "__main__":
    main()
