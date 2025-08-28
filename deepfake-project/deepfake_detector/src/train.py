# deepfake_detector/src/train.py
# Usage:
#   py -m deepfake_detector.src.train --epochs 24 --batch_size 64 --model vit_base_patch16_224 --ema
#   (nếu thiếu VRAM: giảm batch_size; KHÔNG dùng --compile trên Windows)

import argparse, os, time, math, signal, sys, random, io
from datetime import timedelta
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image

import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2

# ---------------- Utils ----------------
def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class ProgressMeter:
    def __init__(self, total):
        self.total = total
        self.start = time.time()
        self.done = 0
    def step(self, n=1): self.done += n
    def text(self):
        elapsed = time.time() - self.start
        rate = self.done / max(elapsed, 1e-9)
        remaining = (self.total - self.done) / max(rate, 1e-9)
        pct = 100.0 * self.done / max(self.total, 1)
        return (f"{pct:6.2f}% | {self.done}/{self.total} files | "
                f"elapsed {timedelta(seconds=int(elapsed))} | "
                f"remaining {timedelta(seconds=int(remaining))}")

def cosine_warmup(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

# JPEG compression augment để mô phỏng nén video
class RandomJPEG:
    def __init__(self, qmin=35, qmax=90, p=0.5):
        self.qmin, self.qmax, self.p = qmin, qmax, p
    def __call__(self, img: Image.Image):
        if random.random() > self.p: return img
        q = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

# ---------------- Trainer ----------------
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_dir = Path(args.out_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best = {"acc": 0.0, "threshold": 0.5}
        self.interrupted = False
        self.frozen = False

        def _handle_sigint(sig, frame):
            print("\n[CTRL+C] Sắp lưu checkpoint an toàn…")
            self.interrupted = True
        signal.signal(signal.SIGINT, _handle_sigint)

    # ---------- Data ----------
    def build_data(self):
        img_size = 224
        self.mean = [0.5, 0.5, 0.5]
        self.std  = [0.5, 0.5, 0.5]

        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.80, 1.00)),
            transforms.RandomHorizontalFlip(),
            RandomJPEG(35, 90, p=0.6),
            transforms.ColorJitter(0.25,0.25,0.25,0.12),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.35),
            transforms.RandomAffine(degrees=6, translate=(0.03,0.03)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            transforms.RandomErasing(p=0.25, scale=(0.02,0.10), ratio=(0.3,3.3), value=0),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        root = self.args.data_root
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   transform=val_tf)

        self.class_names = train_ds.classes  # ['fake','real']
        self.num_train_files = len(train_ds)
        self.num_val_files = len(val_ds)
        print(f"🔎 Tổng số file train: {self.num_train_files} | val: {self.num_val_files}")
        print(f"📂 Classes = {self.class_names}")

        # Sampler cân lớp
        targets = [s[1] for s in train_ds.samples]
        counts = torch.bincount(torch.tensor(targets), minlength=2).float()
        weights = 1.0 / torch.clamp(counts, min=1)
        sample_weights = torch.tensor([weights[t] for t in targets], dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        self.train_loader = DataLoader(train_ds, batch_size=self.args.batch_size,
                                       sampler=sampler, num_workers=self.args.workers,
                                       pin_memory=True, drop_last=True)
        self.val_loader   = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False,
                                       num_workers=self.args.workers, pin_memory=True)

        # Mixup/Cutmix nhẹ để không làm lệch calibration
        self.mixup_fn = None
        if self.args.mixup > 0 or self.args.cutmix > 0:
            self.mixup_fn = Mixup(mixup_alpha=self.args.mixup,
                                  cutmix_alpha=self.args.cutmix,
                                  label_smoothing=self.args.label_smoothing,
                                  num_classes=2)

    # ---------- Model ----------
    def build_model(self):
        model_name = self.args.model
        print(f"🧠 Model: {model_name}")
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=2)
        except RuntimeError as e:
            print(f"[WARN] {e}\n→ Fallback sang vit_base_patch16_224")
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        # KHÔNG dùng torch.compile trên Windows (Triton); chỉ khi bạn truyền --compile và đã cài Triton
        if self.args.compile:
            try:
                model = torch.compile(model, mode="max-autotune")
                print("🔧 torch.compile enabled")
            except Exception as e:
                print(f"[WARN] torch.compile disabled: {e}")

        # Freeze backbone vài epoch đầu để ổn định head
        if hasattr(model, "head"):
            for n,p in model.named_parameters():
                if not n.startswith("head"): p.requires_grad_(False)
            self.frozen = True
            print("🧊 Freeze backbone trong", self.args.freeze_epochs, "epoch đầu")

        if hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing(self.args.grad_ckpt)

        model.to(self.device)
        self.ema = ModelEmaV2(model, decay=0.9998) if self.args.ema else None

        # Loss
        if self.mixup_fn is not None:
            self.criterion = SoftTargetCrossEntropy()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

        # Optim
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr, weight_decay=2e-4, betas=(0.9, 0.999)
        )
        self.total_steps = math.ceil(self.num_train_files / self.args.batch_size) * self.args.epochs
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        self.model = model

    def _unfreeze_all(self):
        if self.frozen:
            for p in self.model.parameters(): p.requires_grad_(True)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.lr, weight_decay=2e-4, betas=(0.9, 0.999)
            )
            self.frozen = False
            print("🔥 Unfreeze toàn bộ backbone & reset optimizer")

    def save_ckpt(self, tag, extra_meta=None):
        meta = {
            "classes": self.class_names,
            "model_name": self.args.model,
            "norm_mean": self.mean,
            "norm_std": self.std,
            "threshold": self.best.get("threshold", 0.5),
        }
        if extra_meta: meta.update(extra_meta)
        path = self.ckpt_dir / f"detector_{tag}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.module.state_dict() if self.ema else None,
            "meta": meta
        }, path)
        print(f"💾 Saved checkpoint: {path}")

    # ---------- Eval ----------
    @torch.no_grad()
    def evaluate(self, use_ema=True, tta=True) -> Tuple[float,float,float,float]:
        model = self.ema.module if (self.ema and use_ema) else self.model
        model.eval()
        loss_sum, correct, count = 0.0, 0, 0
        pm = ProgressMeter(total=self.num_val_files)
        probs_fake, labels_all = [], []

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast('cuda', enabled=(self.device.type=='cuda')):
                logits = model(x)
                if tta:
                    logits = (logits + model(torch.flip(x, dims=[3]))) / 2
                loss = nn.CrossEntropyLoss()(logits, y)

            prob_fake = torch.softmax(logits, dim=1)[:, 0]  # 0='fake'
            probs_fake.append(prob_fake.detach().cpu())
            labels_all.append(y.detach().cpu())
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            count += x.size(0)
            pm.step(x.size(0))
            print(f"\r[VAL] {pm.text()}", end="")
        print()

        probs_fake = torch.cat(probs_fake).numpy()
        labels_all = torch.cat(labels_all).numpy()

        # Quét ngưỡng (nhỏ hơn, hợp calibration)
        best_thr, best_bacc = 0.5, 0.0
        for thr in [i/200 for i in range(80, 121)]:  # 0.40 -> 0.60
            pred_fake = (probs_fake >= thr).astype(int)  # 1 = fake
            y_fake = (labels_all == 0).astype(int)
            tp = ((pred_fake==1) & (y_fake==1)).sum()
            tn = ((pred_fake==0) & (y_fake==0)).sum()
            fp = ((pred_fake==1) & (y_fake==0)).sum()
            fn = ((pred_fake==0) & (y_fake==1)).sum()
            tpr = tp / max(tp+fn, 1)
            tnr = tn / max(tn+fp, 1)
            bacc = 0.5*(tpr+tnr)
            if bacc > best_bacc:
                best_bacc = bacc; best_thr = thr

        return loss_sum / count, correct / count, best_thr, best_bacc

    # ---------- Train ----------
    def train(self):
        gpm_total = math.ceil(self.num_train_files / self.args.batch_size) * self.args.batch_size * self.args.epochs
        gpm = ProgressMeter(total=gpm_total)
        print(f"🚀 Bắt đầu train ({self.args.epochs} epochs) | tổng file cần xử lý: {gpm_total}")

        step_global, no_improve, best_acc = 0, 0, 0.0

        for epoch in range(1, self.args.epochs + 1):
            # Unfreeze sau freeze_epochs
            if epoch == self.args.freeze_epochs + 1 and self.frozen:
                self._unfreeze_all()

            self.model.train()
            epoch_loss, seen = 0.0, 0

            for x, y in self.train_loader:
                if self.interrupted:
                    self.save_ckpt(f"epoch{epoch}_interrupt")
                    print("⏸ Đã lưu checkpoint do Ctrl+C.")
                    sys.exit(0)

                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                if self.mixup_fn is not None:
                    x, y = self.mixup_fn(x, y)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(self.device.type=='cuda')):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.ema:
                    self.ema.update(self.model)

                step_global += 1
                lr_scale = cosine_warmup(step_global, self.total_steps, self.args.warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.args.lr * lr_scale
                lr_now = self.optimizer.param_groups[0]['lr']

                bs = x.size(0)
                seen += bs
                gpm.step(bs)
                epoch_loss += loss.item() * bs
                print(f"\r[Epoch {epoch}/{self.args.epochs}] {gpm.text()} | batch_loss {loss.item():.4f} | lr {lr_now:.2e}", end="")

            print()
            val_loss, val_acc, thr, bacc = self.evaluate(use_ema=True, tta=True)
            train_loss = epoch_loss / max(seen, 1)
            print(f"Epoch {epoch}/{self.args.epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | best_thr {thr:.3f} | val_bacc {bacc:.4f}")

            # checkpoint mỗi epoch
            self.best["threshold"] = thr
            self.save_ckpt(f"epoch{epoch}", extra_meta={"threshold": thr, "val_acc": val_acc, "val_bacc": bacc})

            if val_acc > best_acc:
                best_acc = val_acc; no_improve = 0
                self.best.update({"acc": best_acc, "threshold": thr})
                self.save_ckpt("best", extra_meta={"threshold": thr, "val_acc": val_acc, "val_bacc": bacc})
            else:
                no_improve += 1

            if self.args.early_stop_patience > 0 and no_improve >= self.args.early_stop_patience:
                print(f"⛳ Early stopping sau {epoch} epochs (không cải thiện {self.args.early_stop_patience} epoch).")
                break

        print(f"✅ Train xong. Best val_acc = {self.best['acc']:.4f} | threshold={self.best['threshold']:.3f}")

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/processed/faces")
    p.add_argument("--out_dir", type=str, default="deepfake_detector")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=64)  # 16GB: 64 cho ViT-Base ok
    p.add_argument("--lr", type=float, default=1e-4)      # LR nhỏ hơn để ổn định
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mixup", type=float, default=0.15)
    p.add_argument("--cutmix", type=float, default=0.15)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--early_stop_patience", type=int, default=6)
    p.add_argument("--grad_ckpt", action="store_true", default=True)
    p.add_argument("--model", type=str, default="vit_base_patch16_224")  # quay lại BASE cho ổn định
    p.add_argument("--freeze_epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true", default=False)
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    t = Trainer(args)
    t.build_data()
    t.build_model()
    t.train()

if __name__ == "__main__":
    main()