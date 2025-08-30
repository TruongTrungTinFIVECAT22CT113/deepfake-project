# deepfake_detector/src/train.py
import argparse, os, time, math, signal, sys, random, io, re
from datetime import timedelta
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2

# ----------------- Mapping method (khớp preprocess) -----------------
ALL_METHODS_RAW = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
ALIASES = {"DeepFakeDetection": "Deepfakes"}
METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "Other"]

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ----------------- Face detector cho FaceCrop -----------------
try:
    import mediapipe as mp
    _MP_OK = True
    _mp_fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
except Exception:
    _MP_OK = False
    _mp_fd = None
import cv2
_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class FaceCenterCrop:
    """Cắt quanh mặt lớn nhất rồi resize; nếu fail thì trả ảnh gốc."""
    def __init__(self, enlarge=1.4): self.enlarge = float(enlarge)
    def __call__(self, img: Image.Image):
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        boxes=[]
        if _MP_OK and _mp_fd is not None:
            res = _mp_fd.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if res and res.detections:
                for d in res.detections:
                    bb = d.location_data.relative_bounding_box
                    x1 = max(0, int(bb.xmin*w)); y1 = max(0, int(bb.ymin*h))
                    x2 = min(w-1, int((bb.xmin+bb.width)*w)); y2 = min(h-1, int((bb.ymin+bb.height)*h))
                    if x2>x1 and y2>y1: boxes.append((x1,y1,x2,y2))
        if not boxes:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for (x,y,fw,fh) in _HAAR.detectMultiScale(g, 1.2, 5, minSize=(60,60)):
                boxes.append((x,y,x+fw,y+fh))
        if not boxes: return img
        x1,y1,x2,y2 = max(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
        cx=(x1+x2)/2; cy=(y1+y2)/2; s=max(x2-x1, y2-y1)*self.enlarge
        nx1=int(max(0, cx-s/2)); ny1=int(max(0, cy-s/2))
        nx2=int(min(w, cx+s/2));  ny2=int(min(h, cy+s/2))
        crop=bgr[ny1:ny2, nx1:nx2]
        if crop.size==0: return img
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

# ----------------- Tiện ích -----------------
class RandomJPEG:
    def __init__(self, qmin=35, qmax=90, p=0.6): self.qmin, self.qmax, self.p = qmin, qmax, p
    def __call__(self, img: Image.Image):
        if random.random() > self.p: return img
        q = random.randint(self.qmin, self.qmax); buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q); buf.seek(0)
        return Image.open(buf).convert("RGB")

def infer_method_from_path(path: str) -> str:
    p = path.replace("\\","/").lower()
    if "/real/" in p: return "real"
    for m in ALL_METHODS_RAW:
        name = ALIASES.get(m, m).lower()
        if f"/{name}/" in p or f"/{m.lower()}/" in p: return ALIASES.get(m, m)
    for m in METHODS:
        if m.lower() in p: return m
    return "Other"

class ProgressMeter:
    def __init__(self, total): self.total=int(total); self.start=time.time(); self.done=0
    def step(self, n=1): self.done+=n
    def text(self):
        elapsed=time.time()-self.start; rate=self.done/max(elapsed,1e-9)
        remaining=(self.total-self.done)/max(rate,1e-9) if self.total>self.done else 0
        pct=100.0*self.done/max(self.total,1)
        return (f"{pct:6.2f}% | {self.done}/{self.total} files | "
                f"elapsed {timedelta(seconds=int(elapsed))} | "
                f"remaining {timedelta(seconds=int(remaining))}")

def cosine_warmup(step, total_steps, warmup_steps):
    if step < warmup_steps: return step / max(1, warmup_steps)
    progress=(step-warmup_steps)/max(1, total_steps-warmup_steps)
    return 0.5*(1.0+math.cos(math.pi*progress))

# ----------------- Dataset (trả thêm method_idx) -----------------
class ImageFolderWithMethod(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.method_to_idx = {m:i for i,m in enumerate(METHODS)}  # order fixed
        self.sample_methods = []
        for path, targ in self.samples:
            if targ == 1:  # real
                self.sample_methods.append(-1)
            else:
                m = infer_method_from_path(path)
                if m == "real": m = "Other"
                self.sample_methods.append(self.method_to_idx.get(m, self.method_to_idx["Other"]))
    def __getitem__(self, index):
        img, label = super().__getitem__(index)  # label: 0=fake, 1=real
        m = self.sample_methods[index]
        return img, label, m

# ----------------- Mô hình Multi-head -----------------
class MultiHeadViT(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_224", img_size=224, pretrained=True, num_methods=len(METHODS)):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        feat_dim = self.backbone.num_features
        self.head_cls = nn.Linear(feat_dim, 2)            # fake/real
        self.head_mth = nn.Linear(feat_dim, num_methods)  # method
        self.dropout = nn.Dropout(0.1)
    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable)
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head_cls(feat), self.head_mth(feat)

# ----------------- Trainer -----------------
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_dir = Path(args.out_dir) / "checkpoints"; self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best = {"acc": 0.0, "threshold": 0.5}
        self.interrupted = False; self.frozen = False; self.start_epoch = 0

        def _sigint(sig, frame):
            print("\n[CTRL+C] Sắp lưu checkpoint an toàn…")
            self.interrupted = True
        signal.signal(signal.SIGINT, _sigint)

    # ---------- Data ----------
    def build_data(self):
        img_size = self.args.img_size
        self.mean, self.std = [0.5]*3, [0.5]*3

        train_ops=[]
        if self.args.face_crop: train_ops.append(FaceCenterCrop(enlarge=1.4))
        train_ops += [
            transforms.RandomResizedCrop(img_size, scale=(0.80,1.00)),
            transforms.RandomHorizontalFlip(),
            RandomJPEG(35,90,p=0.6),
            transforms.ColorJitter(0.25,0.25,0.25,0.12),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.35),
            transforms.RandomAffine(degrees=6, translate=(0.03,0.03)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            transforms.RandomErasing(p=0.25, scale=(0.02,0.10), ratio=(0.3,3.3), value=0),
        ]
        val_ops=[]
        if self.args.face_crop: val_ops.append(FaceCenterCrop(enlarge=1.3))
        val_ops += [
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ]
        train_tf, val_tf = transforms.Compose(train_ops), transforms.Compose(val_ops)

        root = self.args.data_root
        self.train_ds = ImageFolderWithMethod(os.path.join(root, "train"), transform=train_tf)
        self.val_ds   = ImageFolderWithMethod(os.path.join(root, "val"),   transform=val_tf)

        self.class_names = self.train_ds.classes  # ['fake','real']
        self.method_names = METHODS
        self.num_train_files = len(self.train_ds); self.num_val_files = len(self.val_ds)
        print(f"🔎 Tổng số file train: {self.num_train_files} | val: {self.num_val_files}")
        print(f"📂 Classes = {self.class_names}")

        # ---- Sampler: cân bằng lớp + (tuỳ chọn) theo method + method_boost ----
        targets = [s[1] for s in self.train_ds.samples]  # 0=fake, 1=real
        counts = torch.bincount(torch.tensor(targets), minlength=2).float()
        class_weights = 1.0 / torch.clamp(counts, min=1)

        method_weights = None
        boost = {}
        if self.args.method_boost:
            for kv in self.args.method_boost.split(","):
                if "=" in kv:
                    k,v = kv.split("="); boost[k.strip()] = float(v)

        if self.args.balance_by_method:
            from collections import Counter
            methods = [infer_method_from_path(p) if t==0 else "real" for (p,t) in self.train_ds.samples]
            cnt = Counter(methods)
            method_weights = {m: 1.0 / max(c,1) for m,c in cnt.items()}
            for k,v in boost.items():
                if k in method_weights: method_weights[k] *= float(v)
            print("⚖️  Balance by method:", cnt, "| boost:", boost or "{}")

        sw=[]
        for (path, tgt), m_idx in zip(self.train_ds.samples, self.train_ds.sample_methods):
            w = float(class_weights[tgt])
            if method_weights is not None:
                m_name = "real" if tgt==1 else self.method_names[m_idx]
                w *= method_weights.get(m_name, 1.0)
            sw.append(w)
        sample_weights = torch.tensor(sw, dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        common = dict(num_workers=self.args.workers, pin_memory=True,
                      persistent_workers=(self.args.workers>0), prefetch_factor=4)
        self.train_loader = DataLoader(self.train_ds, batch_size=self.args.batch_size,
                                       sampler=sampler, drop_last=True, **common)
        val_bs = min(self.args.batch_size, max(1,self.args.micro_batch))
        self.val_loader   = DataLoader(self.val_ds, batch_size=val_bs, shuffle=False, **common)

        # Mixup/Cutmix
        self.mixup_fn = None
        if self.args.mixup > 0 or self.args.cutmix > 0:
            self.mixup_fn = Mixup(mixup_alpha=self.args.mixup, cutmix_alpha=self.args.cutmix,
                                  label_smoothing=self.args.label_smoothing, num_classes=2,
                                  prob=self.args.mixup_prob)

    # ---------- Model ----------
    def build_model(self):
        model_name = self.args.model
        variant_map = {
            ("vit_base_patch16_224", 384): "vit_base_patch16_384",
            ("vit_large_patch16_224", 384): "vit_large_patch16_384",
        }
        eff_name = variant_map.get((model_name, self.args.img_size), model_name)

        print(f"🧠 Model: {model_name} (effective: {eff_name}) | img_size={self.args.img_size}")
        model = MultiHeadViT(eff_name, img_size=self.args.img_size, pretrained=True, num_methods=len(self.method_names))

        # Freeze backbone N epoch đầu
        for n,p in model.backbone.named_parameters(): p.requires_grad_(False)
        self.frozen = True
        print("🧊 Freeze backbone trong", self.args.freeze_epochs, "epoch đầu")

        model.set_grad_checkpointing(self.args.grad_ckpt)

        model.to(self.device)
        self.ema = ModelEmaV2(model, decay=0.9998) if self.args.ema else None

        # LOSS
        self.criterion_bin_soft = SoftTargetCrossEntropy()
        self.criterion_bin_hard = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        self.criterion_mth = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr, weight_decay=2e-4, betas=(0.9, 0.999)
        )
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * self.args.epochs
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type=='cuda'))
        self.model = model
        self.model_name_effective = eff_name

    # ---------- Resume ----------
    def maybe_resume(self):
        # auto-pick latest if requested and --resume is empty
        if not self.args.resume and getattr(self.args, "auto_resume", False):
            ckpts = list(self.ckpt_dir.glob("detector_epoch*.pt"))
            if ckpts:
                ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                self.args.resume = str(ckpts[0])
                print(f"🔁 Auto-resume từ {self.args.resume}")

        if not self.args.resume:
            return

        ckpt = torch.load(self.args.resume, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)

        # ---- OPTIMIZER / SCALER ----
        if not self.args.reset_opt:
            try:
                if ckpt.get("optimizer"):
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                if ckpt.get("scaler"):
                    self.scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"⚠️  Không thể load optimizer từ checkpoint ({e}). Sẽ bỏ qua và dùng optimizer mới.")
        else:
            print("🔄 Bỏ qua optimizer theo yêu cầu (--reset_opt).")

        if self.ema and ckpt.get("ema") is not None:
            self.ema.module.load_state_dict(ckpt["ema"])

        # ---- epoch ----
        meta = ckpt.get("meta", {})
        self.best["threshold"] = float(meta.get("threshold", self.best["threshold"]))

        start_ep = int(meta.get("epoch", 0))
        if start_ep == 0:
            import re, os
            m = re.search(r'epoch(\d+)', os.path.basename(self.args.resume))
            if m: start_ep = int(m.group(1))
        self.start_epoch = start_ep

        print(f"🔁 Resume từ {self.args.resume} (start_epoch={self.start_epoch})")

        # 🔥 UNFREEZE NGAY KHI RESUME nếu đã qua giai đoạn freeze
        if self.frozen and self.start_epoch >= self.args.freeze_epochs:
            print(f"🔥 Unfreeze ngay khi resume (start_epoch={self.start_epoch} ≥ freeze_epochs={self.args.freeze_epochs})")
            self._unfreeze_all()  # sẽ reset optimizer để thêm params backbone

    def _unfreeze_all(self):
        if self.frozen:
            for p in self.model.backbone.parameters(): 
                p.requires_grad_(True)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.lr, weight_decay=2e-4, betas=(0.9,0.999)
            )
            self.frozen = False
            print("🔥 Unfreeze toàn bộ backbone & reset optimizer")

    def save_ckpt(self, tag, extra_meta=None, epoch=None):
        meta = {
            "classes": self.class_names,
            "method_names": self.method_names,
            "model_name": self.model_name_effective,
            "norm_mean": self.mean, "norm_std": self.std,
            "threshold": self.best.get("threshold", 0.5),
            "img_size": self.args.img_size,
        }
        if epoch is not None: meta["epoch"] = epoch
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

    @torch.no_grad()
    def evaluate(self, use_ema=True, tta=True) -> Tuple[float,float,float,float,float]:
        model = self.ema.module if (self.ema and use_ema) else self.model
        model.eval()
        loss_sum, correct, count = 0.0, 0, 0
        pm = ProgressMeter(total=self.num_val_files)
        probs_fake, labels_bin, mth_true, mth_pred = [], [], [], []

        for x, y, m in self.val_loader:
            x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)
            with torch.amp.autocast('cuda', enabled=(self.device.type=='cuda')):
                logits_bin, logits_m = model(x)
                if tta:
                    logits_bin2, logits_m2 = model(torch.flip(x, dims=[3]))
                    logits_bin = (logits_bin + logits_bin2) / 2
                    logits_m   = (logits_m + logits_m2) / 2
                loss = nn.CrossEntropyLoss()(logits_bin, y)

            prob_fake = torch.softmax(logits_bin, dim=1)[:,0]
            probs_fake.append(prob_fake.detach().cpu()); labels_bin.append(y.detach().cpu())
            loss_sum += loss.item() * x.size(0)
            pred = logits_bin.argmax(1)
            correct += (pred == y).sum().item(); count += x.size(0)

            mask = (y == 0) & (m >= 0)
            if mask.any():
                mth_true.append(m[mask].detach().cpu())
                mth_pred.append(logits_m[mask].argmax(1).detach().cpu())

            pm.step(x.size(0)); print(f"\r[VAL] {pm.text()}", end="")
        print()

        probs_fake = torch.cat(probs_fake).numpy()
        labels_bin = torch.cat(labels_bin).numpy()

        best_thr, best_bacc = 0.5, 0.0
        for thr in [i/200 for i in range(80, 121)]:  # 0.40→0.60
            pred_fake = (probs_fake >= thr).astype(int)
            y_fake = (labels_bin == 0).astype(int)
            tp = ((pred_fake==1) & (y_fake==1)).sum()
            tn = ((pred_fake==0) & (y_fake==0)).sum()
            fp = ((pred_fake==1) & (y_fake==0)).sum()
            fn = ((pred_fake==0) & (y_fake==1)).sum()
            tpr = tp / max(tp+fn, 1); tnr = tn / max(tn+fp, 1)
            bacc = 0.5*(tpr+tnr)
            if bacc > best_bacc: best_bacc, best_thr = bacc, thr

        meth_acc = 0.0
        if mth_true:
            mth_true = torch.cat(mth_true).numpy()
            mth_pred = torch.cat(mth_pred).numpy()
            meth_acc = float((mth_true == mth_pred).mean())

        return loss_sum / count, correct / count, best_thr, best_bacc, meth_acc

    # ---------- Train ----------
    def train(self):
        micro = self.args.micro_batch if self.args.micro_batch > 0 else self.args.batch_size
        if (self.args.mixup > 0 or self.args.cutmix > 0) and (micro % 2 == 1):
            micro += 1; print(f"🔧 micro_batch adjusted to even {micro} for mixup/cutmix")
        accum_steps = max(1, math.ceil(self.args.batch_size / micro))
        print(f"🧮 Effective batch = {self.args.batch_size} (micro_batch={micro}, accum_steps={accum_steps})")

        remaining_epochs = max(0, self.args.epochs - self.start_epoch)
        gpm_total = self.steps_per_epoch * self.args.batch_size * remaining_epochs
        gpm = ProgressMeter(total=gpm_total)
        print(f"🚀 Bắt đầu train (epochs {self.start_epoch+1}→{self.args.epochs}) | tổng file cần xử lý: {gpm_total}")

        step_global = self.steps_per_epoch * self.start_epoch
        best_acc = 0.0

        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
            if epoch == self.args.freeze_epochs + 1 and self.frozen:
                self._unfreeze_all()

            self.model.train()
            epoch_loss, seen = 0.0, 0

            # Vòng lặp train – bắt KeyboardInterrupt/RuntimeError để lưu interrupt ckpt
            try:
                for xb, yb, mb in self.train_loader:
                    if self.interrupted:
                        raise KeyboardInterrupt

                    bs = xb.size(0)
                    self.optimizer.zero_grad(set_to_none=True)

                    for s in range(0, bs, micro):
                        x = xb[s:s+micro].to(self.device, non_blocking=True)
                        y = yb[s:s+micro].to(self.device, non_blocking=True)
                        m = mb[s:s+micro].to(self.device, non_blocking=True)

                        used_mix = False
                        y_soft = None
                        if self.mixup_fn is not None and random.random() < self.args.mixup_prob:
                            x, y_soft = self.mixup_fn(x, y)   # y_soft: [B,2]
                            used_mix = True

                        with torch.amp.autocast('cuda', enabled=(self.device.type=='cuda')):
                            logits_bin, logits_m = self.model(x)
                            if used_mix:
                                loss_bin = self.criterion_bin_soft(logits_bin, y_soft)
                            else:
                                loss_bin = self.criterion_bin_hard(logits_bin, y)
                            loss = loss_bin
                            if not used_mix:
                                mask = (m >= 0)
                                if mask.any():
                                    loss_m = self.criterion_mth(logits_m[mask], m[mask])
                                    loss = loss + self.args.method_loss_weight * loss_m

                        epoch_loss += float(loss.detach().item()) * x.size(0)
                        self.scaler.scale(loss / accum_steps).backward()

                    self.scaler.step(self.optimizer); self.scaler.update()
                    if self.ema: self.ema.update(self.model)

                    step_global += 1
                    lr_scale = cosine_warmup(step_global, self.total_steps, self.args.warmup_steps)
                    for pg in self.optimizer.param_groups: pg['lr'] = self.args.lr * lr_scale
                    lr_now = self.optimizer.param_groups[0]['lr']

                    seen += bs; gpm.step(bs)
                    print(f"\r[Epoch {epoch}/{self.args.epochs}] {gpm.text()} | lr {lr_now:.2e}", end="")

            except KeyboardInterrupt:
                # người dùng nhấn Ctrl+C -> lưu interrupt ckpt, thoát sạch
                print("\n⏸ Bị ngắt bởi người dùng.")
                self.save_ckpt(f"epoch{epoch}_interrupt", epoch=epoch)
                print("✅ Đã lưu checkpoint tạm. Thoát.")
                return
            except RuntimeError as e:
                # khi Ctrl+C, đôi khi DataLoader ném lỗi “worker exited unexpectedly”
                if self.interrupted and ("DataLoader worker" in str(e) or "_queue.Empty" in str(e)):
                    print("\n⏸ Bị ngắt giữa chừng (DataLoader).")
                    self.save_ckpt(f"epoch{epoch}_interrupt", epoch=epoch)
                    print("✅ Đã lưu checkpoint tạm. Thoát.")
                    return
                else:
                    raise

            print()
            val_loss, val_acc, thr, bacc, meth_acc = self.evaluate(use_ema=True, tta=True)
            train_loss = epoch_loss / max(seen, 1)
            print(f"Epoch {epoch}/{self.args.epochs} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | best_thr {thr:.3f} "
                  f"| val_bacc {bacc:.4f} | meth_acc(fake) {meth_acc:.4f}")

            self.best["threshold"] = thr
            self.save_ckpt(f"epoch{epoch}",
                           extra_meta={"threshold": thr, "val_acc": val_acc, "val_bacc": bacc,
                                       "val_method_acc": meth_acc}, epoch=epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                self.best.update({"acc": best_acc, "threshold": thr})
                self.save_ckpt("best",
                               extra_meta={"threshold": thr, "val_acc": val_acc, "val_bacc": bacc,
                                           "val_method_acc": meth_acc}, epoch=epoch)

        print(f"✅ Train xong. Best val_acc = {self.best['acc']:.4f} | threshold={self.best['threshold']:.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/processed/faces")
    p.add_argument("--out_dir", type=str, default="deepfake_detector")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--micro_batch", type=int, default=0, help="chia nhỏ batch để tránh OOM; effective = batch_size")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mixup", type=float, default=0.15)
    p.add_argument("--cutmix", type=float, default=0.15)
    p.add_argument("--mixup_prob", type=float, default=0.7)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--warmup_steps", type=int, default=600)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--grad_ckpt",    dest="grad_ckpt", action="store_true")
    g.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    p.set_defaults(grad_ckpt=True)  # như bản trước: mặc định bật
    p.add_argument("--model", type=str, default="vit_base_patch16_224")
    p.add_argument("--freeze_epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balance_by_method", action="store_true")
    p.add_argument("--method_boost", type=str, default="", help='VD: "Face2Face=2,FaceShifter=2,NeuralTextures=2,FaceSwap=1.5"')
    p.add_argument("--face_crop", action="store_true")
    p.add_argument("--method_loss_weight", type=float, default=0.3)

    # resume
    p.add_argument("--resume", type=str, default="", help="path tới checkpoint để tiếp tục")
    p.add_argument("--auto_resume", action="store_true", help="tự chọn checkpoint mới nhất khi --resume rỗng")
    p.add_argument("--reset_opt", action="store_true", help="khởi tạo optimizer mới khi resume (tiếp từ epoch+1)")

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    t = Trainer(args)
    t.build_data(); t.build_model(); t.maybe_resume(); t.train()

if __name__ == "__main__":
    main()