# deepfake_detector/src/train.py

# py -m deepfake_detector.src.train --epochs 8 --batch_size 64 --data_root data/processed/faces

import argparse, os, time, math, signal, sys
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# ---------- Utils: ETA-friendly meter ----------
class ProgressMeter:
    def __init__(self, total):
        self.total = total
        self.start = time.time()
        self.done = 0

    def step(self, n=1):
        self.done += n

    def format(self):
        elapsed = time.time() - self.start
        rate = self.done / max(elapsed, 1e-9)
        remaining = (self.total - self.done) / max(rate, 1e-9)
        pct = 100.0 * self.done / max(self.total, 1)
        return (f"{pct:6.2f}% | {self.done}/{self.total} files | "
                f"elapsed {timedelta(seconds=int(elapsed))} | "
                f"remaining {timedelta(seconds=int(remaining))}")

# ---------- Checkpoint-safe trainer ----------
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_dir = Path(args.out_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.interrupted = False

        def _handle_sigint(sig, frame):
            print("\n[CTRL+C] Sắp lưu checkpoint an toàn…")
            self.interrupted = True
        signal.signal(signal.SIGINT, _handle_sigint)

    def build_data(self):
        img_size = 224
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        root = self.args.data_root  # data/processed/faces
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   transform=val_tf)

        self.num_train_files = len(train_ds)
        self.num_val_files = len(val_ds)
        print(f"🔎 Tổng số file train: {self.num_train_files} | val: {self.num_val_files}")
        self.train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.workers, pin_memory=True)
        self.val_loader   = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False,
                                       num_workers=self.args.workers, pin_memory=True)
        self.class_names = train_ds.classes  # ['fake','real'] nếu theo preprocess.py
        print(f"📂 Classes = {self.class_names}")

    def build_model(self):
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        model.to(self.device)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type=='cuda')

    def save_ckpt(self, tag):
        ckpt_path = self.ckpt_dir / f"detector_{tag}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        loss_sum, correct, count = 0.0, 0, 0
        pm = ProgressMeter(total=self.num_val_files)
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            count += x.size(0)
            pm.step(x.size(0))
            print(f"\r[VAL] {pm.format()}", end="")
        print()
        return loss_sum / count, correct / count

    def train(self):
        best_acc = 0.0
        gpm_total = self.num_train_files * self.args.epochs
        gpm = ProgressMeter(total=gpm_total)
        print(f"🚀 Bắt đầu train ({self.args.epochs} epochs) | tổng file cần xử lý: {gpm_total}")

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            epoch_loss, seen = 0.0, 0
            for x, y in self.train_loader:
                if self.interrupted:
                    self.save_ckpt(f"epoch{epoch}_interrupt")
                    print("⏸ Đã lưu checkpoint do tạm dừng bằng Ctrl+C.")
                    sys.exit(0)

                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.device.type=='cuda'):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                bs = x.size(0)
                seen += bs
                gpm.step(bs)
                epoch_loss += loss.item() * bs
                print(f"\r[Epoch {epoch}/{self.args.epochs}] "
                      f"{gpm.format()} | batch_loss {loss.item():.4f}", end="")
            print()

            val_loss, val_acc = self.evaluate(epoch)
            train_loss = epoch_loss / self.num_train_files
            print(f"Epoch {epoch}/{self.args.epochs} | train_loss {train_loss:.4f} "
                  f"| val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

            # checkpoint mỗi epoch
            self.save_ckpt(f"epoch{epoch}")
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_ckpt("best")

        print(f"✅ Train xong. Best val_acc = {best_acc:.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/processed/faces",
                   help="thư mục dữ liệu đã được preprocess (train/val/real|fake)")
    p.add_argument("--out_dir", type=str, default="deepfake_detector")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    t = Trainer(args)
    t.build_data()
    t.build_model()
    t.train()

if __name__ == "__main__":
    main()