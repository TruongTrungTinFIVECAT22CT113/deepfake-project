# deepfake_generator/src/train.py

# py -m deepfake_generator.src.train --data_root data/processed/faces --max_steps 800

import argparse, os, math, time, signal, sys, random
from datetime import timedelta
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup

# ---- Dataset: lấy ảnh ở data/processed/faces/train/real ----
class RealFaces(Dataset):
    def __init__(self, data_root, size=512, max_images=5000):
        self.paths = []
        real_dir = Path(data_root) / "train" / "real"
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for p in real_dir.rglob("*"):
            if p.suffix.lower() in exts:
                self.paths.append(str(p))
        if len(self.paths) == 0:
            print(f"Không tìm thấy ảnh trong {real_dir}")
            sys.exit(1)
        # để train “nhẹ”, lấy ngẫu nhiên tối đa N ảnh
        if len(self.paths) > max_images:
            self.paths = random.sample(self.paths, max_images)
        self.size = size

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB").resize((self.size, self.size))
        # to [-1,1] CHW
        x = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()/255.
        ).view(self.size, self.size, 3).permute(2,0,1)
        x = x*2-1
        return {"pixel_values": x}

# ---- Tiện ích ETA/Progress ----
class Progress:
    def __init__(self, total): self.t0=time.time(); self.total=total; self.done=0
    def step(self, n=1): self.done+=n
    def text(self):
        el=time.time()-self.t0
        rate=self.done/max(el,1e-9)
        rem=(self.total-self.done)/max(rate,1e-9)
        return (f"{100*self.done/max(self.total,1):6.2f}% | {self.done}/{self.total} files | "
                f"elapsed {timedelta(seconds=int(el))} | remaining {timedelta(seconds=int(rem))}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed/faces",
                    help="thư mục gốc đã preprocess (sẽ dùng train/real)")
    ap.add_argument("--out_dir", type=str, default="deepfake_generator/outputs")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    lora_path = out_dir / "lora_genericface.pt"

    # Dataset lấy từ train/real
    ds = RealFaces(args.data_root, size=args.resolution)
    total_files = len(ds)
    print(f"🔎 Tổng số file huấn luyện (train/real): {total_files}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True)

    # Base SD1.5
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device.type=='cuda' else torch.float32
    ).to(device)
    pipe.safety_checker = None
    unet: UNet2DConditionModel = pipe.unet

    # LoRA “giản lược”: thêm 1 adapter nhỏ cho mọi attention block (demo)
    loras = {}
    for name, module in unet.attn_processors.items():
        rank = 8
        loras[name] = torch.nn.Sequential(
            torch.nn.Linear(module.to_q.in_features, rank, bias=False),
            torch.nn.Linear(rank, module.to_q.in_features, bias=False),
        ).to(device)
    optim = torch.optim.AdamW([p for m in loras.values() for p in m.parameters()],
                              lr=args.lr, weight_decay=1e-4)

    num_update_steps = min(args.max_steps, math.ceil(args.epochs * total_files / args.batch_size))
    sched = get_cosine_schedule_with_warmup(optim, args.warmup_steps, num_update_steps)

    interrupted = {"flag": False}
    def _sigint(sig, frame):
        print("\n[CTRL+C] Lưu checkpoint LoRA …"); interrupted["flag"]=True
    signal.signal(signal.SIGINT, _sigint)

    progress = Progress(total=num_update_steps * args.batch_size)
    step = 0
    unet.train()
    for epoch in range(args.epochs):
        for batch in dl:
            if interrupted["flag"]:
                torch.save({k:v.state_dict() for k,v in loras.items()}, ckpt_dir/f"lora_step{step}.pt")
                print("⏸ Đã lưu checkpoint do Ctrl+C."); sys.exit(0)

            with torch.no_grad():
                latents = pipe.vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()*0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # prompt “genericface” để điều hoà attention — không nhắm 1 danh tính cụ thể
            cond = pipe.tokenizer(["portrait of a generic face"]*latents.size(0),
                                  return_tensors="pt", padding=True).to(device)
            text_emb = pipe.text_encoder(**cond).last_hidden_state

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample
            # LoRA sửa q-proj (demo)
            for name, module in unet.attn_processors.items():
                q = module.to_q(noisy_latents)
                noise_pred = noise_pred + 1e-3 * loras[name](q)

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step(); sched.step()

            bs = latents.size(0)
            progress.step(bs); step += 1
            print(f"\r[LoRA genericface] {progress.text()} | step {step}/{num_update_steps} | loss {loss.item():.4f}", end="")
            if step >= num_update_steps: break
        print()
        if step >= num_update_steps: break

    # Save
    torch.save({k:v.state_dict() for k,v in loras.items()}, lora_path)
    print(f"✅ Huấn luyện xong. LoRA saved: {lora_path}")

if __name__ == "__main__":
    main()