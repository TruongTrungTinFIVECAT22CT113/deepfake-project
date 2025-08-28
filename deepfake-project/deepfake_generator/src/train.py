# deepfake_generator/src/train.py
# py -m deepfake_generator.src.train --data_root data/processed/faces --epochs 3 --max_steps 3000 --batch_size 2 --grad_accum 4 --rank 16 --compile

import argparse, os, math, time, signal, sys, random
from datetime import timedelta
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor
from safetensors.torch import save_file as safetensors_save

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class RealFaces(Dataset):
    def __init__(self, data_root, size=512, max_images=8000):
        real_dir = Path(data_root) / "train" / "real"
        exts = (".jpg",".jpeg",".png",".bmp",".webp")
        self.paths = [str(p) for p in real_dir.rglob("*") if p.suffix.lower() in exts]
        if len(self.paths)==0:
            print(f"Không tìm thấy ảnh trong {real_dir}"); sys.exit(1)
        if len(self.paths) > max_images:
            self.paths = random.sample(self.paths, max_images)
        self.size = size

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB").resize((self.size, self.size))
        x = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()/255.
        ).view(self.size, self.size, 3).permute(2,0,1)  # CHW
        x = x*2-1
        return {"pixel_values": x}

class Progress:
    def __init__(self, total): self.t0=time.time(); self.total=total; self.done=0
    def step(self, n=1): self.done+=n
    def text(self):
        el=time.time()-self.t0
        rate=self.done/max(el,1e-9)
        rem=(self.total-self.done)/max(rate,1e-9)
        return (f"{100*self.done/max(self.total,1):6.2f}% | {self.done}/{self.total} files | "
                f"elapsed {timedelta(seconds=int(el))} | remaining {timedelta(seconds=int(rem))}")

def enable_memory_savers(pipe):
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    try: pipe.unet.enable_gradient_checkpointing()
    except Exception: pass

def prepare_lora_unet(unet, rank=16):
    """
    Tạo LoRA processor đúng kích thước hidden_size cho từng attention block
    dựa trên tên layer (down_blocks / mid_block / up_blocks).
    """
    attn_procs = {}
    for name in unet.attn_processors.keys():
        # attn1 = self-attn (không có cross attention dim)
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        # Suy ra hidden_size theo vị trí block
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]

        attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank
        )
    unet.set_attn_processor(attn_procs)

    # Chỉ train tham số LoRA
    for p in unet.parameters():
        p.requires_grad_(False)
    train_params = []
    for _, proc in unet.attn_processors.items():
        for p in proc.parameters():
            p.requires_grad_(True); train_params.append(p)
    return train_params

def save_lora_safetensors(unet, out_dir):
    """
    Lưu state_dict của tất cả LoRA attention processors
    bằng tên file chuẩn để diffusers nạp lại dễ dàng.
    """
    os.makedirs(out_dir, exist_ok=True)
    state = {}
    for name, proc in unet.attn_processors.items():
        for pn, p in proc.state_dict().items():
            state[f"{name}.{pn}"] = p.detach().cpu()
    # Tên file CHUẨN cho diffusers:
    safetensors_save(state, os.path.join(out_dir, "pytorch_lora_weights.safetensors"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed/faces", help="sẽ dùng train/real")
    ap.add_argument("--out_dir", type=str, default="deepfake_generator/outputs/generic_lora")
    ap.add_argument("--pretrained", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--compile", action="store_true", default=False)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)

    ds = RealFaces(args.data_root, size=args.resolution)
    total_files = len(ds)
    print(f"🔎 Tổng số file huấn luyện (train/real): {total_files}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained, torch_dtype=torch.float16 if device.type=='cuda' else torch.float32
    )
    pipe.to(device); pipe.safety_checker=None
    enable_memory_savers(pipe)

    # chuẩn bị LoRA
    trainable_params = prepare_lora_unet(pipe.unet, rank=args.rank)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    total_steps = min(args.max_steps, math.ceil(args.epochs * total_files / (args.batch_size)) // args.grad_accum)
    progress = Progress(total= total_steps * args.batch_size)
    interrupted = {"flag": False}
    def _sigint(sig, frame):
        print("\n[CTRL+C] Lưu checkpoint LoRA …"); interrupted["flag"]=True
    signal.signal(signal.SIGINT, _sigint)

    if args.compile:
        try: pipe.unet = torch.compile(pipe.unet)
        except Exception: pass

    step, accum = 0, 0
    pipe.unet.train()
    for epoch in range(args.epochs):
        for batch in dl:
            if interrupted["flag"]:
                save_lora_safetensors(pipe.unet, str(ckpt_dir / f"step{step}"))
                print("⏸ Đã lưu checkpoint do Ctrl+C."); sys.exit(0)

            with torch.no_grad():
                latents = pipe.vae.encode(batch["pixel_values"].to(device)).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            prompt = ["portrait of a generic face"]*latents.size(0)
            cond = pipe.tokenizer(prompt, return_tensors="pt", padding=True).to(device)
            text_emb = pipe.text_encoder(**cond).last_hidden_state

            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            (loss/args.grad_accum).backward()
            accum += 1
            if accum % args.grad_accum == 0:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                step += 1

                # warmup + cosine
                w = args.warmup_steps
                lr_scale = step / max(1,w) if step < w else 0.5*(1+math.cos(math.pi*(step-w)/max(1,(total_steps-w))))
                for pg in optimizer.param_groups: pg["lr"] = args.lr * lr_scale

            bs = latents.size(0)
            progress.step(bs)
            print(f"\r[LoRA genericface] {progress.text()} | step {step}/{total_steps} | loss {loss.item():.4f}", end="")

            if step >= total_steps: break
        print()
        if step >= total_steps: break

    # save cuối (đúng tên file để app nạp)
    save_lora_safetensors(pipe.unet, str(out_dir))
    with open(out_dir / "lora_meta.json", "w", encoding="utf-8") as f:
        f.write('{"type":"diffusers_lora"}')
    print(f"✅ Huấn luyện xong. LoRA saved folder: {out_dir}")

if __name__ == "__main__":
    main()