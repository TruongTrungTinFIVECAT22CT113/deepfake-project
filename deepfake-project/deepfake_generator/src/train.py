# deepfake_generator/src/train.py
# Train LoRA SD v1-5 trên ảnh real, lưu checkpoint kiểu detector (epochN, best, interrupt), hỗ trợ resume.
# Ví dụ:
# py -m deepfake_generator.src.train --data_root data/processed/faces --epochs 2 --max_steps 2000 --batch_size 2 --grad_accum 4 --rank 16 --save_every_steps 500 --resume ""

import argparse, os, math, time, signal, sys, random, io, json
from datetime import timedelta
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class RealFaces(Dataset):
    def __init__(self, data_root, size=512, max_images=80000):
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

def enable_memory_savers(pipe):
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    pipe.enable_attention_slicing(1); pipe.enable_vae_slicing()
    try: pipe.unet.enable_gradient_checkpointing()
    except Exception: pass

def prepare_lora_unet(unet, rank=16):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
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
        attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank)
    unet.set_attn_processor(attn_procs)
    # đóng băng gốc, chỉ train LoRA
    for p in unet.parameters(): p.requires_grad_(False)
    train_params = []
    for _, proc in unet.attn_processors.items():
        for p in proc.parameters():
            p.requires_grad_(True); train_params.append(p)
    return train_params

# --- save/load LoRA ---
def lora_state_dict(unet) -> Dict[str, torch.Tensor]:
    state = {}
    for name, proc in unet.attn_processors.items():
        for pn, p in proc.state_dict().items():
            state[f"{name}.{pn}"] = p.detach().cpu()
    return state

def load_lora_into_unet(unet, state: Dict[str, torch.Tensor]):
    # tạo lại cấu trúc nếu cần
    for name, proc in unet.attn_processors.items():
        # subset theo tiền tố 'name.'
        sub = {k.split(".",1)[1]: v for k,v in state.items() if k.startswith(name+".")}
        if sub:
            proc.load_state_dict(sub, strict=False)

def save_ckpt(out_dir: Path, tag: str, unet, optimizer, epoch, step, meta_extra=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    safep = out_dir / f"lora_{tag}.safetensors"
    ptp   = out_dir / f"lora_{tag}.pt"
    meta = {
        "epoch": int(epoch),
        "step": int(step),
        "rank": unet.attn_processors[list(unet.attn_processors.keys())[0]].rank if hasattr(unet.attn_processors[list(unet.attn_processors.keys())[0]], "rank") else None,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if meta_extra: meta.update(meta_extra)
    # safetensors: chỉ LoRA weights
    safetensors_save(lora_state_dict(unet), str(safep))
    # pt: kèm optimizer để resume
    torch.save({"lora": lora_state_dict(unet),
                "optimizer": optimizer.state_dict(),
                "meta": meta}, str(ptp))
    print(f"\n💾 Saved: {safep.name} & {ptp.name}")

def resume_from_ckpt(unet, optimizer, ckpt_path: str) -> Tuple[int,int]:
    ck = torch.load(ckpt_path, map_location="cpu")
    if "lora" in ck:
        load_lora_into_unet(unet, ck["lora"])
    elif ckpt_path.endswith(".safetensors"):
        state = safetensors_load(ckpt_path)
        load_lora_into_unet(unet, state)
    if "optimizer" in ck:
        try: optimizer.load_state_dict(ck["optimizer"])
        except Exception: pass
    meta = ck.get("meta", {})
    epoch = int(meta.get("epoch", 0)); step = int(meta.get("step", 0))
    print(f"🔁 Resume LoRA từ {ckpt_path} (epoch={epoch}, step={step})")
    return epoch, step

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
    ap.add_argument("--save_every_steps", type=int, default=0, help="0=chỉ lưu theo epoch; >0: lưu mỗi N bước")
    ap.add_argument("--resume", type=str, default="", help="đường dẫn .pt hoặc .safetensors để tiếp tục")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)

    ds = RealFaces(args.data_root, size=args.resolution)
    total_files = len(ds)
    print(f"🔎 Tổng số file huấn luyện (train/real): {total_files}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True, persistent_workers=(args.workers>0))

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained, torch_dtype=torch.float16 if device.type=='cuda' else torch.float32
    )
    pipe.to(device); pipe.safety_checker=None
    enable_memory_savers(pipe)
    if args.compile:
        try: pipe.unet = torch.compile(pipe.unet, mode="max-autotune"); print("🔧 torch.compile enabled")
        except Exception as e: print(f"[WARN] compile off: {e}")

    trainable_params = prepare_lora_unet(pipe.unet, rank=args.rank)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # resume
    start_epoch, step = 0, 0
    if args.resume:
        try:
            start_epoch, step = resume_from_ckpt(pipe.unet, optimizer, args.resume)
        except Exception as e:
            print(f"[WARN] Resume thất bại: {e}")

    total_steps = min(args.max_steps, math.ceil(args.epochs * total_files / (args.batch_size)) // max(1,args.grad_accum))

    prog = type("P", (), {})(); prog.t0=time.time(); prog.done=0; prog.total = total_steps * args.batch_size
    def ptext():
        el=time.time()-prog.t0; rate=prog.done/max(el,1e-9); rem=(prog.total-prog.done)/max(rate,1e-9)
        return (f"{100*prog.done/max(prog.total,1):6.2f}% | {prog.done}/{prog.total} files | "
                f"elapsed {timedelta(seconds=int(el))} | remaining {timedelta(seconds=int(rem))}")

    interrupted = {"flag": False}
    def _sigint(sig, frame):
        print("\n[CTRL+C] Lưu checkpoint LoRA …"); interrupted["flag"]=True
    signal.signal(signal.SIGINT, _sigint)

    ema_loss = None; best_loss = float("inf")

    pipe.unet.train()
    epoch = start_epoch
    for e in range(start_epoch, args.epochs):
        epoch = e
        for batch in dl:
            if interrupted["flag"]:
                save_ckpt(ckpt_dir, f"epoch{epoch}_interrupt", pipe.unet, optimizer, epoch, step)
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
            if (step+1) % args.grad_accum == 0:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)

                w = args.warmup_steps
                gs = (step+1)//args.grad_accum
                lr_scale = gs / max(1,w) if gs < w else 0.5*(1+math.cos(math.pi*(gs-w)/max(1,(total_steps-w))))
                for pg in optimizer.param_groups: pg["lr"] = args.lr * lr_scale

            bs = latents.size(0)
            prog.done += bs; step += 1
            ema_loss = loss.item() if ema_loss is None else (0.1*loss.item() + 0.9*ema_loss)
            print(f"\r[LoRA genericface] {ptext()} | epoch {epoch+1}/{args.epochs} | step {step}/{total_steps} | loss {loss.item():.4f} | ema {ema_loss:.4f}", end="")

            # save theo bước
            if args.save_every_steps and (step % args.save_every_steps == 0):
                save_ckpt(ckpt_dir, f"step{step}", pipe.unet, optimizer, epoch, step)

            # best theo ema_loss trên train
            if ema_loss is not None and ema_loss < best_loss:
                best_loss = ema_loss
                save_ckpt(ckpt_dir, "best", pipe.unet, optimizer, epoch, step, meta_extra={"ema_loss": float(ema_loss)})

            if step >= total_steps: break
        print()
        # lưu theo epoch
        save_ckpt(ckpt_dir, f"epoch{epoch+1}", pipe.unet, optimizer, epoch+1, step)
        if step >= total_steps: break

    # lưu cuối cùng ở thư mục gốc để app dùng
    safetensors_save(lora_state_dict(pipe.unet), str(out_dir / "pytorch_lora_weights.safetensors"))
    with open(out_dir / "lora_meta.json", "w", encoding="utf-8") as f:
        json.dump({"type":"diffusers_lora","rank":args.rank,"final_step":step}, f)
    print(f"✅ Huấn luyện xong. LoRA saved: {out_dir / 'pytorch_lora_weights.safetensors'}")

if __name__ == "__main__":
    main()