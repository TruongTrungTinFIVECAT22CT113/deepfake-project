# deepfake_detector/src/train_trans.py
import os, re, time, json, math, random, argparse, csv, signal
from typing import List
from glob import glob
import gc

import torch
import torch._logging  
import logging         
torch._logging.set_logs(inductor=logging.ERROR)
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm
import timm
from torchvision import transforms
import torch._dynamo as torchdynamo
from torch.compiler import cudagraph_mark_step_begin

# ----------------- Global speed knobs -----------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from contextlib import nullcontext

def get_sdpa_ctx():
    """
    Dùng kwargs + list để hợp lệ với TorchDynamo và PyTorch 2.x:
    - kwargs: tránh assert của Dynamo
    - list: tránh AssertionError 'Backend must be ... list of SDPBackend'
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(backends=[
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH
        ])
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel
            # Fallback cho bản cũ của CUDA SDP
            return sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=True
            )
        except Exception:
            return nullcontext()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

BRANCHES = ["face","head","full"]

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def list_images_sorted(d: str):
    exts = (".jpg",".jpeg",".png",".bmp")
    xs = [os.path.join(d,f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts]
    xs.sort(key=natural_key)
    return xs

def parse_boost_map(s: str):
    mp={}
    if not s: return mp
    for kv in s.split(","):
        kv=kv.strip()
        if not kv: continue
        k,v = kv.split("=")
        mp[k.strip()] = float(v)
    return mp

# ----------------- Dataset: Video clips (sliding window) -----------------
class VideoClipsDataset(Dataset):
    """
    Hỗ trợ 2 convention song song:

    A) Kiểu cũ (per-branch suffix):
       data_root/
         face|head|full/
           train|val|test/
             real_face|real_head|real_full/<video-id>/*.jpg
             fake_face|fake_head|fake_full/<Method>/<video-id>/*.jpg

    B) Kiểu mới (preprocess_balanced.py):
       data_root/
         face|head|full/
           train|val|test/
             real/<Dataset>/<video-id>/*.jpg
             fake/<Method>/<video-id>/*.jpg

    Nếu temporal_jitter == 0:
      - Duy trì hành vi cũ: tiền tạo mọi clip (cửa sổ cố định).
    Nếu temporal_jitter > 0:
      - Lưu (video, window_idx) và jitter start tại __getitem__:
        start' = clamp(window_idx*stride + randint(-J,+J), 0, n-T)
    """
    def __init__(self, data_root: str, split: str, img_size: int, frames_per_clip: int, clip_stride: int,
                 branch_filter: str="any", color_jitter: float=0.3, rand_erase_p: float=0.0,
                 random_resized_crop: bool=False, temporal_jitter: int = 0,
                 balance_by_method: bool = False, boost_map: dict | None = None):
        self.data_root = data_root
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.clip_stride = clip_stride
        self.branch_filter = branch_filter
        self.img_size = img_size
        self.temporal_jitter = max(0, int(temporal_jitter))
        self.balance_by_method = bool(balance_by_method)
        self.boost_map = boost_map or {}

        # transforms
        tforms=[]
        if random_resized_crop:
            tforms.append(transforms.RandomResizedCrop(img_size, scale=(0.7,1.0), interpolation=InterpolationMode.BILINEAR, antialias=False))
        else:
            tforms.append(transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR, antialias=False))
        tforms += [transforms.RandomHorizontalFlip(0.5)]
        if color_jitter>0:
            cj=color_jitter
            tforms += [transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=min(0.1,cj*0.2))]
        tforms += [transforms.PILToTensor()]
        if rand_erase_p > 0:
            tforms += [transforms.RandomErasing(p=rand_erase_p, value="random")]
        self.tform = transforms.Compose(tforms)

        # Dạng lưu:
        # - Nếu jitter == 0: self.items = [(br, yb, ym, [frame_paths[T]])... ] (như cũ)
        # - Nếu jitter > 0:  self.videos = [(br, yb, ym, frames_all)], self.items = [(vid_idx, window_idx)]
        self.items=[]
        self.videos=[]
        self.stats={"real":0, "fake":0}
        self.method_names=set()

        for br in BRANCHES:
            if branch_filter!="any" and br!=branch_filter:
                continue
            split_dir = os.path.join(data_root, br, split)
            if not os.path.isdir(split_dir):
                continue

            # ---- Xác định thư mục REAL/FAKE theo cả 2 convention ----
            cand_reals = [os.path.join(split_dir, f"real_{br}"), os.path.join(split_dir, "real")]
            cand_fakes = [os.path.join(split_dir, f"fake_{br}"), os.path.join(split_dir, "fake")]

            real_dir = next((d for d in cand_reals if os.path.isdir(d)), None)
            fake_dir = next((d for d in cand_fakes if os.path.isdir(d)), None)

            # ---------------------- REAL ----------------------
            if real_dir:
                for entry in sorted([d for d in glob(os.path.join(real_dir, "*")) if os.path.isdir(d)], key=natural_key):
                    frames = list_images_sorted(entry)
                    if len(frames) > 0:
                        self._register_video(br, "real", None, entry, frames)
                        continue
                    for vid in sorted([d for d in glob(os.path.join(entry, "*")) if os.path.isdir(d)], key=natural_key):
                        frames = list_images_sorted(vid)
                        if len(frames) > 0:
                            self._register_video(br, "real", None, vid, frames)

            # ---------------------- FAKE ----------------------
            if fake_dir:
                for method_name in sorted([d for d in os.listdir(fake_dir) if os.path.isdir(os.path.join(fake_dir, d))], key=natural_key):
                    meth_path = os.path.join(fake_dir, method_name)
                    for vid in sorted([d for d in glob(os.path.join(meth_path, "*")) if os.path.isdir(d)], key=natural_key):
                        frames = list_images_sorted(vid)
                        if len(frames) > 0:
                            self._register_video(br, "fake", method_name, vid, frames)
                            self.method_names.add(method_name)

        self.method_names = sorted(self.method_names)
        self.method_to_idx = {m:i for i,m in enumerate(self.method_names)}

        # ===== Build self.items =====
        if self.temporal_jitter == 0:
            # Tiền tạo danh sách clip (đường dẫn từng frame)
            items = []
            for (br, kind, method_name, frames) in self.videos:
                T = self.frames_per_clip
                S = self.clip_stride
                n = len(frames)
                if n < T:
                    continue
                mult = self._mult_for(method_name) if (self.split == "train" and kind == "fake" and (self.balance_by_method or self.boost_map)) else 1
                num_win = 0
                for start in range(0, n - T + 1, S):
                    idxs  = list(range(start, start + T))
                    paths = [frames[i] for i in idxs]
                    for _ in range(mult):
                        items.append((br, 1 if kind == "real" else 0, method_name if kind == "fake" else None, paths))
                    num_win += 1
                self.stats[kind] += num_win * mult
            self.items = items
        else:
            # Jitter: lưu (video_idx, window_idx), jitter tại __getitem__
            items = []
            for vid_idx, (br, kind, method_name, frames) in enumerate(self.videos):
                T = self.frames_per_clip
                S = self.clip_stride
                n = len(frames)
                if n < T:
                    continue
                num_win = 1 + (n - T) // S
                mult = self._mult_for(method_name) if (self.split == "train" and kind == "fake" and (self.balance_by_method or self.boost_map)) else 1
                self.stats[kind] += num_win * mult
                for w in range(num_win):
                    for _ in range(mult):
                        items.append((vid_idx, w))
            self.items = items

    def _mult_for(self, method_name: str | None) -> int:
        if not method_name:
            return 1
        if method_name in self.boost_map:
            return max(1, int(round(float(self.boost_map[method_name]))))
        return 1

    def _register_video(self, br: str, kind: str, method_name: str, vid_dir: str, frames: List[str]):
        # Lưu toàn bộ frame list để có thể jitter tại __getitem__
        self.videos.append((br, kind, method_name, frames))

    def __len__(self): return len(self.items)

    def __getitem__(self, i:int):
        if self.temporal_jitter == 0:
            # Hành vi cũ: đã có list T frame sẵn
            br, yb, ym, paths = self.items[i]
            frames=[]
            for p in paths:
                try:
                    im = Image.open(p).convert("RGB"); frames.append(self.tform(im))
                except Exception:
                    frames.append(frames[-1].clone() if frames else torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32))
            clip=torch.stack(frames, dim=0)  # [T,C,H,W]
            ym_idx = self.method_to_idx.get(ym, -1)
            return clip, torch.tensor(yb, dtype=torch.long), torch.tensor(ym_idx, dtype=torch.long), br

        else:
            # Jitter: tính lại start' quanh cửa sổ gốc
            vid_idx, w = self.items[i]
            br, kind, method_name, frames = self.videos[vid_idx]
            yb = 1 if kind=="real" else 0
            ym_idx = self.method_to_idx.get(method_name, -1)

            T=self.frames_per_clip; S=self.clip_stride; n=len(frames)
            base_start = w * S
            if n < T:
                # fallback an toàn (không nên xảy ra)
                base_start = 0
            max_start = max(0, n - T)
            if self.temporal_jitter > 0 and max_start > 0:
                shift = random.randint(-self.temporal_jitter, self.temporal_jitter)
                start = min(max(base_start + shift, 0), max_start)
            else:
                start = min(base_start, max_start)

            paths = frames[start:start+T]
            imgs=[]
            for p in paths:
                try:
                    im = Image.open(p).convert("RGB"); imgs.append(self.tform(im))
                except Exception:
                    imgs.append(imgs[-1].clone() if imgs else torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32))
            clip=torch.stack(imgs, dim=0)  # [T,C,H,W]
            return clip, torch.tensor(yb, dtype=torch.long), torch.tensor(ym_idx, dtype=torch.long), br

# ----------------- Backbone Encoder -----------------
class BackboneEncoder(nn.Module):
    def __init__(self, backbone_name: str, drop_rate=0.1, drop_path_rate=0.1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, drop_rate=drop_rate,
                                          drop_path_rate=drop_path_rate, global_pool='avg')
        try:
            exp_h, exp_w = self.backbone.patch_embed.img_size
        except Exception:
            exp_h = exp_w = 224
        dummy = torch.zeros(1, 3, exp_h, exp_w)
        with torch.no_grad():
            feat = self.backbone(dummy)
        self.feat_dim = feat.shape[-1]

    def forward(self, x_btchw):
        B,T,C,H,W = x_btchw.shape
        x = x_btchw.reshape(B*T, C, H, W)
        x = x.contiguous(memory_format=torch.channels_last) 
        try:
            exp_h, exp_w = self.backbone.patch_embed.img_size
        except Exception:
            exp_h = exp_w = x.shape[-1]
        if x.shape[-2] != exp_h or x.shape[-1] != exp_w:
            x = F.interpolate(x, size=(exp_h, exp_w), mode='bilinear', align_corners=False)
        f = self.backbone(x)            # [B*T, D]
        f = f.view(B, T, -1)            # [B, T, D]
        return f

# ----------------- Temporal Transformer -----------------
class TemporalTransformer(nn.Module):
    """
    Transformer theo thời gian cho chuỗi đặc trưng frame:
    - Positional embedding learnable (max_len đủ lớn).
    - Tuỳ chọn pool: 'mean' hoặc 'cls' (prepend 1 token).
    """
    def __init__(self, d_model=512, nhead=8, layers=3, drop=0.1, pool="mean", max_len=1024):
        super().__init__()
        self.pool = pool
        self.d_model = d_model
        self.max_len = max_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.drop = nn.Dropout(drop)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x_btd):  # [B,T,D]
        B,T,D = x_btd.shape
        if T > self.max_len:
            # cắt an toàn nếu vượt quá max_len
            x_btd = x_btd[:, :self.max_len, :]
            T = x_btd.size(1)

        pe = self.pos_embed[:, :T, :]  # [1,T,D]
        x = x_btd + pe
        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)  # [B,1,D]
            x = torch.cat([cls, x], dim=1)          # [B,1+T,D]

        x = self.drop(x)
        with get_sdpa_ctx():
            x = self.enc(x)                         # [B,T,D] hoặc [B,1+T,D]

        x = self.ln(x)

        if self.pool == "cls":
            return x[:,0,:]                         # [B,D]
        else:
            return x.mean(dim=1)                    # [B,D]

class VideoTemporalTransformer(nn.Module):
    def __init__(self, backbone_name: str, num_methods: int, d_model=512,
                 nhead=8, trans_layers=3, drop_rate=0.1, drop_path_rate=0.1,
                 pool="mean"):
        super().__init__()
        self.enc = BackboneEncoder(backbone_name, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        D = self.enc.feat_dim
        self.proj = nn.Identity() if D==d_model else nn.Linear(D, d_model)
        self.temporal = TemporalTransformer(
            d_model=d_model, nhead=nhead, layers=trans_layers, drop=drop_rate, pool=pool
        )
        out_dim = d_model
        self.head_bin = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(out_dim, 2))
        self.head_met = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(out_dim, num_methods))

    def forward(self, clip_btchw):  # [B,T,C,H,W]
        f = self.enc(clip_btchw)     # [B,T,D]
        f = self.proj(f)             # [B,T,d_model]
        ht = self.temporal(f)        # [B, d_model]
        return self.head_bin(ht), self.head_met(ht)
    
def build_model(backbone_model: str, img_size: int, d_model: int, nhead: int, trans_layers: int,
                drop_rate: float, drop_path_rate: float, num_methods: int, pretrained: bool=False,
                pool: str="mean"):
        # img_size không dùng trực tiếp trong VideoTemporalTransformer (resize đã xử lý ở Dataset),
        # nhưng giữ tham số để phù hợp args.
        return VideoTemporalTransformer(
            backbone_name=backbone_model,
            num_methods=num_methods,
            d_model=d_model,
            nhead=nhead,
            trans_layers=trans_layers,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pool=pool
        )
# ----------------- EMA -----------------
class EMA:
    def __init__(self, model: nn.Module, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.data.detach().clone() for n,p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n,p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.shadow:
                self.shadow[n] = p.data.detach().clone()
            else:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)

    def apply_to(self, model: nn.Module):
        self._backup_cpu = {}
        for n,p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._backup_cpu[n] = p.data.detach().cpu()
            p.data.copy_(self.shadow[n].to(p.device, non_blocking=True))

    def restore(self, model: nn.Module):
        if not hasattr(self, "_backup_cpu"):
            return
        for n,p in model.named_parameters():
            if n in self._backup_cpu:
                p.data.copy_(self._backup_cpu[n].to(p.device, non_blocking=True))
        del self._backup_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_shadow_to_cpu(self):
        for n in list(self.shadow.keys()):
            self.shadow[n] = self.shadow[n].cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload_shadow_to_gpu(self, device):
        for n in list(self.shadow.keys()):
            self.shadow[n] = self.shadow[n].to(device, non_blocking=True)

# ----------------- Metrics -----------------
@torch.no_grad()
def evaluate(
    model, loader, device, method_names, thr_min, thr_max, thr_steps,
    label_smoothing=0.0, use_ema=False, ema_obj: EMA = None, phase_name="VAL",
    eval_micro_batch: int = 0  # <-- NEW: cho phép chia nhỏ batch khi eval để hạ VRAM
):
    if use_ema and ema_obj is not None:
        ema_obj.apply_to(model)
    model.eval()

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 1, 3, 1, 1)

    probs = []
    yb_all = []
    ym_all = []
    method_correct = 0
    method_total = 0

    for clip, yb, ym, _ in tqdm(loader, desc=f"[{phase_name}]", dynamic_ncols=True):
        # To device + chuẩn hóa
        clip = clip.to(device, non_blocking=True).float().div_(255.0)
        yb   = yb.to(device, non_blocking=True)
        ym   = ym.to(device, non_blocking=True)

        # --- Eval micro-batch to lower peak VRAM ---
        if eval_micro_batch and 0 < eval_micro_batch < clip.size(0):
            mb = eval_micro_batch
            probs_mb = []

            s = 0
            while s < clip.size(0):
                e = min(clip.size(0), s + mb)
                clip_s = clip[s:e]
                # normalize (inside autocast is fine)
                with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    clip_s = (clip_s - mean) / std
                    lb_s, lm_s = model(clip_s)

                # binary prob (fake prob index 0 như code gốc)
                p_s = torch.softmax(lb_s.float(), dim=1)[:, 0]
                probs_mb.append(p_s.detach().cpu())

                # method acc trên mẫu fake
                sel = (ym[s:e] >= 0) & (yb[s:e] == 0)
                if sel.any():
                    pred_m = torch.argmax(lm_s, dim=1)
                    method_correct += int((pred_m[sel] == ym[s:e][sel]).sum().item())
                    method_total   += int(sel.sum().item())

                s = e

            p = torch.cat(probs_mb, dim=0)

        else:
            # Full batch forward with autocast (tiết kiệm VRAM)
            with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                clip = (clip - mean) / std
                lb, lm = model(clip)

            p = torch.softmax(lb.float(), dim=1)[:, 0]

            sel = (ym >= 0) & (yb == 0)
            if sel.any():
                pred_m = torch.argmax(lm, dim=1)
                method_correct += int((pred_m[sel] == ym[sel]).sum().item())
                method_total   += int(sel.sum().item())

        probs.append(p.detach().cpu())
        yb_all.append(yb.detach().cpu())
        ym_all.append(ym.detach().cpu())

    probs = torch.cat(probs)
    yb = torch.cat(yb_all)
    ym = torch.cat(ym_all)

    thrs = torch.linspace(thr_min, thr_max, steps=thr_steps)
    correct_fake = torch.zeros_like(thrs)
    correct_real = torch.zeros_like(thrs)
    tot_fake = int((yb == 0).sum().item())
    tot_real = int((yb == 1).sum().item())

    for i, t in enumerate(thrs):
        pred = (probs >= t).long()
        pred = torch.where(pred == 1, torch.zeros_like(pred), torch.ones_like(pred))
        m_fake = (yb == 0); m_real = (yb == 1)
        correct_fake[i] = (pred[m_fake] == 0).sum()
        correct_real[i] = (pred[m_real] == 1).sum()

    rec_fake = (correct_fake / max(1, tot_fake)).cpu()
    rec_real = (correct_real / max(1, tot_real)).cpu()
    bacc = 0.5 * (rec_fake + rec_real)
    acc  = (correct_fake + correct_real).cpu() / float(tot_fake + tot_real + 1e-12)
    best_idx = int(torch.argmax(bacc))

    if use_ema and ema_obj is not None:
        ema_obj.restore(model)

    res = {
        "acc": float(acc[best_idx]),
        "bacc": float(bacc[best_idx]),
        "best_thr": float(thrs[best_idx].cpu()),
        "rec_fake": float(rec_fake[best_idx]),
        "rec_real": float(rec_real[best_idx]),
        "method_acc_fake": (method_correct / method_total if method_total > 0 else 0.0),
    }

    print(f"[KQ {phase_name}] acc={res['acc']:.4f} | bacc={res['bacc']:.4f} | rec_fake={res['rec_fake']:.4f} | rec_real={res['rec_real']:.4f} | thr*={res['best_thr']:.3f} | m_acc(fake)={res['method_acc_fake']:.4f}")
    return res

# ----------------- memory -----------------
def mem():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(f"[MEM] used={used/1024**3:.2f} GB / total={total/1024**3:.2f} GB")

# ----------------- main -----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--micro_batch", type=int, default=0, help="Nếu >0 và <batch_size → chia micro-batch để tích luỹ gradient")
    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--backbone_model", default="vit_base_patch16_384",
                    help="Tên model timm (vd: tf_efficientnet_b4.in1k | vit_base_patch16_384 | xception71.tf_in1k)")
    ap.add_argument("--drop_rate", type=float, default=0.10)
    ap.add_argument("--drop_path_rate", type=float, default=0.10)

    # Temporal Transformer
    ap.add_argument("--d_model", type=int, default=512, help="Chiều sau khi chiếu đặc trưng backbone → đầu vào Transformer")
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--trans_layers", type=int, default=3)
    ap.add_argument("--temporal_pool", choices=["mean","cls"], default="mean")

    # Clip sampling
    ap.add_argument("--frames_per_clip", type=int, default=20)
    ap.add_argument("--clip_stride", type=int, default=3)
    ap.add_argument("--temporal_jitter", type=int, default=0, help="±J khung cho TRAIN (0 = tắt)")
    ap.add_argument("--temporal_jitter_eval", type=int, default=0, help="±J khung cho VAL/TEST (0 = tắt)")
    ap.add_argument("--branch", choices=["any","face","head","full"], default="any")

    # aug
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--mixup_prob", type=float, default=0.5)
    ap.add_argument("--color_jitter", type=float, default=0.30)
    ap.add_argument("--rand_erase_p", type=float, default=0.10)
    ap.add_argument("--random_resized_crop", action="store_true")

    # loss
    ap.add_argument("--bin_weight_real", type=float, default=1.5)
    ap.add_argument("--focal_gamma", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.07)
    ap.add_argument("--method_loss_weight", type=float, default=0.2)

    # optim
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_after_unfreeze", type=float, default=2e-5, help="LR dùng sau khi backbone được unfreeze")
    ap.add_argument("--cosine_min_lr", type=float, default=1e-6, help="eta_min cho cosine decay")
    ap.add_argument("--warmup_steps", type=int, default=600)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # threshold sweep
    ap.add_argument("--thr_min", type=float, default=0.0)
    ap.add_argument("--thr_max", type=float, default=1.0)
    ap.add_argument("--thr_steps", type=int, default=101)

    # EMA / early stop
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--val_use_ema", action="store_true")
    ap.add_argument("--early_stop_patience", type=int, default=2)

    # Eval mode
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_split", choices=["val","test"], default="val")
    ap.add_argument("--ckpt", type=str, default="", help="Đường dẫn đến checkpoint .pt để eval_only")
    ap.add_argument('--val_batch', type=int, default=None, help='Batch size for val/test (default: equals --batch_size)')
    ap.add_argument('--eval_micro_batch', type=int, default=0, help='If >0 and <val_batch, chunk eval batch to lower peak VRAM')

    # sampler balancing
    ap.add_argument("--balance_by_method", action="store_true", help="Cân bằng sampler theo từng phương pháp fake")
    ap.add_argument("--method_boost", type=str, default="", help='Boost per-method, ví dụ "NeuralTextures=1.5,Audio2Animation=1.3"')

    # init backbone & freeze epochs
    ap.add_argument("--init_backbone_ckpt", type=str, default="",
                    help="Đường dẫn ckpt non-time để nạp backbone (bỏ các head cũ)")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0,
                    help="Số epoch đầu tiên đóng băng backbone; 0 = không freeze")
    ap.add_argument("--unfreeze_last_blocks", type=int, default=0,
                    help="0=unfreeze toàn bộ backbone; >0 chỉ mở K blocks cuối của ViT")

    # NEW: save every epoch
    ap.add_argument("--save_all_epochs", action="store_true",
                    help="Nếu bật, lưu thêm detector_e{epoch}.pt mỗi epoch bên cạnh bản best")

    args=ap.parse_args()

    boost_map = parse_boost_map(args.method_boost)
    val_bs = args.val_batch if args.val_batch is not None else args.batch_size

    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets
    ds_train=VideoClipsDataset(args.data_root,'train', 
                               args.img_size,args.frames_per_clip, 
                               args.clip_stride,
                               branch_filter=args.branch, 
                               color_jitter=args.color_jitter,
                               rand_erase_p=args.rand_erase_p, 
                               random_resized_crop=args.random_resized_crop, 
                               temporal_jitter=args.temporal_jitter,
                               balance_by_method=args.balance_by_method,   
                               boost_map=boost_map  
                               )
    ds_val  =VideoClipsDataset(args.data_root,'val',  
                               args.img_size,args.frames_per_clip,
                               args.clip_stride,
                               branch_filter=args.branch, 
                               color_jitter=0.0, rand_erase_p=0.0, 
                               random_resized_crop=False, 
                               temporal_jitter=args.temporal_jitter_eval
                               )

    has_test=False
    for br in BRANCHES:
        if args.branch!="any" and br!=args.branch: continue
        if os.path.isdir(os.path.join(args.data_root, br, 'test')): has_test=True
    ds_test = None
    if has_test:
        ds_test=VideoClipsDataset(args.data_root,'test', 
                                  args.img_size,args.frames_per_clip,
                                  args.clip_stride,
                                  branch_filter=args.branch, 
                                  color_jitter=0.0, 
                                  rand_erase_p=0.0, 
                                  random_resized_crop=False,
                                  temporal_jitter=args.temporal_jitter_eval
                                  )

    method_names = ds_train.method_names
    print(f"📊 Methods: {method_names}")
    print(f"🔎 TRAIN clips: {len(ds_train)} | VAL: {len(ds_val)} | TEST: {0 if ds_test is None else len(ds_test)}")
    print(f"🧩 Stats: {ds_train.stats}")

    # Dataloaders
    dl_train=DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, prefetch_factor=1, persistent_workers=True)
    dl_val  =DataLoader(ds_val, batch_size=val_bs, shuffle=False, num_workers=args.workers,
                        pin_memory=True, drop_last=False, prefetch_factor=1, persistent_workers=True)
    dl_test=None
    if ds_test is not None and len(ds_test)>0:
        dl_test=DataLoader(ds_test, batch_size=val_bs, shuffle=False, num_workers=args.workers,
                           pin_memory=True, drop_last=False, prefetch_factor=1, persistent_workers=True)

    # model (Temporal Transformer)
    model = build_model(
        backbone_model=args.backbone_model,
        img_size=args.img_size,
        d_model=args.d_model,
        nhead=args.nhead,
        trans_layers=args.trans_layers,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        num_methods=len(method_names),
        pretrained=True,
        pool=args.temporal_pool
    ).to(device)
    # ---- Compile (Inductor + Triton) với fallback aot_eager ----
    if torch.__version__ >= "2.0.0":
        torchdynamo.config.suppress_errors = False
        if os.getenv("TORCH_COMPILE_DISABLE", "0") != "1" and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="default")  # inductor + triton
            except Exception as e:
                print(f"[compile] Inductor failed: {e}\n→ Falling back to aot_eager.")
                torchdynamo.config.suppress_errors = True
                model = torch.compile(model, mode="reduce-overhead", backend="aot_eager")

    # init backbone from ckpt
    if args.init_backbone_ckpt:
        try:
            ck = torch.load(args.init_backbone_ckpt, map_location="cpu")
            sd = ck.get("state_dict", ck.get("model", ck))
            bb = model.enc.backbone.state_dict()
            filtered = {k: v for k,v in sd.items() if k in bb and v.shape==bb[k].shape}
            missing, unexpected = model.enc.backbone.load_state_dict(filtered, strict=False)
            print(f"🔁 Loaded backbone from {args.init_backbone_ckpt} | missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"⚠️  Không nạp được backbone từ ckpt: {e}")

    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
    ema = EMA(model, 0.9999) if args.ema else None

    # class weights
    pos_w = torch.tensor([1.0, args.bin_weight_real], device=device)

    # schedule helpers
    total_steps = args.epochs * max(1,len(dl_train))
    warmup = args.warmup_steps
    def cosine_lr(step_global:int, base_lr:float, eta_min:float):
        t = max(0, step_global - warmup)
        T = max(1, total_steps - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * t / T))
        return eta_min + (base_lr - eta_min) * cos

    def set_backbone_requires_grad(req: bool, last_k: int = 0):
        bb = model.enc.backbone
        for p in bb.parameters():
            p.requires_grad = False
        if req:
            if last_k and hasattr(bb, "blocks"):
                for blk in bb.blocks[-last_k:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                if hasattr(bb, "norm"):
                    for p in bb.norm.parameters():
                        p.requires_grad = True
            else:
                for p in bb.parameters():
                    p.requires_grad = True

    if args.freeze_backbone_epochs > 0:
        set_backbone_requires_grad(False)
        print(f"🧊 Freeze backbone for first {args.freeze_backbone_epochs} epochs")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path=os.path.join(args.out_dir, "train_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=[
                "epoch","time_min",
                "train_acc","train_bacc","train_rec_fake","train_rec_real","train_thr_star",
                "val_acc","val_bacc","val_rec_fake","val_rec_real","val_thr_star","val_method_acc_fake",
                "test_acc","test_bacc","test_rec_fake","test_rec_real","test_thr_star","test_method_acc_fake"
            ]); w.writeheader()

    # eval-only
    if args.eval_only:
        if not args.ckpt or not os.path.isfile(args.ckpt):
            raise RuntimeError("--eval_only cần --ckpt trỏ tới file .pt hợp lệ")

        ck = torch.load(args.ckpt, map_location="cpu")
        sd = ck.get("state_dict", ck.get("model", ck))
        model.load_state_dict(sd, strict=True)
        print(f"🔁 Loaded model weights từ {args.ckpt}")

        loader = dl_val if args.eval_split=="val" else dl_test
        if loader is None: raise RuntimeError(f"Không có dữ liệu cho split '{args.eval_split}'.")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if ema is not None and args.val_use_ema: ema.offload_shadow_to_cpu()
        evaluate(model, loader, device, method_names, args.thr_min, args.thr_max, args.thr_steps,
                 label_smoothing=args.label_smoothing, use_ema=(args.val_use_ema and ema is not None), ema_obj=ema,
                 phase_name=args.eval_split.upper())
        if ema is not None and args.val_use_ema: ema.reload_shadow_to_gpu(device)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return

    interrupted={'flag':False}
    def on_sigint(sig,frame): interrupted['flag']=True
    signal.signal(signal.SIGINT, on_sigint)

    best_metric=-1.0; epochs_no_improve=0
    unfreezed = (args.freeze_backbone_epochs <= 0)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1,1,3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1,1,3,1,1)

    for epoch in range(1, args.epochs+1):
        if args.freeze_backbone_epochs>0 and epoch==args.freeze_backbone_epochs+1:
            set_backbone_requires_grad(True, last_k=args.unfreeze_last_blocks)
            unfreezed = True
            print(f"🧊→🔥 Unfreeze backbone at epoch {epoch}")

        model.train()
        t0=time.time()
        pbar=tqdm(dl_train, dynamic_ncols=True, desc=f"[Epoch {epoch}/{args.epochs}]")
        last_end = time.perf_counter()
        opt.zero_grad(set_to_none=True)

        accum = max(1, int(args.batch_size // max(1,args.micro_batch))) if (args.micro_batch and args.micro_batch>0 and args.micro_batch<args.batch_size) else 1
        micro = args.micro_batch if (args.micro_batch and args.micro_batch>0 and args.micro_batch<args.batch_size) else args.batch_size

        for step,(clip,yb,ym,_) in enumerate(pbar):
            t0_iter = time.perf_counter()
            data_ms = (t0_iter - last_end) * 1000.0
            if interrupted['flag']:
                ck=os.path.join(args.out_dir, f"detector_interrupt_e{epoch}_s{step}.pt")
                torch.save({'state_dict': model.state_dict(), 'args': vars(args), 'method_names': method_names}, ck)
                print(f"\n⛔ Dừng thủ công. Đã lưu {ck}")
                return

            if not (accum > 1):
                clip = clip.to(device, non_blocking=True)
                clip = clip.float().div_(255.0)
                clip = (clip - mean) / std
                yb   = yb.to(device, non_blocking=True)
                ym   = ym.to(device, non_blocking=True)

            # mixup
            if args.mixup>0 and args.mixup_prob>0 and random.random()<args.mixup_prob:
                idx = torch.randperm(clip.size(0), device=clip.device if clip.is_cuda else torch.device("cpu"))
                lam = random.betavariate(args.mixup, args.mixup); lam=max(0,min(1,lam))
                clip = lam*clip + (1-lam)*clip[idx]
                yb_soft = lam*F.one_hot(yb,2).float() + (1-lam)*F.one_hot(yb[idx],2).float()
                mixed=True
            else:
                yb_soft=None; mixed=False

            # micro-batch accumulation
            if accum > 1:
                B = clip.size(0)
                lb_last = None
                yb_last = None
                for mb_start in range(0, B, micro):
                    mb_end  = min(B, mb_start + micro)

                    clip_mb = clip[mb_start:mb_end].to(device, non_blocking=True)
                    clip_mb = clip_mb.float().div_(255.0)
                    clip_mb = (clip_mb - mean) / std
                    yb_mb   = yb[mb_start:mb_end].to(device, non_blocking=True)
                    ym_mb   = ym[mb_start:mb_end].to(device, non_blocking=True)

                    with get_sdpa_ctx():
                        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                            cudagraph_mark_step_begin()
                            lb, lm = model(clip_mb)

                            if mixed and yb_soft is not None:
                                yb_soft_mb = yb_soft[mb_start:mb_end].to(device, non_blocking=True)
                                loss_b = F.kl_div(F.log_softmax(lb.float(), dim=1),
                                                  yb_soft_mb, reduction='batchmean')
                            else:
                                loss_b = F.cross_entropy(lb.float(), yb_mb,
                                                         weight=pos_w,
                                                         label_smoothing=args.label_smoothing)

                            loss_m = 0.0
                            if args.method_loss_weight > 0 and (ym_mb >= 0).any():
                                loss_m = F.cross_entropy(lm.float(), ym_mb.clamp_min(0), reduction='mean')

                            loss = (loss_b + args.method_loss_weight * loss_m) / accum

                        scaler.scale(loss).backward()
                        lb_last = lb
                        yb_last = yb_mb

                if args.clip_grad and args.clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if ema is not None: ema.update(model)
                lb_ref = lb_last; yb_ref = yb_last
            else:
                with get_sdpa_ctx():
                    with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                        lb, lm = model(clip)
                        if mixed and yb_soft is not None:
                            loss_b = F.kl_div(F.log_softmax(lb.float(), dim=1),
                                              yb_soft, reduction='batchmean')
                        else:
                            loss_b = F.cross_entropy(lb.float(), yb,
                                                     weight=pos_w,
                                                     label_smoothing=args.label_smoothing)
                        loss_m = 0.0
                        if args.method_loss_weight > 0 and (ym >= 0).any():
                            loss_m = F.cross_entropy(lm.float(), ym.clamp_min(0), reduction='mean')
                        loss = loss_b + args.method_loss_weight * loss_m

                scaler.scale(loss).backward()
                if args.clip_grad and args.clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if ema is not None: ema.update(model)
                lb_ref = lb; yb_ref = yb

            # LR schedule
            step_global = (epoch-1)*len(dl_train) + step + 1
            base_lr = (args.lr_after_unfreeze if unfreezed else args.lr)
            if warmup>0 and step_global <= warmup:
                lr_now = base_lr * step_global / warmup
            else:
                lr_now = cosine_lr(step_global, base_lr, args.cosine_min_lr)
            for pg in opt.param_groups:
                pg['lr'] = lr_now

            # progress
            with torch.no_grad():
                p = torch.softmax(lb_ref.detach().float(), dim=1)[:,0]
                pred=(p>=0.5).long()
                pred=torch.where(pred==1, torch.zeros_like(pred), torch.ones_like(pred))
                acc=(pred==yb_ref).float().mean().item()
                last_end = time.perf_counter()
                iter_ms = (last_end - t0_iter) * 1000.0
                comp_ms = max(0.0, iter_ms - data_ms)

                if step % 100 == 0:
                    pbar.set_postfix_str(
                        f"loss={loss.item():.4f} acc~={acc:.4f} data={data_ms:.1f}ms comp={comp_ms:.1f}ms lr={opt.param_groups[0]['lr']:.2e}"
                    )

        # quick TRAIN summary (batch cuối)
        train_acc=train_bacc=train_rf=train_rr=train_thr=None
        try:
            p = torch.softmax(lb_ref.detach().float().cpu(), dim=1)[:,0]
            y = yb_ref.detach().cpu()
            thrs = torch.linspace(args.thr_min,args.thr_max,steps=51)
            best_b=-1
            for t in thrs:
                pred=torch.where(p>=t, torch.zeros_like(y), torch.ones_like(y))
                acc=(pred==y).float().mean().item()
                m_fake=(y==0); m_real=(y==1)
                rf=(pred[m_fake]==0).float().mean().item() if m_fake.any() else 0.0
                rr=(pred[m_real]==1).float().mean().item() if m_real.any() else 0.0
                b=0.5*(rf+rr)
                if b>best_b: best_b,train_acc,train_rf,train_rr,train_thr=b,acc,rf,rr,float(t.item())
            train_bacc=best_b
        except:
            pass

        # ========== EVAL ==========
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if ema is not None and args.val_use_ema: ema.offload_shadow_to_cpu()

        mem()
        val_res = evaluate(model, dl_val, device, method_names, args.thr_min, args.thr_max, args.thr_steps,
                           label_smoothing=args.label_smoothing, use_ema=(args.val_use_ema and ema is not None), ema_obj=ema,
                           phase_name="VAL", eval_micro_batch=args.eval_micro_batch)
        mem()

        test_res=None
        if dl_test is not None:
            print("\n[ĐÁNH GIÁ TEST] ——")
            mem()
            test_res = evaluate(model, dl_test, device, method_names, args.thr_min, args.thr_max, args.thr_steps,
                                label_smoothing=args.label_smoothing, use_ema=(args.val_use_ema and ema is not None), ema_obj=ema,
                                phase_name="TEST", eval_micro_batch=args.eval_micro_batch)
            mem()

        if ema is not None and args.val_use_ema: ema.reload_shadow_to_gpu(device)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ---- CSV log ----
        row = {
            "epoch": epoch, "time_min": round((time.time()-t0)/60.0,3),
            "train_acc": train_acc, "train_bacc": train_bacc, "train_rec_fake": train_rf, "train_rec_real": train_rr, "train_thr_star": train_thr,
            "val_acc": val_res["acc"], "val_bacc": val_res["bacc"], "val_rec_fake": val_res["rec_fake"], "val_rec_real": val_res["rec_real"], "val_thr_star": val_res["best_thr"],
            "test_acc": None if test_res is None else test_res["acc"],
            "test_bacc": None if test_res is None else test_res["bacc"],
            "test_rec_fake": None if test_res is None else test_res["rec_fake"],
            "test_rec_real": None if test_res is None else test_res["rec_real"],
            "test_thr_star": None if test_res is None else test_res["best_thr"],
            "test_method_acc_fake": None if test_res is None else test_res["method_acc_fake"],
        }
        with open(csv_path,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=[
                "epoch","time_min",
                "train_acc","train_bacc","train_rec_fake","train_rec_real","train_thr_star",
                "val_acc","val_bacc","val_rec_fake","val_rec_real","val_thr_star","val_method_acc_fake",
                "test_acc","test_bacc","test_rec_fake","test_rec_real","test_thr_star","test_method_acc_fake"
            ]); w.writerow(row)

        # ---- save every epoch (optional) ----
        if args.save_all_epochs:
            ep_path = os.path.join(args.out_dir, f"detector_temporal_e{epoch}.pt")
            payload = {
                "state_dict": model.state_dict(),
                "args": vars(args),                     # để build lại kiến trúc
                "method_names": method_names,           # đưa lên top-level
                "epoch": epoch,
                "metadata": {
                    "img_size": args.img_size,
                    "backbone_model": args.backbone_model,
                    "temporal": {
                        "type": "Transformer",
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "layers": args.trans_layers,
                        "pool": args.temporal_pool,
                    }
                },
                # tuỳ chọn cho resume:
                # "optimizer": optimizer.state_dict(),
                # "scheduler": scheduler.state_dict() if scheduler else None,
                # "scaler": scaler.state_dict() if scaler else None,
                # "ema_state_dict": ema.averaged_model.state_dict() if use_ema else None,
            }
            torch.save(payload, ep_path)
            print(f"📝 Lưu EPOCH {epoch} → {ep_path}")

        # ---- save best ----
        is_best = (val_res["bacc"] > best_metric)
        if is_best:
            best_metric = val_res["bacc"]
            best_path = os.path.join(args.out_dir, "detector_temporal_best.pt")
            payload = {
                "state_dict": model.state_dict(),
                "args": vars(args),
                "method_names": method_names,
                "best_val_bacc": best_metric,
                "best_thr": val_res.get("best_thr", None),
                "metadata": {
                    "img_size": args.img_size,
                    "backbone_model": args.backbone_model,
                    "temporal": {
                        "type": "Transformer",
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "layers": args.trans_layers,
                        "pool": args.temporal_pool,
                    }
                },
                # tuỳ chọn cho resume:
                # "optimizer": optimizer.state_dict(),
                # "scheduler": scheduler.state_dict() if scheduler else None,
                # "scaler": scaler.state_dict() if scaler else None,
                # "ema_state_dict": ema.averaged_model.state_dict() if use_ema else None,
                # "epoch": epoch,
            }
            torch.save(payload, best_path)
            print(f"💾 Lưu BEST → {best_path} (thr*={val_res.get('best_thr', None)})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"🛑 Early stopping sau {args.early_stop_patience} epoch không cải thiện BACC.")
            break

    print("✅ Train xong.")

if __name__ == "__main__":
    main()
