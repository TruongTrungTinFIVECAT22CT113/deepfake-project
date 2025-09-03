# eval_method_detailed.py
import os, sys, time, math, csv, argparse, glob, gc
from collections import Counter
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- optional: memory log ----------
def _try_mem():
    used_vram = None
    if torch.cuda.is_available():
        used_vram = torch.cuda.memory_allocated() / (1024**2)
    try:
        import psutil
        used_ram = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except Exception:
        used_ram = None
    return used_vram, used_ram

# ---------- Face crop (Mediapipe) ----------
_FACE_DET = None
def _lazy_face_det():
    global _FACE_DET
    if _FACE_DET is None:
        try:
            import mediapipe as mp
            _FACE_DET = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.3
            )
        except Exception:
            _FACE_DET = False
    return _FACE_DET

def detect_faces_xyxy(img_rgb: np.ndarray) -> List[List[int]]:
    det = _lazy_face_det()
    H, W, _ = img_rgb.shape
    boxes = []
    if not det:
        return boxes
    res = det.process(img_rgb[..., ::-1])  # mediapipe expects BGR
    if not res or not getattr(res, "detections", None):
        return boxes
    for d in res.detections:
        r = d.location_data.relative_bounding_box
        x0 = int(max(0, r.xmin * W)); y0 = int(max(0, r.ymin * H))
        x1 = int(min(W, (r.xmin + r.width) * W))
        y1 = int(min(H, (r.ymin + r.height) * H))
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes

def crop_main_face(pil_img: Image.Image, expand: float = 0.25) -> Image.Image:
    # trả về crop mặt lớn nhất; nếu không thấy mặt thì trả full ảnh
    img_rgb = np.array(pil_img.convert("RGB"))
    H, W = img_rgb.shape[:2]
    boxes = detect_faces_xyxy(img_rgb)
    if not boxes:
        return pil_img
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    x0, y0, x1, y1 = boxes[0]
    dx = int((x1-x0)*expand); dy = int((y1-y0)*expand)
    xx0 = max(0, x0-dx); yy0 = max(0, y0-dy)
    xx1 = min(W, x1+dx); yy1 = min(H, y1+dy)
    return Image.fromarray(img_rgb[yy0:yy1, xx0:xx1])

# ---------- Model ----------
class MultiHeadViT(nn.Module):
    def __init__(self, backbone="vit_base_patch16_224", img_size=224, num_methods=6):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, img_size=img_size)
        feat = self.backbone.num_features
        self.dropout = nn.Dropout(0.1)
        self.head_bin = nn.Linear(feat, 2)           # fake/real
        self.head_m  = nn.Linear(feat, num_methods)  # method classes

    def forward(self, x):
        f = self.backbone(x)
        f = self.dropout(f)
        return self.head_bin(f), self.head_m(f)

def _remap_heads(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        if k.startswith("head_cls."):
            out[k.replace("head_cls.", "head_bin.")] = v
        elif k.startswith("head_mth."):
            out[k.replace("head_mth.", "head_m.")] = v
        else:
            out[k] = v
    return out

def load_detector(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    classes = meta.get("classes", ["fake","real"])
    method_names = meta.get("method_names", ["Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures","Other"])
    mean = meta.get("norm_mean", [0.5,0.5,0.5])
    std  = meta.get("norm_std",  [0.5,0.5,0.5])
    thr  = float(meta.get("threshold", 0.5))
    model_name = meta.get("model_name", "vit_base_patch16_224")
    img_size   = int(meta.get("img_size", 224))

    model = MultiHeadViT(model_name, img_size=img_size, num_methods=len(method_names))
    state = ckpt.get("model_ema") or ckpt.get("model") or ckpt
    state = _remap_heads(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict missing= {list(missing)} unexpected= {list(unexpected)}")
    model.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return model, tfm, classes, method_names, img_size, thr

@torch.no_grad()
def predict_batch(model, xb: torch.Tensor, tta: int = 2, amp: bool = True):
    # xb: (B,3,H,W) on DEVICE -> pbin(B,2), pmth(B,M)
    if amp and DEVICE.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            lb, lm = model(xb)
            if tta >= 2:
                lb2, lm2 = model(torch.flip(xb, dims=[3]))
                lb = (lb + lb2) / 2
                lm = (lm + lm2) / 2
    else:
        lb, lm = model(xb)
        if tta >= 2:
            lb2, lm2 = model(torch.flip(xb, dims=[3]))
            lb = (lb + lb2) / 2
            lm = (lm + lm2) / 2
    pbin = torch.softmax(lb, dim=1).cpu().numpy()
    pmth = torch.softmax(lm, dim=1).cpu().numpy()
    return pbin, pmth

# ---------- Eval per-method ----------
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def collect_images(root_dir: str) -> Dict[str, List[str]]:
    out = {}
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if not os.path.isdir(p):
            continue
        imgs = []
        for ext in IMG_EXTS:
            imgs += glob.glob(os.path.join(p, f"*{ext}"))
        if imgs:
            out[p] = imgs
    return out

def normalize_method_name(name: str) -> str:
    n = os.path.basename(name).strip().lower()
    alias = {
        "deepfake": "deepfakes", "deepfakedetection":"deepfakes", "deepfakes":"deepfakes",
        "face2face":"face2face", "faceshifter":"faceshifter", "faceswap":"faceswap",
        "neuraltexture":"neuraltextures", "neuraltextures":"neuraltextures", "other":"other",
    }
    return alias.get(n, n)

def _load_one(path: str, tfm, face_crop: bool):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if face_crop:
                img = crop_main_face(img)
            return tfm(img)
    except Exception:
        return None

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def run_eval(args):
    model, tfm, classes, method_names, img_size, thr_ckpt = load_detector(args.ckpt)
    thr = float(args.threshold) if args.threshold is not None else thr_ckpt

    # ép workers=0 khi dùng face_crop để tránh Mediapipe ăn RAM/không thread-safe
    workers = args.workers
    if args.face_crop and workers > 0:
        print("[WARN] --face_crop bật → ép --workers 0 để tránh Mediapipe ngốn RAM / lỗi thread.")
        workers = 0

    print(f"== CKPT: {args.ckpt} | Model={model.backbone.__class__.__name__} | img={img_size} | "
          f"Thr={thr:.3f} | TTA={args.tta} | face_crop={args.face_crop} | batch={args.batch} | workers={workers}")

    meta_map = {normalize_method_name(m): i for i, m in enumerate(method_names)}
    all_dirs = collect_images(args.root)
    if not all_dirs:
        print(f"[!] Không tìm thấy ảnh ở: {args.root}")
        return

    os.makedirs("reports", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join("reports", f"method_eval_{ts}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["method_dir","meta_label","N","det_fake","correct_method","wrong_method","missed_real","acc_fake","acc_method_overall"])

        for method_dir, img_paths in all_dirs.items():
            target_key = normalize_method_name(method_dir)
            m_idx = meta_map.get(target_key, None)
            if m_idx is None:
                print(f"[WARN] Bỏ qua '{method_dir}' vì không khớp meta labels {method_names}")
                continue

            N = len(img_paths)
            det_fake = 0; correct_method = 0; wrong_method = 0; missed_real = 0
            conf_wrong = Counter()

            # progress bar theo folder
            pbar = tqdm(total=N, desc=os.path.basename(method_dir), unit="img")

            # ThreadPoolExecutor: chỉ tạo 1 lần / folder khi KHÔNG face_crop
            executor = None
            if workers > 0:
                executor = ThreadPoolExecutor(max_workers=workers)

            try:
                for batch_idx, paths in enumerate(chunked(img_paths, args.batch), 1):
                    # load song song nếu executor != None
                    if executor is not None:
                        xs = [None] * len(paths)
                        futures = {executor.submit(_load_one, p, tfm, args.face_crop): i for i, p in enumerate(paths)}
                        for fut in as_completed(futures):
                            i = futures[fut]
                            xs[i] = fut.result()
                    else:
                        xs = [_load_one(p, tfm, args.face_crop) for p in paths]

                    # lọc ảnh đọc lỗi
                    keep_tensors = [x for x in xs if x is not None]
                    missed_real += (len(xs) - len(keep_tensors))
                    if not keep_tensors:
                        pbar.update(len(paths))
                        continue

                    xb = torch.stack(keep_tensors, dim=0).to(DEVICE, non_blocking=True)
                    pbin, pmth = predict_batch(model, xb, tta=args.tta, amp=(not args.no_amp))

                    for pf, pm in zip(pbin, pmth):
                        p_fake = float(pf[0])
                        if p_fake >= thr:
                            det_fake += 1
                            pred_m = int(np.argmax(pm))
                            if pred_m == m_idx:
                                correct_method += 1
                            else:
                                wrong_method += 1
                                conf_wrong[method_names[pred_m]] += 1
                        else:
                            missed_real += 1

                    # progress + memory log
                    pbar.update(len(paths))
                    if args.log_mem and (batch_idx % args.log_mem == 0):
                        vram_mb, ram_mb = _try_mem()
                        msg = []
                        if vram_mb is not None: msg.append(f"VRAM={vram_mb:.0f}MB")
                        if ram_mb is not None:  msg.append(f"RAM={ram_mb:.0f}MB")
                        if msg: pbar.set_postfix_str(" | ".join(msg))

                    # cleanup batch
                    del xb, xs, keep_tensors, pbin, pmth
                    if torch.cuda.is_available() and (batch_idx % 10 == 0):
                        torch.cuda.empty_cache()
                    if batch_idx % 20 == 0:
                        gc.collect()

            finally:
                pbar.close()
                if executor is not None:
                    executor.shutdown(wait=True)

            acc_fake = det_fake / max(1, N)
            acc_method_overall = correct_method / max(1, N)

            print(f"{method_dir:<18} | N={N:6d} | det_fake={det_fake:6d} ({acc_fake*100:5.1f}%) | "
                  f"correct={correct_method:6d} ({acc_method_overall*100:5.1f}%) | wrong={wrong_method:6d} | missed={missed_real:6d}")

            if conf_wrong:
                top = ", ".join([f"{k}:{v}" for k,v in conf_wrong.most_common(5)])
                print(f"   → nhầm nhiều sang: {top}")

            wr.writerow([method_dir, method_names[m_idx], N, det_fake, correct_method, wrong_method, missed_real,
                         f"{acc_fake:.6f}", f"{acc_method_overall:.6f}"])

    print(f"\n✔ Đã lưu báo cáo CSV: {csv_path}")

if __name__ == "__main__":
    pa = argparse.ArgumentParser("Eval từng loại phương pháp deepfake (per-folder)")
    pa.add_argument("--ckpt", type=str, required=True, help="đường dẫn checkpoint .pt (có meta)")
    pa.add_argument("--root", type=str, default="data/processed/faces/val/fake", help="thư mục gốc chứa các thư mục phương pháp")
    pa.add_argument("--tta", type=int, default=2)
    pa.add_argument("--face_crop", action="store_true", help="crop mặt bằng Mediapipe (chậm)")
    pa.add_argument("--threshold", type=float, default=None, help="ngưỡng fake (mặc định lấy từ ckpt meta)")

    # tốc độ / bộ nhớ
    pa.add_argument("--batch", type=int, default=64, help="batch size infer (tăng nếu còn VRAM)")
    pa.add_argument("--workers", type=int, default=0, help="số luồng đọc/tiền xử lý ảnh (0 = tuần tự). Với --face_crop sẽ bị ép = 0")
    pa.add_argument("--no_amp", action="store_true", help="tắt autocast fp16 khi có CUDA (mặc định bật)")
    pa.add_argument("--log_mem", type=int, default=0, help="in VRAM/RAM mỗi N batch (0 = tắt)")
    args = pa.parse_args()
    run_eval(args)