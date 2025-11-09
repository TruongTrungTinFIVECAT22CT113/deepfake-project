# -*- coding: utf-8 -*-
"""
Batch evaluation for backend (standalone) — mirrors eval_videos_vit_facecrop.py logic.

- For REAL videos:
    accuracy = r_frames / n_frames, with r_frames = n_frames - f_frames, where f_frames = count(p_fake >= thr)
- For FAKE videos:
    video_accuracy = c_frames / n_frames, where c_frames = count(p_fake >= thr AND pred_method == true_method)

- Face detector:
    default: RetinaFace (InsightFace, GPU if available). Fallback: CPU provider.
    option:  --detector_backend mediapipe  (CPU)

Outputs:
    results_real.csv  and  results_fake.csv
"""

import os, csv, argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import timm
import cv2
from torchvision import transforms
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ---------------- Model khớp train_vit.py (giống eval_videos_vit_facecrop.py) ----------------
class MultiHeadViT(nn.Module):
    def __init__(self, model_name: str, img_size: int,
                 num_methods: int, num_face_classes: int, num_head_classes: int, num_full_classes: int,
                 drop_rate: float=0.0, drop_path_rate: float=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, img_size=img_size,
            drop_rate=drop_rate, drop_path_rate=drop_path_rate
        )
        feat = self.backbone.num_features

        def head(n):
            return nn.Sequential(
                nn.Dropout(p=drop_rate if drop_rate > 0 else 0.0),
                nn.Linear(feat, n)
            )

        self.head_bin  = head(2)
        self.head_met  = head(num_methods)
        self.head_face = head(max(1, num_face_classes))
        self.head_head = head(max(1, num_head_classes))
        self.head_full = head(max(1, num_full_classes))

    def forward(self, x):
        f = self.backbone(x)
        return self.head_bin(f), self.head_met(f), self.head_face(f), self.head_head(f), self.head_full(f)

def _infer_head_sizes_from_ckpt_state(ckpt_model_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    sizes = {"num_methods": 0, "num_face_classes": 1, "num_head_classes": 1, "num_full_classes": 1}
    if "head_met.1.weight"  in ckpt_model_state: sizes["num_methods"]      = ckpt_model_state["head_met.1.weight"].shape[0]
    if "head_face.1.weight" in ckpt_model_state: sizes["num_face_classes"] = ckpt_model_state["head_face.1.weight"].shape[0]
    if "head_head.1.weight" in ckpt_model_state: sizes["num_head_classes"] = ckpt_model_state["head_head.1.weight"].shape[0]
    if "head_full.1.weight" in ckpt_model_state: sizes["num_full_classes"] = ckpt_model_state["head_full.1.weight"].shape[0]
    return sizes

def _filter_state_dict_by_shape(dst_state: Dict[str, torch.Tensor], src_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in src_state.items() if k in dst_state and dst_state[k].shape == v.shape}

def load_checkpoint_build_model(ckpt_path: str, device: torch.device,
                                img_size_arg: Optional[int],
                                model_name_arg: Optional[str]) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    model_state = ckpt.get("model", {})
    if not model_state:
        raise RuntimeError("Checkpoint thiếu key 'model'.")

    head_sizes = _infer_head_sizes_from_ckpt_state(model_state)
    num_methods       = head_sizes["num_methods"] or len(meta.get("method_names", [])) or 7
    num_face_classes  = head_sizes["num_face_classes"]
    num_head_classes  = head_sizes["num_head_classes"]
    num_full_classes  = head_sizes["num_full_classes"]

    model_name = model_name_arg or meta.get("backbone_model") or meta.get("model_name", "vit_base_patch16_384")
    img_size   = img_size_arg  or meta.get("img_size", 384)

    model = MultiHeadViT(model_name, img_size, num_methods, num_face_classes, num_head_classes, num_full_classes).to(device)
    dst = model.state_dict()
    if "ema" in ckpt and ckpt["ema"]:
        ema_state = ckpt["ema"]
        dst.update(_filter_state_dict_by_shape(dst, ema_state))
        dst.update(_filter_state_dict_by_shape(dst, model_state))
        model.load_state_dict(dst, strict=False)
    else:
        dst.update(_filter_state_dict_by_shape(dst, model_state))
        model.load_state_dict(dst, strict=False)
    model.eval()

    if not meta.get("method_names"):
        meta["method_names"] = [f"method_{i}" for i in range(num_methods)]
    if "img_size" not in meta:   meta["img_size"]   = img_size
    if "model_name" not in meta: meta["model_name"] = model_name
    if "best_thr" not in meta:   meta["best_thr"]   = 0.5
    return model, meta

# ---------------- Face detection backends ----------------
class RetinaFaceLargest:
    def __init__(self, device: torch.device, det_thresh: float = 0.5, det_size: int = 640):
        from insightface.app import FaceAnalysis
        if device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]; ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]; ctx_id = -1
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self.det_thresh = det_thresh

    def detect_largest(self, bgr) -> Optional[Tuple[int,int,int,int]]:
        faces = self.app.get(bgr)
        faces = [f for f in faces if getattr(f, "det_score", 1.0) >= self.det_thresh]
        if not faces: return None
        def area(face): x1,y1,x2,y2 = face.bbox; return (x2-x1)*(y2-y1)
        best = max(faces, key=area)
        x1,y1,x2,y2 = [int(round(v)) for v in best.bbox]
        return x1,y1,x2,y2

def mp_detect_all(bgr) -> List[Tuple[int,int,int,int,float]]:
    import mediapipe as mp
    H,W = bgr.shape[:2]
    det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    res = det.process(bgr[:, :, ::-1])
    out=[]
    if res.detections:
        for d in res.detections:
            score = d.score[0] if d.score else 0.0
            box = d.location_data.relative_bounding_box
            x = int(round(box.xmin * W)); y = int(round(box.ymin * H))
            w = int(round(box.width * W)); h = int(round(box.height * H))
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(W, x + max(1,w)); y2 = min(H, y + max(1,h))
            if x2 > x1 and y2 > y1: out.append((x1,y1,x2,y2,float(score)))
    return out

def square_crop_from_bbox(bgr, bbox, scale: float=1.10):
    H,W = bgr.shape[:2]
    x1,y1,x2,y2 = bbox
    cx = (x1+x2)/2.0; cy=(y1+y2)/2.0
    side = max(x2-x1, y2-y1) * scale
    nx1 = max(0, int(round(cx-side/2))); ny1 = max(0, int(round(cy-side/2)))
    nx2 = min(W, int(round(cx+side/2))); ny2 = min(H, int(round(cy+side/2)))
    if nx2 <= nx1 or ny2 <= ny1: return None
    return bgr[ny1:ny2, nx1:nx2]

def build_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ---------------- Video batching ----------------
def iter_video_batches(video_path: str, batch_size: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = float(total / fps) if fps > 0 else 0.0

    batches, buf = [], []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        buf.append((idx, frame_bgr))
        if len(buf) == batch_size:
            batches.append(buf); buf = []
        idx += 1
    if buf: batches.append(buf)
    cap.release()
    return total, duration, batches

def preprocess_face_batch(batch, detector_backend: str, device: torch.device,
                          tx, img_size: int, bbox_scale: float, retina_det: Optional[RetinaFaceLargest]):
    imgs = []; ok_flags=[]
    for _, bgr in batch:
        if detector_backend == "mediapipe":
            dets = mp_detect_all(bgr)
            if dets:
                dets.sort(key=lambda t: t[4], reverse=True)
                x1,y1,x2,y2,_ = dets[0]
                crop = square_crop_from_bbox(bgr, (x1,y1,x2,y2), scale=bbox_scale)
            else:
                crop = None
        else:
            bb = retina_det.detect_largest(bgr) if retina_det is not None else None
            crop = square_crop_from_bbox(bgr, bb, scale=bbox_scale) if bb is not None else None

        if crop is None or crop.size == 0:
            ok_flags.append(False)
            imgs.append(torch.zeros(3, img_size, img_size))
        else:
            imgs.append(tx(crop))
            ok_flags.append(True)
    return torch.stack(imgs, 0), ok_flags

# ---------------- REAL eval ----------------
@torch.no_grad()
def eval_real_dir(model, device, img_size: int, in_root: Path, out_csv: str,
                  thr: float, batch_size: int, detector_backend: str,
                  bbox_scale: float, fake_index: int, retina_det: Optional[RetinaFaceLargest]):
    tx = build_eval_transform(img_size)
    rows = []
    for ds in [p for p in sorted(in_root.glob("*")) if p.is_dir()]:
        videos = [v for v in sorted(ds.rglob("*")) if v.suffix.lower() in VIDEO_EXTS]
        for v in tqdm(videos, desc=f"[REAL] {ds.name}", unit="vid"):
            try:
                n_frames, duration, batches = iter_video_batches(str(v), batch_size)
                if n_frames == 0:
                    rows.append([v.stem, 0, 0, 0, 0.0, thr, 0.0, ds.name]); continue
                fake_cnt = 0
                for b in batches:
                    x, ok = preprocess_face_batch(b, detector_backend, device, tx, img_size, bbox_scale, retina_det)
                    x = x.to(device, non_blocking=True)
                    log_bin, log_met, *_ = model(x)
                    probs = torch.softmax(log_bin, dim=1)[:, fake_index]
                    for i, o in enumerate(ok):
                        if not o: continue
                        if probs[i].item() >= thr: fake_cnt += 1
                real_cnt = n_frames - fake_cnt
                acc = real_cnt / max(1, n_frames)
                rows.append([v.stem, n_frames, fake_cnt, real_cnt, round(duration,3), thr, round(acc,4), ds.name])
            except Exception as e:
                rows.append([v.stem, 0, 0, 0, 0.0, thr, 0.0, f"{ds.name} (ERROR: {e})"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Videoid","n_frames","f_frames","r_frames","duration","threshold","accuracy","dataset"])
        w.writerows(rows)

# ---------------- FAKE eval ----------------
@torch.no_grad()
def eval_fake_dir(model, device, img_size: int, in_root: Path, out_csv: str,
                  thr: float, batch_size: int, detector_backend: str,
                  bbox_scale: float, method_names: List[str], fake_index: int,
                  retina_det: Optional[RetinaFaceLargest]):
    tx = build_eval_transform(img_size)
    rows = []
    name2idx = {n: i for i, n in enumerate(method_names)}
    for mdir in [p for p in sorted(in_root.glob("*")) if p.is_dir()]:
        mname = mdir.name
        vids = [v for v in sorted(mdir.rglob("*")) if v.suffix.lower() in VIDEO_EXTS]
        for v in tqdm(vids, desc=f"[FAKE] {mname}", unit="vid"):
            try:
                n_frames, duration, batches = iter_video_batches(str(v), batch_size)
                if n_frames == 0:
                    rows.append([v.stem, 0, 0, 0, 0, 0, 0.0, thr, 0.0, mname]); continue
                fake_cnt = 0; correct_m = 0
                true_idx = name2idx.get(mname, None)
                for b in batches:
                    x, ok = preprocess_face_batch(b, detector_backend, device, tx, img_size, bbox_scale, retina_det)
                    x = x.to(device, non_blocking=True)
                    log_bin, log_met, *_ = model(x)
                    probs  = torch.softmax(log_bin, dim=1)[:, fake_index]
                    m_pred = torch.softmax(log_met, dim=1).argmax(1)
                    for i, o in enumerate(ok):
                        if not o: continue
                        p_fake = probs[i].item() >= thr
                        if p_fake:
                            fake_cnt += 1
                            if true_idx is not None and (m_pred[i].item() == true_idx):
                                correct_m += 1
                real_cnt = n_frames - fake_cnt
                c_frames = correct_m
                m_frames = max(0, fake_cnt - c_frames)
                acc = c_frames / max(1, n_frames)
                rows.append([v.stem, n_frames, fake_cnt, real_cnt, c_frames, m_frames,
                             round(duration,3), thr, round(acc,4), mname])
            except Exception as e:
                rows.append([v.stem, 0, 0, 0, 0, 0, 0.0, thr, 0.0, f"{mname} (ERROR: {e})"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Videoid","n_frames","f_frames","r_frames","c_frames","m_frames","duration","threshold","accuracy","method"])
        w.writerows(rows)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser("Backend batch evaluator (mirror eval_videos_vit_facecrop.py)")
    ap.add_argument("--root", type=str, required=True, help="data/videos_test")
    ap.add_argument("--ckpt", type=str, required=True, help="detector_best.pt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--img_size", type=int, default=0, help="0 = lấy từ ckpt.meta.img_size")
    ap.add_argument("--model_name", type=str, default="", help="rỗng = lấy từ ckpt.meta.model_name/backbone_model")
    ap.add_argument("--batch", type=int, default=64, help="frame batch size")
    ap.add_argument("--threshold", type=float, default=-1.0, help="<0 = dùng ckpt.meta.best_thr hoặc 0.5")
    ap.add_argument("--bbox_scale", type=float, default=1.10, help="nới ô vuông quanh bbox (1.0 = tight)")
    ap.add_argument("--det_thr", type=float, default=0.5, help="ngưỡng RetinaFace")
    ap.add_argument("--fake_index", type=int, default=0, help="chỉ số lớp FAKE trong head nhị phân (0 hoặc 1)")
    ap.add_argument("--detector_backend", type=str, default="retinaface", choices=["retinaface","mediapipe"])
    ap.add_argument("--out_real", type=str, default="results_real.csv")
    ap.add_argument("--out_fake", type=str, default="results_fake.csv")
    ap.add_argument("--skip_real", action="store_true", help="Bỏ qua test real/")
    ap.add_argument("--skip_fake", action="store_true", help="Bỏ qua test fake/")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint_build_model(
        ckpt_path=args.ckpt, device=device,
        img_size_arg=(None if args.img_size <= 0 else args.img_size),
        model_name_arg=(None if not args.model_name else args.model_name)
    )

    method_names = list(meta.get("method_names", []))
    img_size = (args.img_size if args.img_size > 0 else meta.get("img_size", 384))
    thr = args.threshold if args.threshold >= 0 else meta.get("best_thr", 0.5)

    retina_det = None
    if args.detector_backend == "retinaface":
        try:
            retina_det = RetinaFaceLargest(device=device, det_thresh=args.det_thr, det_size=640)
        except Exception as e:
            print(f"[WARN] RetinaFace init failed ({e}); fallback to MediaPipe.")
            args.detector_backend = "mediapipe"

    root = Path(args.root)
    real_dir = root / "real"
    fake_dir = root / "fake"

    if (not args.skip_real) and real_dir.is_dir():
        eval_real_dir(model, device, img_size, real_dir, args.out_real, thr, args.batch,
                      detector_backend=args.detector_backend, bbox_scale=args.bbox_scale,
                      fake_index=args.fake_index, retina_det=retina_det)
        print(f"[DONE] Real -> {args.out_real}")
    else:
        print("[SKIP] Real evaluation skipped.")

    if (not args.skip_fake) and fake_dir.is_dir():
        eval_fake_dir(model, device, img_size, fake_dir, args.out_fake, thr, args.batch,
                      detector_backend=args.detector_backend, bbox_scale=args.bbox_scale,
                      method_names=method_names, fake_index=args.fake_index, retina_det=retina_det)
        print(f"[DONE] Fake -> {args.out_fake}")
    else:
        print("[SKIP] Fake evaluation skipped.")

if __name__ == "__main__":
    main()
