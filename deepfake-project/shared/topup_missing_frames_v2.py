# topup_missing_frames_v2.py — Append frames without overwrite; wider ROI; auto-anchor; skip-on-miss
# Usage example:
#   python topup_missing_frames_v2.py ^
#     --processed_root H:\deepfake-project\deepfake-project\data\processed_multi\face ^
#     --videos_root    H:\deepfake-project\deepfake-project\data\videos_norm ^
#     --min_frames 90 --fps 25 --seg_len 3.9 --img_size 384 --roi_tol 0.30

from __future__ import annotations
import argparse, math, random, json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import mediapipe as mp

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def count_frames_in_dir(vid_dir: Path) -> int:
    return sum(1 for f in vid_dir.iterdir() if is_img(f)) if vid_dir.exists() else 0

def list_video_dirs(processed_root: Path) -> List[Tuple[str,str,str,Path]]:
    out = []
    for split in ["train","val","test"]:
        for kind in ["real","fake"]:
            base = processed_root / split / kind
            if not base.exists(): continue
            for group in sorted([d for d in base.iterdir() if d.is_dir()]):
                for vid_dir in sorted([d for d in group.iterdir() if d.is_dir()]):
                    out.append((split, kind, group.name, vid_dir))
    return out

def load_manifest_map(manifest_path: Path, videos_root: Path) -> Dict[Tuple[str,str,str,str], Path]:
    m: Dict[Tuple[str,str,str,str], Path] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",", 7)
            if len(parts) < 6: continue
            split, branch, kind, name, src_idx, rel_path = parts[:6]
            if branch != "face": continue
            rel = Path(rel_path)
            src = (videos_root / rel).resolve()
            key = (split, kind, name, rel.stem)
            m[key] = src
    return m

# ------------ Minimal video probe ------------
def probe_video(path: Path) -> Tuple[int,float,float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return 0, 0.0, 0.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    dur = (n / fps) if (fps > 0 and n > 0) else 0.0
    cap.release()
    return n, fps, dur

# ------------ Mediapipe detectors ------------
_fd = None
_fm = None
def _get_face_det():
    global _fd
    if _fd is None:
        _fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return _fd

def _get_face_mesh():
    global _fm
    if _fm is None:
        _fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2, refine_landmarks=False,
            min_detection_confidence=0.4, min_tracking_confidence=0.4
        )
    return _fm

def detect_faces(frame) -> List[Tuple[int,int,int,int,float]]:
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = []
    fd = _get_face_det()
    res = fd.process(rgb)
    if res.detections:
        for d in res.detections:
            rb = d.location_data.relative_bounding_box
            x = int(max(0, rb.xmin * W)); y = int(max(0, rb.ymin * H))
            w = int(max(1,  rb.width * W)); h = int(max(1,  rb.height * H))
            x = min(x, W-1); y = min(y, H-1)
            if x + w > W: w = W - x
            if y + h > H: h = H - y
            score = float(d.score[0]) if d.score else 1.0
            boxes.append((x,y,w,h,score))
    if boxes: return boxes

    # fallback partial-face
    fm = _get_face_mesh()
    res2 = fm.process(rgb)
    if not res2.multi_face_landmarks:
        return []
    for lm in res2.multi_face_landmarks:
        xs = [int(pt.x * W) for pt in lm.landmark]
        ys = [int(pt.y * H) for pt in lm.landmark]
        if not xs or not ys: continue
        x1, x2 = max(0, min(xs)), min(W-1, max(xs))
        y1, y2 = max(0, min(ys)), min(H-1, max(ys))
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        boxes.append((x1, y1, w, h, 0.4))
    return boxes

def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0: return 0.0
    union = aw*ah + bw*bh - inter
    return inter / max(union, 1)

def pick_face(boxes, last_bbox, image_shape, min_frac=0.04, iou_thr=0.15, roi=None):
    """
    image_shape: tuple như (H, W) hoặc (H, W, C)
    """
    # Chuẩn hóa H, W
    if isinstance(image_shape, tuple) and len(image_shape) >= 2:
        H, W = int(image_shape[0]), int(image_shape[1])
    else:
        # fallback an toàn
        try:
            H, W = int(image_shape.shape[0]), int(image_shape.shape[1])
        except Exception:
            raise ValueError(f"image_shape phải là tuple (H, W[, C]) hoặc ndarray.shape, got: {type(image_shape)}")

    if not boxes:
        return None

    min_side = min(H, W)
    boxes2 = [(x, y, w, h, s) for (x, y, w, h, s) in boxes if max(w, h) >= min_frac * min_side]
    if not boxes2:
        boxes2 = boxes

    if roi is not None:
        xmin_px = int(np.clip(roi[0], 0, 1) * W)
        xmax_px = int(np.clip(roi[1], 0, 1) * W)
        def in_roi(b):
            x, y, w, h, _ = b
            cx = x + w // 2
            return xmin_px <= cx <= xmax_px
        filt = [b for b in boxes2 if in_roi(b)]
        if filt:
            boxes2 = filt
        else:
            return None

    if last_bbox is not None:
        best = None
        best_i = -1.0
        for (x, y, w, h, s) in boxes2:
            ii = iou(last_bbox, (x, y, w, h))
            if ii > best_i:
                best_i = ii
                best = (x, y, w, h, s)
        if best_i >= iou_thr:
            return best[:4]

    x, y, w, h, s = max(boxes2, key=lambda b: b[2] * b[3])
    return (x, y, w, h)


def crop_face(frame, last_bbox=None, img_size=384, roi=None, skip_on_miss=True):
    H, W = frame.shape[:2]
    boxes = detect_faces(frame)
    # ✅ truyền shape thay vì frame để pick_face không bị nhầm mảng
    bb = pick_face(boxes, last_bbox, frame.shape, min_frac=0.04, iou_thr=0.15, roi=roi)
    if bb is None:
        if skip_on_miss:
            return None, last_bbox, False
        # fallback center crop
        sz = int(min(H, W) * 0.7)
        cy, cx = H // 2, W // 2
        y1 = max(0, cy - sz // 2); y2 = min(H, cy + sz // 2)
        x1 = max(0, cx - sz // 2); x2 = min(W, cx + sz // 2)
        face = frame[y1:y2, x1:x2]
        return cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA), None, True

    x, y, w, h = bb
    pad = int(0.22 * max(w, h))
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
    face = frame[y1:y2, x1:x2]
    return cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA), (x, y, w, h), True


def write_frame_append(out_dir: Path, idx: int, frame):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame)

def auto_anchor_roi(cap: cv2.VideoCapture, roi_tol: float) -> Optional[Tuple[float,float]]:
    # giống preprocess: chọn anchor quanh tâm + cho phép nới roi_tol
    t0 = 0.15; probes = 6; warmup_sec = 0.5
    step = max(warmup_sec / max(probes-1,1), 0.05)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    best=None; best_score=-1e9
    cx, cy = W/2, H/2
    for k in range(probes):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + k*step) * 1000.0)
        ok, f0 = cap.read()
        if not ok: continue
        bxs = detect_faces(f0)
        if not bxs: continue
        def score(b):
            x,y,w,h,s = b
            cxx = x + w/2; cyy = y + h/2
            return (w*h) - 0.15*((cxx-cx)**2 + (cyy-cy)**2)
        cand = max(bxs, key=score)
        sc = score(cand)
        if sc > best_score:
            best_score, best = sc, cand
    if best is None: return None
    x,y,w,h,s = best
    anchor_cx = (x + w/2) / max(W,1)
    tol = float(roi_tol)
    xmin = max(0.0, anchor_cx - tol - 0.05)
    xmax = min(1.0, anchor_cx + tol + 0.05)
    return (xmin, xmax)

def append_segments(path: Path, out_dir: Path, base_idx: int, fps: float, starts: List[float],
                    seg_len: float, img_size: int, roi_tol: float, skip_on_miss: bool) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return 0
    wrote = 0
    # auto-anchor ROI
    roi = auto_anchor_roi(cap, roi_tol=roi_tol)
    last_bbox = None
    f_per_seg = max(1, int(round(seg_len * max(fps, 1e-6))))
    for s in starts:
        for i in range(f_per_seg):
            t = s + i / max(fps, 1e-6)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok: break
            face, last_bbox, okw = crop_face(
                frame, last_bbox=last_bbox, img_size=img_size, roi=roi, skip_on_miss=skip_on_miss
            )
            if not okw: continue
            write_frame_append(out_dir, base_idx + wrote, face)
            wrote += 1
    cap.release()
    return wrote

def choose_starts(dur: float, need_segments: int, seg_len: float, rng: random.Random) -> List[float]:
    if dur <= 0 or need_segments <= 0: return []
    max_start = max(0.0, dur - seg_len)
    if need_segments == 1:
        return [min(max_start, max(0.0, (dur - seg_len)/2.0))]
    spacing = (dur - seg_len) / max(need_segments - 1, 1)
    jitter = min(0.2, spacing/4.0)
    starts = []
    for i in range(need_segments):
        s = i * spacing
        s = max(0.0, min(max_start, s + (0 if jitter<=0 else rng.uniform(-jitter, jitter))))
        starts.append(float(s))
    return sorted(starts)

def main():
    ap = argparse.ArgumentParser("Top-up missing frames (append, no overwrite) with wider ROI")
    ap.add_argument("--processed_root", required=True)
    ap.add_argument("--videos_root",    required=True)
    ap.add_argument("--min_frames", type=int, default=90)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seg_len", type=float, default=3.9)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--roi_tol", type=float, default=0.30, help="±ROI tolerance around anchor center (0.25–0.35 hợp lý)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no_skip_on_miss", action="store_true")
    args = ap.parse_args()

    processed_root = Path(args.processed_root)
    manifest_path  = processed_root.parent / "manifest_videos.csv"
    videos_root    = Path(args.videos_root)
    if not processed_root.exists(): raise SystemExit(f"[ERR] {processed_root} not found")
    if not manifest_path.exists():  raise SystemExit(f"[ERR] {manifest_path} not found")

    key2src = load_manifest_map(manifest_path, videos_root)
    rng = random.Random(args.seed)

    fixed = skipped = notfound = 0
    frames_added_total = 0

    print(f"🔎 Scanning: {processed_root}")
    for split, kind, group, vid_dir in list_video_dirs(processed_root):
        video_id = vid_dir.name
        n_before = count_frames_in_dir(vid_dir)
        if n_before >= args.min_frames:
            continue
        key = (split, kind, group, video_id)
        src = key2src.get(key)
        if src is None or not src.exists():
            print(f"[MISS-MAP] {split}/{kind}/{group}/{video_id} -> source missing")
            notfound += 1
            continue

        _, _, dur = probe_video(src)
        frames_per_seg = max(1, int(round(args.seg_len * max(args.fps, 1e-6))))
        need = args.min_frames - n_before
        need_segments = max(1, math.ceil(need / max(frames_per_seg,1)))
        starts = choose_starts(dur, need_segments, args.seg_len, rng)

        print(f"[TOPUP] {split}/{kind}/{group}/{video_id}: {n_before} -> need +{need} "
              f"(~{need_segments} seg @ {args.seg_len}s, roi_tol={args.roi_tol}) from {src.name}")

        wrote = append_segments(
            path=src, out_dir=vid_dir, base_idx=n_before, fps=args.fps, starts=starts,
            seg_len=args.seg_len, img_size=args.img_size, roi_tol=args.roi_tol,
            skip_on_miss=(not args.no_skip_on_miss)
        )
        n_after = count_frames_in_dir(vid_dir)
        delta = max(0, n_after - n_before)
        frames_added_total += delta

        if n_after >= args.min_frames:
            fixed += 1
        else:
            skipped += 1
            print(f"  ↳ Still short ({n_after} < {args.min_frames}). "
                  f"Thử tăng --seg_len lên 4.0 hoặc --roi_tol 0.32–0.35.")

    print("\n✅ Done top-up (append mode).")
    print(f"  Fixed videos (>= min_frames): {fixed}")
    print(f"  Still short: {skipped}")
    print(f"  Missing mapping/source: {notfound}")
    print(f"  Frames added total: {frames_added_total}")
    print(f"  Min target per video: {args.min_frames}")

if __name__ == "__main__":
    main()
