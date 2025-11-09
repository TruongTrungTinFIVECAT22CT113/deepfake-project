# preprocess_balanced.py — Face-only, Auto-Anchor, Skip-on-Miss (no wrong person)
# Dependencies: opencv-python, mediapipe, tqdm, numpy
# Output:
#   out_root/processed_multi/face/{train,val,test}/{real|fake}/<Name>/<video-id>/000000.jpg ...

import argparse, json, time, random, hashlib
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ====== REAL dataset layout (yours) ======
REAL_ROOT       = "Original"
REAL_SUB_FFPP   = "Faceforensics"
REAL_SUB_CELEB  = "Celeb-DF-v2"
REAL_SUB_TIKTOK = "Tiktok-DF"

# ----------------- utils -----------------
def sha256_file(p: Path, chunk=1024*1024) -> str:
    h = hashlib.sha256()
    if p.exists():
        with open(p, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b: break
                h.update(b)
    return h.hexdigest()

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS

def probe_video(path: Path) -> Tuple[int,float,float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return 0, 0.0, 0.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    dur = (n / fps) if (fps > 0 and n > 0) else 0.0
    cap.release()
    return n, fps, dur

def max_nonoverlap_segments(dur, seg_len, margin):
    if dur <= 0 or seg_len <= 0: return 0
    return int(np.floor((dur + margin) / (seg_len + margin)))

def choose_segment_starts(dur, k, seg_len, strategy, margin, rng):
    if dur <= 0 or seg_len <= 0 or k <= 0: return []
    k = min(k, max_nonoverlap_segments(dur, seg_len, margin))
    if k <= 0: return []
    max_start = max(0.0, dur - seg_len)
    if k == 1:
        return [min(max_start, max(0.0, (dur - seg_len) / 2.0))]
    if strategy == "uniform":
        spacing = (dur - seg_len) / (k - 1)
        jitter  = min(margin / 2.0, spacing / 4.0)
        starts  = []
        for i in range(k):
            s = i * spacing
            s = max(0.0, min(max_start, s + (0 if jitter <= 0 else rng.uniform(-jitter, jitter))))
            starts.append(float(s))
        return sorted(starts)
    # random non-overlap
    starts, attempts = [], 0
    while len(starts) < k and attempts < 2000:
        s = rng.uniform(0.0, max_start)
        if all(abs(s - t) >= (seg_len + margin) for t in starts):
            starts.append(s)
        attempts += 1
    if len(starts) < k:
        return choose_segment_starts(dur, k, seg_len, "uniform", margin, rng)
    return sorted(starts)

# ----------------- Face detector (MediaPipe) + partial-face fallback -----------------
_fd = None
_fm = None

def _get_face_det():
    global _fd
    if _fd is None:
        _fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    return _fd

def _get_face_mesh():
    global _fm
    if _fm is None:
        _fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    return _fm

def detect_faces(frame) -> List[Tuple[int,int,int,int,float]]:
    """
    Trả về list (x,y,w,h,score). Ưu tiên FaceDetection; nếu rỗng -> FaceMesh fallback (partial-face tolerant).
    """
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1) face detection
    fd = _get_face_det()
    res = fd.process(rgb)
    boxes = []
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
    if boxes:
        return boxes

    # 2) FaceMesh fallback
    fm = _get_face_mesh()
    res2 = fm.process(rgb)
    if not res2.multi_face_landmarks:
        return []
    boxes2 = []
    for lm in res2.multi_face_landmarks:
        xs = [int(pt.x * W) for pt in lm.landmark]
        ys = [int(pt.y * H) for pt in lm.landmark]
        if not xs or not ys: 
            continue
        x1, x2 = max(0, min(xs)), min(W-1, max(xs))
        y1, y2 = max(0, min(ys)), min(H-1, max(ys))
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        boxes2.append((x1, y1, w, h, 0.4))
    return boxes2

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
    - Lọc mặt quá nhỏ; nếu có roi=(xmin_frac,xmax_frac) thì lọc theo tâm bbox.
    - Nếu có last_bbox và iou>=iou_thr -> bám.
    - Nếu không -> chọn box lớn nhất.
    """
    if hasattr(image_shape, "shape"):
        H, W = image_shape.shape[:2]
    else:
        H, W = image_shape[:2]

    if not boxes: return None

    min_side = min(H, W)
    boxes2 = [(x,y,w,h,s) for (x,y,w,h,s) in boxes if max(w,h) >= min_frac*min_side]
    if not boxes2: boxes2 = boxes

    if roi is not None:
        xmin_px = int(max(0.0, min(1.0, roi[0])) * W)
        xmax_px = int(max(0.0, min(1.0, roi[1])) * W)
        def in_roi(b):
            x,y,w,h,_ = b
            cx = x + w//2
            return xmin_px <= cx <= xmax_px
        filt = [b for b in boxes2 if in_roi(b)]
        if filt: boxes2 = filt
        else:
            # không có ai trong ROI → coi như miss hoàn toàn
            return None

    if last_bbox is not None:
        best = None; best_i = -1
        for (x,y,w,h,s) in boxes2:
            ii = iou(last_bbox, (x,y,w,h))
            if ii > best_i:
                best_i = ii; best = (x,y,w,h,s)
        if best_i >= iou_thr:
            return best[:4]

    x,y,w,h,s = max(boxes2, key=lambda b: b[2]*b[3])
    return (x,y,w,h)

def crop_face(frame, last_bbox=None, img_size=384, roi=None, skip_on_miss=True):
    """
    Trả về (crop, new_bbox, wrote_flag).
    - Nếu không thấy anchor và skip_on_miss=True → (None, last_bbox, False): KHÔNG ghi frame.
    - Nếu không thấy nhưng đã có last_bbox và không bắt trong ROI → cũng bỏ qua.
    - Chỉ center-crop khi không dùng ROI (trường hợp rất hiếm vì ta luôn auto_lock).
    """
    H, W = frame.shape[:2]
    boxes = detect_faces(frame)
    bb = pick_face(boxes, last_bbox, frame, min_frac=0.04, iou_thr=0.15, roi=roi)

    if bb is None:
        if skip_on_miss:
            return None, last_bbox, False
        # fallback an toàn (ít dùng): center-crop
        sz = int(min(H, W) * 0.7)
        cy, cx = H//2, W//2
        y1 = max(0, cy - sz//2); y2 = min(H, cy + sz//2)
        x1 = max(0, cx - sz//2); x2 = min(W, cx + sz//2)
        face = frame[y1:y2, x1:x2]
        return cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA), None, True

    x, y, w, h = bb
    pad = int(0.22 * max(w, h))
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
    face = frame[y1:y2, x1:x2]
    return cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA), (x, y, w, h), True

def write_frame(out_dir: Path, idx: int, frame):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame)

def extract_segments(path: Path, out_dir: Path, fps: float, starts: List[float],
                     seg_len: float, img_size: int, auto_lock=True, skip_on_miss=True) -> int:
    """
    Auto-anchor: tìm anchor ở đầu video, tạo ROI quanh anchor, bám suốt video.
    Khi không thấy anchor → SKIP frame (không ghi gì) để tránh lẫn người khác.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return 0
    frames_written = 0
    f_per_seg = max(1, int(round(seg_len * max(fps, 1e-6))))

    # --- Auto-anchor ---
    roi = None
    if auto_lock:
        t0 = 0.15
        probes = 6
        warmup_sec = 0.5
        step = max(warmup_sec / max(probes-1,1), 0.05)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        best, best_score = None, -1e9
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
        if best is not None:
            x,y,w,h,s = best
            anchor_cx = (x + w/2) / max(W,1)
            tol = 0.25
            xmin = max(0.0, anchor_cx - tol - 0.05)
            xmax = min(1.0, anchor_cx + tol + 0.05)
            roi = (xmin, xmax)

    # --- Extract ---
    last_bbox = None
    for s in starts:
        for i in range(f_per_seg):
            t = s + i / max(fps, 1e-6)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok: break

            face, last_bbox, wrote = crop_face(
                frame, last_bbox=last_bbox, img_size=img_size,
                roi=roi, skip_on_miss=skip_on_miss
            )
            if not wrote:
                continue  # SKIP: không ghi frame nếu không bắt được anchor

            write_frame(out_dir, frames_written, face)
            frames_written += 1

    cap.release()
    return frames_written

# ----------------- split helpers -----------------
def split_for_ffpp(idx: int) -> str:
    if 0 <= idx <= 699: return "train"
    if 700 <= idx <= 899: return "val"
    if 900 <= idx <= 999: return "test"
    return "train"

def split_mod_70_20_10(num: int) -> str:
    r = num % 10
    return "train" if r < 7 else ("val" if r < 9 else "test")

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("Preprocess face-only (auto-locked anchor, skip-on-miss)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root",  required=True)
    # basic controls
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seg_len", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--strategy", choices=["uniform","random"], default="uniform")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--frames_per_face", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    # simplified switches
    ap.add_argument("--auto_lock", action="store_true", default=True)
    ap.add_argument("--no_skip_on_miss", action="store_true", help="Nếu bật, sẽ không skip khi mất anchor (không khuyến nghị)")

    args = ap.parse_args()
    skip_on_miss = not args.no_skip_on_miss

    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    out_root  = Path(args.out_root) / "processed_multi" / "face"
    ensure_dir(out_root)
    for sp in ["train","val","test"]:
        ensure_dir(out_root / sp)

    # manifest
    manifest = out_root.parent / "manifest_videos.csv"
    rows = ["split,branch,kind,name,source_index,rel_path,bytes,sha256"]

    def add_row(split: str, kind: str, name: str, src_idx: int, rel: Path):
        p = data_root / rel
        size = p.stat().st_size if p.exists() else 0
        rows.append(f"{split},face,{kind},{name},{src_idx},{rel.as_posix()},{size},{sha256_file(p)}")

    # REAL
    real_sets = [
        ("Faceforensics", Path(REAL_ROOT) / REAL_SUB_FFPP,   split_for_ffpp),
        ("Celeb-DF-v2",   Path(REAL_ROOT) / REAL_SUB_CELEB,  split_mod_70_20_10),
        ("Tiktok-DF",     Path(REAL_ROOT) / REAL_SUB_TIKTOK, split_mod_70_20_10),
    ]
    for ds_name, rel_dir, split_fn in real_sets:
        abs_dir = data_root / rel_dir
        if not abs_dir.exists():
            print(f"[WARN] Real set missing: {abs_dir}")
            continue
        for vid in sorted(abs_dir.iterdir()):
            if not vid.is_file() or not is_video(vid): continue
            stem = vid.stem
            try: idx = int(stem)
            except: idx = abs(hash(stem)) % 10000
            sp = split_fn(idx)
            add_row(sp, "real", ds_name, idx, rel_dir / vid.name)

    # FAKE
    fake_method_dirs = [p for p in data_root.iterdir() if p.is_dir() and p.name != REAL_ROOT]
    for mdir in sorted(fake_method_dirs, key=lambda p: p.name.lower()):
        mname = mdir.name
        vids = [v for v in sorted(mdir.iterdir(), key=lambda p: p.name) if v.is_file() and is_video(v)]
        for v in vids:
            stem = v.stem
            try:
                vid_idx = int(stem); sp = split_for_ffpp(vid_idx)
            except:
                vid_idx = abs(hash(stem)) % 10000; sp = split_mod_70_20_10(vid_idx)
            add_row(sp, "fake", mname, vid_idx, Path(mname) / v.name)

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(rows), encoding="utf-8")
    print(f"✅ Wrote manifest: {manifest}")

    # iterate manifest
    def iter_manifest(kind: str, split: str):
        with open(manifest, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                sp, branch, k, name, src_idx, rel, *_ = line.strip().split(",", 7)
                if sp == split and branch == "face" and k == kind:
                    yield name, int(src_idx), Path(rel)

    totals = {"train":0,"val":0,"test":0}
    t0=time.time()

    for split in ["train","val","test"]:
        # REAL
        for ds_name, src_idx, rel in tqdm(list(iter_manifest("real", split)), desc=f"[FACE | {split} | REAL]"):
            in_path = data_root / rel
            vid_id  = rel.stem
            out_dir = out_root / split / "real" / ds_name / vid_id
            _, _, dur = probe_video(in_path)
            frames_target = args.frames_per_face
            frames_per_seg = max(1, int(round(args.seg_len * max(args.fps, 1e-6))))
            k = max(1, frames_target // frames_per_seg)
            k = min(k, max_nonoverlap_segments(dur, args.seg_len, args.margin))
            starts = choose_segment_starts(
                dur, k, args.seg_len, args.strategy, args.margin,
                random.Random((hash(rel.as_posix()) ^ 0xFACEFACE) & 0xffffffff)
            )
            c = extract_segments(in_path, out_dir, args.fps, starts, args.seg_len,
                                 args.img_size, auto_lock=args.auto_lock, skip_on_miss=skip_on_miss)
            totals[split]+=c

        # FAKE
        for mname, src_idx, rel in tqdm(list(iter_manifest("fake", split)), desc=f"[FACE | {split} | FAKE]"):
            in_path = data_root / rel
            vid_id  = rel.stem
            out_dir = out_root / split / "fake" / mname / vid_id
            _, _, dur = probe_video(in_path)
            frames_target = args.frames_per_face
            frames_per_seg = max(1, int(round(args.seg_len * max(args.fps, 1e-6))))
            k = max(1, frames_target // frames_per_seg)
            k = min(k, max_nonoverlap_segments(dur, args.seg_len, args.margin))
            starts = choose_segment_starts(
                dur, k, args.seg_len, args.strategy, args.margin,
                random.Random((hash(rel.as_posix()) ^ 0xDEADBEEF) & 0xffffffff)
            )
            c = extract_segments(in_path, out_dir, args.fps, starts, args.seg_len,
                                 args.img_size, auto_lock=args.auto_lock, skip_on_miss=skip_on_miss)
            totals[split]+=c

    summary = {"args": vars(args), "totals": totals, "time_sec": time.time()-t0}
    (out_root.parent / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"⏱  Total time: {int(summary['time_sec'])} sec")

if __name__ == "__main__":
    main()
