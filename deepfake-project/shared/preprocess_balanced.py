# preprocess_balanced.py
# Multi-branch preprocessor for deepfake detection
# Output layout (dynamic, future-proof):
#   out_root/
#     processed_multi/
#       face|head|full/
#         train|val|test/
#           real_* | <FakeMethod> / <video-id> / 000000.jpg ...

import argparse, os, json, time, random, hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# -------------------------------
# Config nguồn dữ liệu (có thể giữ nguyên)
# -------------------------------
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Thư mục real chuẩn (bạn giữ như đã dùng: Original/Faceforensics|Celeb-DF-v2|Tiktok-DF)
REAL_ROOT = "Original"
REAL_SUB_FFPP  = "Faceforensics"   # file 000..999.mp4
REAL_SUB_CELEB = "Celeb-DF-v2"     # file 1000..1889.mp4 (gợi ý đặt tên vậy)
REAL_SUB_TIK   = "Tiktok-DF"       # file 1890..1999.mp4

# Bộ phương pháp "gợi ý" theo nhánh (không khóa cứng — chỉ để log; script vẫn quét động)
SUG_FACE = {"Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures"}
SUG_HEAD = {"Audio2Animation"}
SUG_FULL = {"Video2VideoID"}

# -------------------------------
# Utils
# -------------------------------

def sha256_file(p: Path, chunk=1024*1024) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def ensure_clean_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS

def probe_video(path: Path) -> Tuple[int,float,float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0.0, 0.0
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
        spacing = (dur - seg_len) / (k - 1) if k > 1 else 0.0
        jitter  = min(margin / 2.0, spacing / 4.0 if k > 1 else 0.0)
        starts  = []
        for i in range(k):
            s = i * spacing
            if jitter > 0:
                s = max(0.0, min(max_start, s + rng.uniform(-jitter, jitter)))
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

# -------- face/head crop ----------
_haar = None
def _get_haar():
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _haar

def detect_face_bbox(frame):
    haar = _get_haar()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0: return None
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    return (x, y, w, h)

def crop_face(frame, pad_ratio=0.2):
    bb = detect_face_bbox(frame)
    if bb is None: return frame
    x, y, w, h = bb
    pad = int(pad_ratio * max(w, h))
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad); y2 = min(frame.shape[0], y + h + pad)
    return frame[y1:y2, x1:x2]

def crop_head(frame, scale=1.7):
    bb = detect_face_bbox(frame)
    if bb is None: return frame
    x, y, w, h = bb
    cx = x + w / 2.0; cy = y + h / 2.0
    side = int(max(w, h) * scale)
    x1 = int(cx - side / 2); y1 = int(cy - side / 2)
    x2 = x1 + side; y2 = y1 + side
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    return frame[y1:y2, x1:x2]

def resize_square(frame, size):
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

def write_frame(out_dir: Path, idx: int, frame):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame)

def extract_segments(path: Path, out_dir: Path, fps: float, starts: List[float],
                     seg_len: float, crop_mode: str, img_size: int, head_scale: float) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return 0
    frames_written = 0
    f_per_seg = max(1, int(round(seg_len * max(fps, 1e-6))))
    for s in starts:
        for i in range(f_per_seg):
            t = s + i / max(fps, 1e-6)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok: break
            if crop_mode == "face":
                frame = crop_face(frame)
            elif crop_mode == "head":
                frame = crop_head(frame, scale=head_scale)
            # full: không crop
            frame = resize_square(frame, img_size)
            write_frame(out_dir, frames_written, frame)
            frames_written += 1
    cap.release()
    return frames_written

# -------------------------------
# Split rules
# -------------------------------
def split_for_ffpp(idx: int) -> str:
    # 000..699 -> train, 700..899 -> val, 900..999 -> test
    if 0 <= idx <= 699: return "train"
    if 700 <= idx <= 899: return "val"
    if 900 <= idx <= 999: return "test"
    return "train"  # fallback

def split_mod_70_20_10(num: int) -> str:
    r = num % 10
    return "train" if r < 7 else ("val" if r < 9 else "test")

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser("Preprocess multi-branch (face/head/full) — leak-safe splits")
    ap.add_argument("--data_root", required=True, help="Input videos root")
    ap.add_argument("--out_root",  required=True, help="Where to write processed_multi")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seg_len", type=float, default=0.5, help="seconds per segment")
    ap.add_argument("--margin", type=float, default=0.1, help="non-overlap margin between segments (sec)")
    ap.add_argument("--strategy", choices=["uniform","random"], default="uniform")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames_per_face", type=int, default=64)
    ap.add_argument("--frames_per_head", type=int, default=64)
    ap.add_argument("--frames_per_full", type=int, default=64)
    ap.add_argument("--head_scale", type=float, default=1.7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    out_root  = Path(args.out_root) / "processed_multi"
    ensure_clean_dir(out_root)  # không xóa mạnh; chỉ đảm bảo tồn tại

    # Tạo sẵn cây thư mục
    for br in ["face","head","full"]:
        for sp in ["train","val","test"]:
            (out_root / br / sp).mkdir(parents=True, exist_ok=True)

    # Manifest
    manifest = out_root / "manifest_videos.csv"
    rows = ["split,branch,submethod,source_dataset,source_index,rel_path,bytes,sha256"]

    def add_row(split: str, branch: str, submethod: str, src_ds: str, src_idx: int, rel: Path):
        p = data_root / rel
        size = p.stat().st_size if p.exists() else 0
        rows.append(f"{split},{branch},{submethod},{src_ds},{src_idx},{rel.as_posix()},{size},{sha256_file(p) if p.exists() else ''}")

    # -------------------------------
    # 1) Quét nguồn REAL (3 tập)
    # -------------------------------
    real_sets = [
        ("FF++",     Path(REAL_ROOT) / REAL_SUB_FFPP,  split_for_ffpp),
        ("CelebDFv2",Path(REAL_ROOT) / REAL_SUB_CELEB, split_mod_70_20_10),
        ("TikTokDF", Path(REAL_ROOT) / REAL_SUB_TIK,   split_mod_70_20_10),
    ]
    for name, rel_dir, split_fn in real_sets:
        abs_dir = data_root / rel_dir
        if not abs_dir.exists(): 
            print(f"[WARN] Real set missing: {abs_dir}")
            continue
        for vid in sorted(abs_dir.iterdir()):
            if not vid.is_file() or not is_video(vid): continue
            stem = vid.stem
            try:
                idx = int(stem)
            except:
                # nếu tên không phải số, dùng hash để modulo
                idx = abs(hash(stem)) % 10000
            sp = split_fn(idx)
            for branch, real_name in (("face","real_face"), ("head","real_head"), ("full","real_full")):
                add_row(sp, branch, real_name, name, idx, rel_dir / vid.name)

    # -------------------------------
    # 2) Quét phương pháp FAKE theo nhánh (động)
    #   data_root / <MethodName> / *.mp4  (đặt ở root, ngang với Original/)
    #   -> map method vào branch theo quy tắc:
    #      - nếu tên nằm trong SUG_FACE/HEAD/FULL -> theo đó
    #      - nếu không, gán vào 'face' mặc định (bạn có thể đổi logic tại đây)
    # -------------------------------
    # Thu thập danh sách method (dir ở root, không phải 'Original')
    all_method_dirs = [p for p in data_root.iterdir() if p.is_dir() and p.name != REAL_ROOT]
    for mdir in sorted(all_method_dirs, key=lambda p: p.name.lower()):
        mname = mdir.name
        # chọn branch
        if   mname in SUG_FACE: branch = "face"
        elif mname in SUG_HEAD: branch = "head"
        elif mname in SUG_FULL: branch = "full"
        else:
            # Mặc định tạm: coi là face-method (bạn có thể chỉnh rule này)
            branch = "face"
        # file .mp4 trong thư mục đó coi là FF++-like: 000..999.mp4 để giữ split theo index;
        # không thì ta fallback modulo 70/20/10
        vids = [v for v in sorted(mdir.iterdir(), key=lambda p: p.name) if v.is_file() and is_video(v)]
        for v in vids:
            stem = v.stem
            sp_by = None
            try:
                vid_idx = int(stem)
                sp_by = split_for_ffpp(vid_idx)
            except:
                vid_idx = abs(hash(stem)) % 10000
                sp_by = split_mod_70_20_10(vid_idx)
            add_row(sp_by, branch, mname, "FAKE", vid_idx, Path(mname) / v.name)

    # Ghi manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(rows), encoding="utf-8")
    print(f"✅ Wrote manifest: {manifest}")

    # -------------------------------
    # 3) Trích khung theo manifest
    # -------------------------------
    def iter_manifest(branch: str, split: str):
        with open(manifest, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                sp,b,subm,src_ds,src_idx,rel,_,_ = line.strip().split(",", 7)
                if sp == split and b == branch:
                    yield subm, int(src_idx), Path(rel)

    branch_cfg = {
        "face": {"crop": "face", "frames": args.frames_per_face},
        "head": {"crop": "head", "frames": args.frames_per_head},
        "full": {"crop": "full", "frames": args.frames_per_full},
    }

    totals = { "face": {"train":0,"val":0,"test":0},
               "head": {"train":0,"val":0,"test":0},
               "full": {"train":0,"val":0,"test":0} }

    t0 = time.time()
    for branch in ["face","head","full"]:
        cfg = branch_cfg[branch]
        for split in ["train","val","test"]:
            items = list(iter_manifest(branch, split))
            for subm, src_idx, rel in tqdm(items, desc=f"[{branch.upper()} | {split}]"):
                in_path = data_root / rel
                vid_id  = rel.stem

                # >>> chọn out_dir đúng real/fake (tầng fake_{branch})
                if subm.startswith("real_"):
                    out_dir = out_root / branch / split / subm / vid_id
                else:
                    fake_parent = f"fake_{branch}"
                    out_dir = out_root / branch / split / fake_parent / subm / vid_id

                # ước lượng số segment để đủ frames
                _, _, dur = probe_video(in_path)
                frames_per_video = cfg["frames"]
                frames_per_seg   = max(1, int(round(args.seg_len * max(args.fps, 1e-6))))
                k = max(1, frames_per_video // frames_per_seg)
                k = min(k, max_nonoverlap_segments(dur, args.seg_len, args.margin))  # không chồng lắp

                # chọn vị trí bắt đầu
                starts = choose_segment_starts(
                    dur, k, args.seg_len, args.strategy, args.margin,
                    random.Random((hash(rel.as_posix()) ^ hash(subm)) & 0xffffffff)
                )

                # trích & ghi frame
                count = extract_segments(
                    in_path, out_dir, args.fps, starts, args.seg_len,
                    cfg["crop"], args.img_size, args.head_scale
                )
                totals[branch][split] += count

    # -------------------------------
    # 4) Ghi tóm tắt
    # -------------------------------
    summary = {
        "args": vars(args),
        "totals": totals,
        "time_sec": time.time() - t0
    }
    (out_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"⏱  Total time: {int(summary['time_sec'])} sec")

if __name__ == "__main__":
    main()
