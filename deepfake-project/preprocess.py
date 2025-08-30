# preprocess.py
import os, glob, random, argparse
from pathlib import Path
from tqdm import tqdm
import cv2

# preprocess.py (đoạn đầu)
METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
ALIASES = {"DeepFakeDetection": "Deepfakes"}  # gộp DFD → Deepfakes

def extract_frames(video_path, out_dir, frame_every=10):
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % frame_every == 0:
            out_path = out_dir / f"{Path(video_path).stem}_{saved}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        count += 1
    cap.release()
    return saved

def collect_videos(data_root):
    data_root = Path(data_root)
    real = sorted([str(p) for p in data_root.joinpath("original").rglob("*.mp4")])
    fake_by_method = {m: [] for m in METHODS}
    # nạp cả thư mục alias nếu còn tồn tại
    for m in list(METHODS) + list(ALIASES.keys()):
        dst = ALIASES.get(m, m)
        vids = sorted([str(p) for p in data_root.joinpath(m).rglob("*.mp4")])
        fake_by_method[dst].extend(vids)
    return real, fake_by_method

def split_list(lst, val_split):
    n = int(len(lst) * (1.0 - val_split))
    return lst[:n], lst[n:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/videos")
    ap.add_argument("--out_root", type=str,  default="data/processed/faces")
    ap.add_argument("--frame_every", type=int, default=10, help="lấy 1 frame mỗi N frames")
    ap.add_argument("--val_split", type=float, default=0.2)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    # --- Xoá 2 thư mục train/val cũ nếu tồn tại ---
    for split in ["train", "val"]:
        d = out_root / split
        if d.exists():
            print(f"⚠️  Xoá thư mục cũ: {d}")
            for p in d.rglob("*"):
                if p.is_file(): p.unlink()
            for p in sorted(d.glob("**/*"), reverse=True):
                if p.is_dir():
                    try: p.rmdir()
                    except Exception: pass

    real_videos, fake_by_method = collect_videos(args.data_root)
    print(f"🎞️  Found {len(real_videos)} real videos")
    for m in METHODS:
        print(f"    - {m}: {len(fake_by_method[m])} videos")

    # --- REAL ---
    random.shuffle(real_videos)
    tr, va = split_list(real_videos, args.val_split)
    totals = {"train_real":0, "val_real":0}
    for v in tqdm(tr, desc="real-train"):
        totals["train_real"] += extract_frames(v, out_root/"train"/"real", args.frame_every)
    for v in tqdm(va, desc="real-val"):
        totals["val_real"] += extract_frames(v, out_root/"val"/"real", args.frame_every)

    # --- FAKE BY METHOD ---
    totals_fake = {}
    for m in METHODS:
        vids = fake_by_method[m]
        random.shuffle(vids)
        tr, va = split_list(vids, args.val_split)
        totals_fake[m] = {"train":0,"val":0}
        for v in tqdm(tr, desc=f"{m}-train"):
            totals_fake[m]["train"] += extract_frames(v, out_root/"train"/"fake"/m, args.frame_every)
        for v in tqdm(va, desc=f"{m}-val"):
            totals_fake[m]["val"]   += extract_frames(v, out_root/"val"/"fake"/m, args.frame_every)

    print("✅ Preprocessing done!")
    print(f"[REAL] train: {totals['train_real']} | val: {totals['val_real']}")
    for m in METHODS:
        print(f"[FAKE/{m}] train: {totals_fake[m]['train']} | val: {totals_fake[m]['val']}")

if __name__ == "__main__":
    main()