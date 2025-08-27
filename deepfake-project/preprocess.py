import cv2
import os
import glob
import random
from pathlib import Path
from tqdm import tqdm

# Input data
DATA_ROOT = "data/FaceForensics_C23"
OUT_ROOT = "data/processed/faces"
FRAME_INTERVAL = 10  # lấy 1 frame mỗi 10 frames
VAL_SPLIT = 0.2

def extract_frames(video_path, out_dir, label):
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % FRAME_INTERVAL == 0:
            out_path = os.path.join(out_dir, f"{Path(video_path).stem}_{saved}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        count += 1
    cap.release()
    return saved

def main():
    real_videos = glob.glob(os.path.join(DATA_ROOT, "original", "**", "*.mp4"), recursive=True)
    fake_videos = []
    for sub in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "DeepFakeDetection"]:
        fake_videos += glob.glob(os.path.join(DATA_ROOT, sub, "**", "*.mp4"), recursive=True)

    print(f"Found {len(real_videos)} real and {len(fake_videos)} fake videos.")

    data_map = {"real": real_videos, "fake": fake_videos}

    for label, videos in data_map.items():
        random.shuffle(videos)
        split_idx = int(len(videos) * (1 - VAL_SPLIT))
        train_videos, val_videos = videos[:split_idx], videos[split_idx:]

        for split, split_videos in [("train", train_videos), ("val", val_videos)]:
            out_dir = os.path.join(OUT_ROOT, split, label)
            os.makedirs(out_dir, exist_ok=True)
            total_frames = 0
            for vid in tqdm(split_videos, desc=f"{label}-{split}"):
                total_frames += extract_frames(vid, out_dir, label)
            print(f"[{label}-{split}] Extracted {total_frames} frames → {out_dir}")

    print("✅ Preprocessing done!")

if __name__ == "__main__":
    main()