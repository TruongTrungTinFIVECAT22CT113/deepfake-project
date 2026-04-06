# copy_val_videos_flat.py
import argparse
from pathlib import Path
import shutil
import sys

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
REAL_ROOT_NAME = "Original"

def find_video_by_id(root: Path, rel_dir: Path, vid_id: str):
    base = root / rel_dir
    if not base.exists():
        return None
    for ext in VIDEO_EXTS:
        cand = base / f"{vid_id}{ext}"
        if cand.exists():
            return cand
    # fallback
    for p in base.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stem == vid_id:
            return p
    return None

def copy_one(src: Path, dst_path: Path, overwrite: bool):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and not overwrite:
        return "skip-exists", dst_path
    shutil.copy2(src, dst_path)
    return "copied", dst_path

def main():
    ap = argparse.ArgumentParser(description="Copy original validation videos into flat structure.")
    # Thay đổi default paths sang 'val'
    ap.add_argument("--project_root", default=r"H:\deepfake-project\deepfake-project")
    ap.add_argument("--processed_face_val", default=r"H:\deepfake-project\deepfake-project\data\processed_multi\face\val")
    ap.add_argument("--videos_norm", default=r"H:\deepfake-project\deepfake-project\data\videos_norm")
    ap.add_argument("--out_root", default=r"H:\deepfake-project\deepfake-project\data\videos_val")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    processed_val = Path(args.processed_face_val)
    videos_norm = Path(args.videos_norm)
    out_root = Path(args.out_root)

    if not processed_val.exists():
        print(f"Error: Thư mục nguồn không tồn tại: {processed_val}")
        return

    fake_root = processed_val / "fake"
    real_root = processed_val / "real"

    total = {"fake": 0, "real": 0}
    copied = {"fake": 0, "real": 0}
    skipped = {"fake": 0, "real": 0}
    missing = {"fake": 0, "real": 0}
    missing_list = []

    # -------- FAKE --------
    if fake_root.exists():
        print("Processing FAKE validation videos...")
        for method_dir in sorted(fake_root.iterdir()):
            if not method_dir.is_dir(): continue
            method = method_dir.name
            for vid_dir in sorted(method_dir.iterdir()):
                if not vid_dir.is_dir(): continue
                vid_id = vid_dir.name
                total["fake"] += 1
                src = find_video_by_id(videos_norm, Path(method), vid_id)
                if src is None:
                    missing["fake"] += 1
                    missing_list.append(("fake", method, vid_id))
                    continue
                dst_path = out_root / "fake" / method / f"{vid_id}{src.suffix}"
                status, _ = copy_one(src, dst_path, args.overwrite)
                if status == "copied": copied["fake"] += 1
                elif status == "skip-exists": skipped["fake"] += 1

    # -------- REAL --------
    if real_root.exists():
        print("Processing REAL validation videos...")
        for ds_dir in sorted(real_root.iterdir()):
            if not ds_dir.is_dir(): continue
            dataset = ds_dir.name
            for vid_dir in sorted(ds_dir.iterdir()):
                if not vid_dir.is_dir(): continue
                vid_id = vid_dir.name
                total["real"] += 1
                src = find_video_by_id(videos_norm, Path(REAL_ROOT_NAME) / dataset, vid_id)
                if src is None:
                    missing["real"] += 1
                    missing_list.append(("real", dataset, vid_id))
                    continue
                dst_path = out_root / "real" / dataset / f"{vid_id}{src.suffix}"
                status, _ = copy_one(src, dst_path, args.overwrite)
                if status == "copied": copied["real"] += 1
                elif status == "skip-exists": skipped["real"] += 1

    # -------- SUMMARY --------
    print("\n=== VALIDATION SUMMARY ===")
    for k in ("fake", "real"):
        print(f"{k.upper():>4}: total={total[k]} | copied={copied[k]} | skip={skipped[k]} | missing={missing[k]}")
    
    if missing_list:
        print(f"\nMissing videos ({len(missing_list)} items):")
        for kind, name, vid_id in missing_list[:50]:
            print(f"- [{kind}] {name}/{vid_id}")

if __name__ == "__main__":
    main()