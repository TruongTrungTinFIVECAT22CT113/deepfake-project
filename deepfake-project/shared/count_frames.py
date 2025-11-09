# count_frames.py — for processed_multi/face/{train,val,test}/{real/<dataset>/, fake/<method>/}<video_id>/*
# Counts frames per split; lists per-dataset (real) and per-method (fake).
# Usage:
#   python count_frames.py --data_root H:\...\processed_multi

from __future__ import annotations
import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def count_images_one_level(dir_path: Path) -> int:
    """Count images directly under each subdir (assumes dir_path/<video_id>/*.jpg)."""
    if not dir_path.exists():
        return 0
    total = 0
    for vid_dir in dir_path.iterdir():
        if vid_dir.is_dir():
            total += sum(1 for f in vid_dir.iterdir() if is_image(f))
    return total

def count_real_datasets(real_root: Path) -> dict[str, int]:
    """
    Real layout: .../face/<split>/real/<DatasetName>/<video_id>/*.jpg
    Also supports flat layout: .../face/<split>/real/<video_id>/*.jpg
    """
    result: dict[str, int] = {}
    if not real_root.exists():
        return result

    subdirs = [d for d in real_root.iterdir() if d.is_dir()]
    # có thư mục tên dataset “chuẩn”?
    has_named = any((real_root / name).exists() for name in ["Faceforensics", "Celeb-DF-v2", "Tiktok-DF"])

    if has_named or subdirs:
        # Case A: có các thư mục con → coi như dạng real/<DatasetName>/
        for d in sorted(subdirs):
            cnt = count_images_one_level(d)
            if cnt > 0:
                result[d.name] = cnt
    else:
        # Case B: flat real/<video_id>/*.jpg
        flat = count_images_one_level(real_root)
        if flat > 0:
            result["__flat__"] = flat

    return result

def count_fake_methods(fake_root: Path) -> dict[str, int]:
    """Fake layout: .../face/<split>/fake/<Method>/<video_id>/*.jpg"""
    out: dict[str, int] = {}
    if not fake_root.exists():
        return out
    for mdir in sorted([p for p in fake_root.iterdir() if p.is_dir()]):
        total = count_images_one_level(mdir)
        if total > 0:
            out[mdir.name] = total
    return out

def pretty_int(n: int) -> str:
    return f"{n:,}".replace(",", "_")

def main():
    ap = argparse.ArgumentParser(description="Count frames in processed_multi/face structure")
    ap.add_argument("--data_root", required=True,
                    help="Path to processed_multi directory (contains 'face/' subdir)")
    args = ap.parse_args()

    proc = Path(args.data_root)
    face_root = proc / "face"
    if not face_root.exists():
        raise SystemExit(f"[ERR] Not found: {face_root}")

    splits = ["train", "val", "test"]
    grand_total = 0

    print(f"📊 Frame counts per branch/split (root={face_root})\n")

    for sp in splits:
        split_dir = face_root / sp
        if not split_dir.exists():
            print(f"[{sp.upper()}] (missing)")
            continue

        real_dir = split_dir / "real"
        fake_dir = split_dir / "fake"

        real_by_ds = count_real_datasets(real_dir)     # dict dataset -> frames
        fake_by_m  = count_fake_methods(fake_dir)      # dict method  -> frames

        real_total = sum(real_by_ds.values())
        fake_total = sum(fake_by_m.values())
        split_total = real_total + fake_total
        grand_total += split_total

        print(f"[{sp.upper()}] total: {pretty_int(split_total)}")
        print(f"  face  | real: {pretty_int(real_total)} | fake_total: {pretty_int(fake_total)} | sum: {pretty_int(split_total)}")

        if real_by_ds:
            for ds in sorted(real_by_ds.keys()):
                label = ds if ds != "__flat__" else "(real-flat)"
                print(f"     └─ {label:<15}: {pretty_int(real_by_ds[ds])}")
        else:
            print(f"     └─ (no real frames)")

        if fake_by_m:
            for mname in sorted(fake_by_m.keys()):
                print(f"     └─ {mname:<15}: {pretty_int(fake_by_m[mname])}")
        else:
            print(f"     └─ (no fake frames)")
        print()

    print(f"Σ GRAND TOTAL (all splits): {pretty_int(grand_total)}")

if __name__ == "__main__":
    main()
