import os
from collections import defaultdict

DATA_ROOT = r"H:\Deepfake\deepfake-project\data\processed"
# BỔ SUNG CÁC ĐUÔI ẢNH PHỔ BIẾN
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def count_in_dir(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                total += 1
    return total

def main():
    counts = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.isdir(split_dir):
            continue

        # REAL
        real_dir = os.path.join(split_dir, "real")
        if os.path.isdir(real_dir):
            c = count_in_dir(real_dir)
            counts[split]["Original"] += c
            total[split] += c

        # FAKE / per method
        fake_dir = os.path.join(split_dir, "fake")
        if os.path.isdir(fake_dir):
            for m in sorted(d for d in os.listdir(fake_dir) if os.path.isdir(os.path.join(fake_dir, d))):
                c = count_in_dir(os.path.join(fake_dir, m))
                counts[split][m] += c
                total[split] += c

    print("📊 Frame counts per class:\n")
    for split in ["train", "val", "test"]:
        print(f"[{split.upper()}] total: {total[split]:,}")
        for cls, num in sorted(counts[split].items()):
            print(f"  - {cls:<16}: {num:,}")
        print()

    # (tùy chọn) lưu CSV tổng kết
    out_dir = os.path.join(DATA_ROOT, "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "frame_summary.csv")
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Split","Class","Frames"])
        for split in counts:
            for cls, num in counts[split].items():
                w.writerow([split, cls, num])
    print(f"✅ Báo cáo đã lưu: {out_csv}")

if __name__ == "__main__":
    main()
