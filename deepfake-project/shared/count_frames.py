# count_frames.py
# Đếm số frame cho cấu trúc processed_multi mới:
# processed_multi/<branch>/<split>/
#   - real_<branch>/ <video-id>/*.jpg
#   - fake_<branch>/<Method>/<video-id>/*.jpg

import os
import argparse
from collections import defaultdict

# Mặc định trỏ tới processed_multi; có thể đổi bằng --data_root
DEFAULT_DATA_ROOT = r"H:\deepfake-project\deepfake-project\data\processed_multi"

# Các đuôi ảnh phổ biến
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BRANCHES = ["face", "head", "full"]
SPLITS = ["train", "val", "test"]


def count_in_dir(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                total += 1
    return total


def scan_split_branch(data_root: str, branch: str, split: str):
    """
    Trả về:
      real_count: int
      per_method: dict[method] = int
      fake_total: int
      split_total: int (real + fake_total)
    """
    split_dir = os.path.join(data_root, branch, split)
    real_dir = os.path.join(split_dir, f"real_{branch}")
    fake_parent = os.path.join(split_dir, f"fake_{branch}")

    real_count = count_in_dir(real_dir) if os.path.isdir(real_dir) else 0

    per_method = {}
    fake_total = 0
    if os.path.isdir(fake_parent):
        for m in sorted(d for d in os.listdir(fake_parent) if os.path.isdir(os.path.join(fake_parent, d))):
            c = count_in_dir(os.path.join(fake_parent, m))
            per_method[m] = c
            fake_total += c

    split_total = real_count + fake_total
    return real_count, per_method, fake_total, split_total


def main():
    ap = argparse.ArgumentParser("Đếm frame trong processed_multi (branch/split/real-fake-method)")
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="Thư mục processed_multi")
    ap.add_argument("--csv", type=str, default=None, help="Đường dẫn CSV xuất báo cáo. Nếu bỏ trống sẽ lưu vào ../reports/frame_summary.csv")
    ap.add_argument("--show_empty", action="store_true", help="Hiển thị cả các mục 0 frame")
    args = ap.parse_args()

    data_root = args.data_root

    # Cấu trúc lưu kết quả
    # counts[split][branch]['real'] = int
    # counts[split][branch]['fake_total'] = int
    # counts[split][branch]['methods'][method] = int
    counts = defaultdict(lambda: defaultdict(lambda: {"real": 0, "fake_total": 0, "methods": defaultdict(int)}))
    totals_per_split = defaultdict(int)
    totals_per_branch = defaultdict(int)  # cộng gộp qua tất cả split

    # Duyệt
    for split in SPLITS:
        for branch in BRANCHES:
            if not os.path.isdir(os.path.join(data_root, branch, split)):
                continue
            real_c, per_m, fake_c, split_tot = scan_split_branch(data_root, branch, split)
            counts[split][branch]["real"] = real_c
            counts[split][branch]["fake_total"] = fake_c
            for m, c in per_m.items():
                counts[split][branch]["methods"][m] += c

            totals_per_split[split] += split_tot
            totals_per_branch[branch] += split_tot

    # In bảng gọn
    print("📊 Frame counts per branch/split (processed_multi):\n")
    for split in SPLITS:
        split_total = totals_per_split.get(split, 0)
        print(f"[{split.upper()}] total: {split_total:,}")
        for branch in BRANCHES:
            if branch not in counts[split]:
                if args.show_empty:
                    print(f"  {branch:<5} | real: 0 | fake_total: 0")
                continue
            real_c  = counts[split][branch]["real"]
            fake_c  = counts[split][branch]["fake_total"]
            print(f"  {branch:<5} | real: {real_c:,} | fake_total: {fake_c:,} | sum: {real_c + fake_c:,}")

            # liệt kê từng method
            methods = counts[split][branch]["methods"]
            if methods:
                for m in sorted(methods.keys()):
                    v = methods[m]
                    if v > 0 or args.show_empty:
                        print(f"     └─ {m:<16}: {v:,}")
        print()

    # Tổng theo branch (gộp mọi split)
    print("📦 Tổng theo branch (gộp train/val/test):")
    for branch in BRANCHES:
        print(f"  - {branch:<5}: {totals_per_branch.get(branch, 0):,}")
    print()

    # Xuất CSV
    if args.csv:
        out_csv = args.csv
    else:
        # ../reports/frame_summary.csv (so với processed_multi)
        reports_dir = os.path.normpath(os.path.join(data_root, "..", "reports"))
        os.makedirs(reports_dir, exist_ok=True)
        out_csv = os.path.join(reports_dir, "frame_summary.csv")

    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Split", "Branch", "Type", "Name", "Frames"])
        for split in SPLITS:
            for branch in BRANCHES:
                if branch not in counts[split]:
                    if args.show_empty:
                        w.writerow([split, branch, "real", f"real_{branch}", 0])
                        w.writerow([split, branch, "fake_total", "", 0])
                    continue

                # real
                w.writerow([split, branch, "real", f"real_{branch}", counts[split][branch]["real"]])

                # fake per method
                methods = counts[split][branch]["methods"]
                if methods:
                    for m, c in methods.items():
                        w.writerow([split, branch, "fake_method", m, c])

                # fake total
                w.writerow([split, branch, "fake_total", "", counts[split][branch]["fake_total"]])

                # branch total (split-level)
                branch_sum = counts[split][branch]["real"] + counts[split][branch]["fake_total"]
                w.writerow([split, branch, "branch_total", "", branch_sum])

        # split total
        for split in SPLITS:
            w.writerow([split, "", "split_total", "", totals_per_split.get(split, 0)])

        # overall totals per branch
        for branch in BRANCHES:
            w.writerow(["", branch, "overall_branch_total", "", totals_per_branch.get(branch, 0)])

    print(f"✅ Báo cáo CSV đã lưu: {out_csv}")


if __name__ == "__main__":
    main()
