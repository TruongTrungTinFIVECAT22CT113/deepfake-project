import os
import math
import argparse
from collections import defaultdict, Counter

try:
    import cv2
except Exception as e:
    raise SystemExit("⚠️ Cần cài opencv-python: pip install opencv-python") from e

def sec_fmt(s):
    return f"{s:.2f}s"

def percent(a, b):
    if b <= 0: return 0.0
    return 100.0 * float(a) / float(b)

def quantiles(vals, qs=(0.05, 0.5, 0.95)):
    if not vals: return [0.0 for _ in qs]
    vs = sorted(vals)
    out = []
    for q in qs:
        idx = q * (len(vs) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi: out.append(float(vs[lo])); continue
        w = idx - lo
        out.append(float(vs[lo] * (1 - w) + vs[hi] * w))
    return out

def video_duration_seconds(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0, 0.0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    dur = (frames / fps) if (fps and frames) else 0.0
    return dur, fps, frames

def walk_leaf_dirs(root, video_exts={".mp4", ".mov", ".mkv", ".avi"}):
    """
    Trả về các thư mục 'lá' có chứa video (không có subdir chứa video bên dưới).
    """
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # có file video trong dirpath?
        vids_here = any(os.path.splitext(f)[1].lower() in video_exts for f in filenames)
        # có subdir bên dưới cũng chứa video?
        has_child_with_videos = False
        for d in dirnames:
            child = os.path.join(dirpath, d)
            for _, _, files in os.walk(child):
                if any(os.path.splitext(ff)[1].lower() in video_exts for ff in files):
                    has_child_with_videos = True
                    break
            if has_child_with_videos: break
        if vids_here and not has_child_with_videos:
            leaf_dirs.append(dirpath)
    return sorted(leaf_dirs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=False,
                    default=r"H:\Deepfake\deepfake-project\data\videos",
                    help="Thư mục gốc chứa video (giống bạn đang dùng).")
    ap.add_argument("--fps", type=float, default=30.0, help="FPS mục tiêu dùng khi ước tính frames per segment.")
    ap.add_argument("--segments", type=int, default=10, help="Số segment tối đa mỗi video (segments mode).")
    ap.add_argument("--seg_len", type=float, default=0.6, help="Độ dài mỗi segment (giây).")
    ap.add_argument("--margin", type=float, default=0.1, help="Khoảng cách giữa các segment (giây, không chồng lấn).")
    ap.add_argument("--target_train", type=int, default=100000)
    ap.add_argument("--target_val", type=int, default=10000)
    ap.add_argument("--target_test", type=int, default=5000)
    ap.add_argument("--report_txt", type=str, default=r"H:\Deepfake\deepfake-project\reports\probe_report.txt")
    ap.add_argument("--report_csv", type=str, default=r"H:\Deepfake\deepfake-project\reports\probe_report.csv")
    args = ap.parse_args()

    root = args.root
    leaf_dirs = walk_leaf_dirs(root)
    if not leaf_dirs:
        print(f"⚠️ Không tìm thấy thư mục lá có video trong: {root}")
        return

    K_MAX = lambda dur: int(math.floor((max(0.0, dur) + args.margin) / (args.seg_len + args.margin)))
    FRAMES_PER_SEG = int(round(args.seg_len * args.fps))

    rows = []
    total_capacity_frames = 0

    print("🔎 Đang quét…")
    for d in leaf_dirs:
        rel = os.path.relpath(d, root)
        # liệt kê file video
        vids = [f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in {".mp4", ".mov", ".mkv", ".avi"}]
        vids.sort()
        durs = []
        caps_frames = []  # capacity frames per video
        fps_bag = []
        for v in vids:
            path = os.path.join(d, v)
            dur, fps, frames = video_duration_seconds(path)
            durs.append(dur)
            fps_bag.append(fps if fps else 0.0)
            kmax = K_MAX(dur)
            segs = min(kmax, max(0, args.segments))
            caps_frames.append(segs * FRAMES_PER_SEG)

        n = len(vids)
        min_d = min(durs) if durs else 0.0
        p5, med, p95 = quantiles(durs, qs=(0.05, 0.5, 0.95))
        max_d = max(durs) if durs else 0.0
        cap_total = int(sum(caps_frames))
        total_capacity_frames += cap_total

        rows.append({
            "folder": rel.replace("\\", "/"),
            "count": n,
            "dur_min": min_d,
            "dur_p5": p5,
            "dur_med": med,
            "dur_p95": p95,
            "dur_max": max_d,
            "fps_median": quantiles([f for f in fps_bag if f > 0], qs=(0.5,))[0] if any(fps_bag) else 0.0,
            "cap_frames": cap_total
        })

    # Gộp cấp cha tiện xem nhanh (vd Diffusion/match tổng)
    def parent_key(p):
        parts = p.split("/")
        # gộp sâu tối đa 3 lớp cho dễ đọc (bạn có thể chỉnh nếu muốn)
        if len(parts) >= 3:
            return "/".join(parts[:3])
        return "/".join(parts)

    agg = defaultdict(int)
    for r in rows:
        agg[parent_key(r["folder"])] += r["cap_frames"]

    # Ước lượng so với target
    # Ở đây chỉ so trên TỔNG (không chia theo split) để biết có “đủ nguyên liệu” hay không.
    target_total_per_class = args.target_train + args.target_val + args.target_test  # ~115k
    # In & ghi file
    os.makedirs(os.path.dirname(args.report_txt), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_csv), exist_ok=True)

    with open(args.report_txt, "w", encoding="utf-8") as fout:
        fout.write(f"Probe root: {root}\n")
        fout.write(f"Config: fps={args.fps}, segments={args.segments}, seg_len={args.seg_len}s, margin={args.margin}s\n")
        fout.write(f"Frames/segment ≈ {FRAMES_PER_SEG}\n")
        fout.write(f"Target per class (total all splits) ≈ {target_total_per_class}\n\n")

        fout.write("=== Per leaf folder ===\n")
        for r in rows:
            ok = "✅" if r["cap_frames"] >= target_total_per_class else "⚠️"
            fout.write(
                f"{ok} {r['folder']:<60} | files={r['count']:4d} | "
                f"dur[min/p5/med/p95/max]=[{sec_fmt(r['dur_min'])}/{sec_fmt(r['dur_p5'])}/{sec_fmt(r['dur_med'])}/{sec_fmt(r['dur_p95'])}/{sec_fmt(r['dur_max'])}] | "
                f"fps_med={r['fps_median']:.1f} | cap_frames≈{r['cap_frames']}\n"
            )

        fout.write("\n=== Aggregated by parent (first 3 levels) ===\n")
        for k, cap in sorted(agg.items()):
            ok = "✅" if cap >= target_total_per_class else "⚠️"
            fout.write(f"{ok} {k:<60} | cap_frames≈{cap}\n")

        fout.write(f"\nTotal capacity frames across all leaf folders ≈ {total_capacity_frames}\n")

    # CSV
    import csv
    with open(args.report_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["folder","count","dur_min","dur_p5","dur_med","dur_p95","dur_max","fps_median","cap_frames"])
        for r in rows:
            w.writerow([r["folder"], r["count"], f"{r['dur_min']:.4f}", f"{r['dur_p5']:.4f}",
                        f"{r['dur_med']:.4f}", f"{r['dur_p95']:.4f}", f"{r['dur_max']:.4f}",
                        f"{r['fps_median']:.2f}", r["cap_frames"]])

    print(f"✅ Done. Reports:\n  - {args.report_txt}\n  - {args.report_csv}")
    print("💡 Tip: Nếu một mục bị ⚠️ (cap_frames < target_total_per_class), hãy giảm seg_len / tăng segments / giảm margin.")

if __name__ == "__main__":
    main()
