import subprocess
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--resolution", type=str, required=True,
                help="VD: 640x480 | 960x720 | 1024x768")
args = ap.parse_args()

# Tách width x height
try:
    w, h = args.resolution.lower().split("x")
    w, h = int(w), int(h)
except:
    raise ValueError(f"--resolution phải có dạng WxH, ví dụ: 640x480. Nhận được: {args.resolution}")

src_root = r"H:\deepfake-project\deepfake-project\data\videos_test\640x480"
dst_root = rf"H:\deepfake-project\deepfake-project\data\videos_test\{args.resolution}"

print(f"Resolution : {w}x{h}")
print(f"Source     : {src_root}")
print(f"Destination: {dst_root}")

success = 0
failed  = []

for dirpath, dirnames, filenames in os.walk(src_root):
    # Bỏ qua các thư mục resolution đã tạo (tránh đệ quy vào 640x480, 960x720...)
    dirnames[:] = [d for d in dirnames if not d[0].isdigit()]

    videos = [f for f in filenames if f.lower().endswith(".mp4")]
    if not videos:
        continue

    rel_path = os.path.relpath(dirpath, src_root)
    dst_dir  = os.path.join(dst_root, rel_path)
    os.makedirs(dst_dir, exist_ok=True)

    total = len(videos)
    print(f"\n{'='*60}")
    print(f"Thư mục : {dirpath}")
    print(f"Đích    : {dst_dir}")
    print(f"Số video: {total}")

    for idx, filename in enumerate(sorted(videos), 1):
        src_path = os.path.join(dirpath, filename)
        dst_path = os.path.join(dst_dir, filename)

        cmd = [
            "ffmpeg",
            "-i", src_path,
            "-vf", f"scale={w}:{h},setsar=1",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-tune", "hq",
            "-rc",  "vbr",
            "-cq",  "18",
            "-b:v", "0",
            "-c:a", "copy",
            dst_path, "-y"
        ]

        print(f"  [{idx:3d}/{total}] {filename} ...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            success += 1
            print("✓")
        else:
            failed.append(os.path.join(rel_path, filename))
            print("✗ LỖI")
            print(result.stderr[-300:])

print(f"\n{'='*60}")
print(f"✓ Thành công : {success}")
if failed:
    print(f"✗ Thất bại   : {len(failed)}")
    for f in failed:
        print(f"   - {f}")
print(f"Hoàn thành! Videos {args.resolution} lưu tại: {dst_root}")