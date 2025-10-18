import os
os.environ["PATH"] += r";C:\ffmpeg\bin"
import subprocess
from pathlib import Path

ROOT = Path("data/original videos")      # chỗ chứa video gốc
OUT_ROOT = Path("data/processed videos") # chỗ chứa video đã convert

VIDEO_EXTS = {".avi", ".mov", ".mkv", ".mp4", ".webm", ".m4v"}

def convert_video(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(dst)
    ]
    print("Converting:", src, "->", dst)
    subprocess.run(cmd, check=True)

def main():
    videos = [p for p in ROOT.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    print(f"Tìm thấy {len(videos)} video cần convert.")

    for src in videos:
        rel = src.relative_to(ROOT)
        dst = OUT_ROOT / rel
        dst = dst.with_suffix(".mp4")
        if dst.exists():
            print("Bỏ qua (đã có):", dst)
            continue
        convert_video(src, dst)

if __name__ == "__main__":
    main()
