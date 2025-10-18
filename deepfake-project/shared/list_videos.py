import os

ROOT = r"H:\Deepfake\deepfake-project\data\videos"
LOG_FILE = r"H:\Deepfake\deepfake-project\reports\video_list.log"

def scan_videos(root):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for dirpath, dirnames, filenames in os.walk(root):
            # lọc file video
            videos = [fn for fn in filenames if fn.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))]
            if videos:
                rel = os.path.relpath(dirpath, root)
                f.write(f"\n📁 {rel}\n")
                f.write("-" * (len(rel) + 5) + "\n")
                for vid in sorted(videos):
                    f.write(f"{vid}\n")
    print(f"✅ Đã quét xong. Log lưu tại: {LOG_FILE}")

if __name__ == "__main__":
    scan_videos(ROOT)
