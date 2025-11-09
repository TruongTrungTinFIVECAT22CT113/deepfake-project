from pathlib import Path

root = Path(r"H:\deepfake-project\deepfake-project\data\processed_multi\face")
NEEDED = 90  # cho 30@3
for split in ["train","val","test"]:
    for kind in ["real","fake"]:
        base = root/split/kind
        if not base.exists(): continue
        for first in base.iterdir():    # real: dataset, fake: method
            if not first.is_dir(): continue
            for vid in first.iterdir(): # video_id
                if not vid.is_dir(): continue
                n = sum(1 for f in vid.iterdir() if f.suffix.lower() in [".jpg",".jpeg",".png"])
                if n < NEEDED:
                    print(f"{split}/{kind}/{first.name}/{vid.name}: {n}")
