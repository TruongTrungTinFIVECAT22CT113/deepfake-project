# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, cv2, imageio, numpy as np
import os

def to_vec(img, size=112):
    rsz = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    rsz = cv2.cvtColor(rsz, cv2.COLOR_BGR2YCrCb)
    v = rsz.astype(np.float32).reshape(-1)
    v = (v - v.mean()) / (v.std() + 1e-6)
    n = np.linalg.norm(v) + 1e-6
    return v / n

def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6))

def load_ref_vec(ref_dir: Path, take=8):
    ims = []
    for p in sorted(ref_dir.glob("*.jpg"))[:take]:
        im = cv2.imread(str(p))
        if im is None: continue
        ims.append(to_vec(im))
    if not ims: return None
    v = np.mean(np.stack(ims,0),0)
    return v / (np.linalg.norm(v)+1e-6)

def frame_at(reader, t):
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 25.0))
    idx = max(0, int(round(t*fps)))
    try:
        img = reader.get_data(idx)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--tmax", type=float, default=5.0, help="quét trong [0, tmax] giây")
    ap.add_argument("--step", type=float, default=0.04, help="bước quét (s)")
    args = ap.parse_args()

    v = Path(args.video); r = Path(args.ref_dir)
    ref = load_ref_vec(r, take=8)
    if ref is None:
        print("ERR: no ref vec"); return

    reader = imageio.get_reader(str(v), "ffmpeg")
    best_t, best_s = 0.0, -1.0
    t = 0.0
    while t <= args.tmax:
        f = frame_at(reader, t)
        if f is not None:
            s = cos(to_vec(f), ref)
            if s > best_s:
                best_s, best_t = s, t
        t += args.step
    try: reader.close()
    except: pass

    print(f"[anchor] best_t={best_t:.2f}s  best_sim={best_s:.3f}")

if __name__ == "__main__":
    main()
