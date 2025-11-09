# -*- coding: utf-8 -*-
"""
extract_face_window.py  — auto-anchor + lock + append-safe

Chức năng:
- Tự tìm "anchor" (thời điểm tốt nhất trong 0..tmax) để bắt đầu cắt mặt đúng người (nếu có --auto_anchor)
- Khóa theo dõi khuôn mặt bằng IoU với bbox trước đó (+ optional so khớp với ref)
- Append an toàn: nếu --no_overwrite, tự dò index lớn nhất và ghi nối tiếp thay vì xóa/ghi đè
- Hỗ trợ chọn ROI ngang (x-min, x-max) để hạn chế khuôn mặt ngoài vùng quan tâm (tùy chọn)

Yêu cầu:
- mediapipe (face_detection)
- imageio[ffmpeg], opencv-python
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import math
import sys
from typing import Optional, Tuple, List

import cv2
import numpy as np
import imageio

# mediapipe face detection (light, nhanh)
import mediapipe as mp
mp_fd = mp.solutions.face_detection


# ----------------------------- utils -----------------------------

def log(*a):
    print(*a)
    sys.stdout.flush()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def zpad(n: int, k: int = 6) -> str:
    return str(n).zfill(k)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    denom = area_a + area_b - inter
    return float(inter/denom) if denom > 0 else 0.0


def to_vec_simple(img_bgr: np.ndarray, size: int = 112) -> np.ndarray:
    """Embed cực nhẹ: resize + YCrCb + chuẩn hóa chuẩn tắc."""
    rsz = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rsz = cv2.cvtColor(rsz, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    v = rsz.reshape(-1)
    v = (v - v.mean()) / (v.std() + 1e-6)
    n = np.linalg.norm(v) + 1e-6
    return v / n


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def load_ref_vec(ref_dir: Path, take: int = 12) -> Optional[np.ndarray]:
    imgs = []
    for p in sorted(ref_dir.glob("*.jpg"))[:take]:
        im = cv2.imread(str(p))
        if im is None: continue
        imgs.append(to_vec_simple(im))
    if not imgs:
        return None
    v = np.mean(np.stack(imgs, 0), 0)
    v /= (np.linalg.norm(v) + 1e-6)
    return v


def frame_index_from_time(t: float, fps: float) -> int:
    return max(0, int(round(t * fps)))


def read_frame(reader, idx: int) -> Optional[np.ndarray]:
    try:
        frame_rgb = reader.get_data(idx)  # RGB
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


@dataclass
class DetectConfig:
    iou_thr: float = 0.30
    pad_frac: float = 0.25
    roi: Optional[Tuple[float, float]] = None  # (xmin, xmax) normalized [0..1]
    sim_thr: float = 0.70
    stop_on_miss: bool = True


# ----------------------- detector & cropper -----------------------

class FaceDetector:
    def __init__(self, model_selector: int = 0, min_confidence: float = 0.5):
        # model_selector: 0=short range, 1=full range (mediapipe)
        self.fd = mp_fd.FaceDetection(model_selection=model_selector, min_detection_confidence=min_confidence)

    def detect(self, bgr: np.ndarray) -> List[Tuple[float,float,float,float,float]]:
        """Trả về list (x1,y1,x2,y2, score) trong toạ độ tuyệt đối."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.fd.process(rgb)
        out = []
        if res.detections:
            for det in res.detections:
                score = det.score[0] if det.score else 0.0
                bb = det.location_data.relative_bounding_box
                # mediapipe trả về (x,y,w,h) normalized
                x1 = clamp(bb.xmin * w, 0, w)
                y1 = clamp(bb.ymin * h, 0, h)
                x2 = clamp((bb.xmin + bb.width) * w, 0, w)
                y2 = clamp((bb.ymin + bb.height) * h, 0, h)
                if x2 > x1 and y2 > y1:
                    out.append((x1, y1, x2, y2, score))
        return out

    def close(self):
        self.fd.close()


def crop_with_pad(bgr: np.ndarray, box: Tuple[float,float,float,float], pad_frac: float, out_size: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = max(bw, bh)
    side = side * (1.0 + 2.0 * pad_frac)
    nx1 = clamp(int(round(cx - side / 2)), 0, w - 1)
    ny1 = clamp(int(round(cy - side / 2)), 0, h - 1)
    nx2 = clamp(int(round(cx + side / 2)), 1, w)
    ny2 = clamp(int(round(cy + side / 2)), 1, h)
    crop = bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop


def pick_face(
    boxes: List[Tuple[float,float,float,float,float]],
    last_box: Optional[np.ndarray],
    frame_bgr: np.ndarray,
    cfg: DetectConfig,
    ref_vec: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Trả về bbox (x1,y1,x2,y2) được chọn hoặc None.
    Ưu tiên:
      - Lọc theo ROI nếu có
      - Nếu có ref_vec: chọn mặt có cosine sim cao nhất trên crop
      - Nếu không có ref_vec: chọn bbox có IoU với last_box cao nhất (nếu last_box tồn tại), else diện tích lớn nhất
    """
    if not boxes: return None
    h, w = frame_bgr.shape[:2]

    # lọc theo ROI ngang (x-min, x-max) nếu có
    candidates = []
    if cfg.roi is not None:
        rx1 = cfg.roi[0] * w
        rx2 = cfg.roi[1] * w
        for (x1, y1, x2, y2, s) in boxes:
            cx = 0.5 * (x1 + x2)
            if rx1 <= cx <= rx2:
                candidates.append((x1, y1, x2, y2, s))
    else:
        candidates = boxes

    if not candidates:  # nếu ROI quá chặt, rơi về toàn bộ boxes
        candidates = boxes

    # nếu có ref_vec: chọn theo sim cao nhất
    if ref_vec is not None:
        best = None
        best_sim = -1.0
        for (x1, y1, x2, y2, s) in candidates:
            crop = crop_with_pad(frame_bgr, (x1, y1, x2, y2), cfg.pad_frac, 128)
            if crop is None: continue
            v = to_vec_simple(crop)
            sim = cos_sim(v, ref_vec)
            if sim > best_sim:
                best_sim = sim
                best = (x1, y1, x2, y2)
        if best is None:
            return None
        # nếu cần, đây có thể kiểm sim >= cfg.sim_thr để "lock theo danh tính"
        return np.array(best, dtype=np.float32)

    # không có ref_vec:
    # - nếu đã lock (có last_box): ưu tiên IoU cao nhất
    # - nếu chưa lock: chọn bbox lớn nhất
    if last_box is not None:
        best = None
        best_iou = -1.0
        for (x1, y1, x2, y2, s) in candidates:
            i = iou_xyxy(np.array([x1, y1, x2, y2]), last_box)
            if i > best_iou:
                best_iou = i
                best = (x1, y1, x2, y2)
        return np.array(best, dtype=np.float32) if best is not None else None
    else:
        # diện tích lớn nhất
        areas = [max(0.0, (x2-x1)) * max(0.0, (y2-y1)) for (x1,y1,x2,y2,_) in candidates]
        k = int(np.argmax(areas))
        x1, y1, x2, y2, _ = candidates[k]
        return np.array([x1, y1, x2, y2], dtype=np.float32)


# ----------------------- auto-anchor scanner -----------------------

def scan_anchor_time(
    reader, fps: float, tmax: float, step: float,
    detector: FaceDetector,
    ref_vec: Optional[np.ndarray],
    cfg: DetectConfig
) -> float:
    """
    Tìm thời điểm tốt nhất để bắt đầu (0..tmax).
    - Nếu có ref_vec: chọn frame có cosine sim cao nhất (với crop tốt nhất theo ref).
    - Nếu không có ref_vec: chọn frame có khuôn mặt "lớn nhất".
    """
    best_t = 0.0
    best_score = -1.0
    t = 0.0
    HINT = "sim" if ref_vec is not None else "area"

    while t <= tmax + 1e-6:
        idx = frame_index_from_time(t, fps)
        bgr = read_frame(reader, idx)
        if bgr is None:
            t += step
            continue
        boxes = detector.detect(bgr)

        if not boxes:
            t += step
            continue

        if ref_vec is not None:
            # chọn bbox có sim cao nhất với ref
            best_sim = -1.0
            for (x1,y1,x2,y2,_) in boxes:
                crop = crop_with_pad(bgr, (x1,y1,x2,y2), cfg.pad_frac, 128)
                if crop is None: continue
                s = cos_sim(to_vec_simple(crop), ref_vec)
                if s > best_sim:
                    best_sim = s
            score = best_sim
        else:
            # không có ref: dùng diện tích lớn nhất
            score = max((x2-x1)*(y2-y1) for (x1,y1,x2,y2,_) in boxes)

        if score > best_score:
            best_score = score
            best_t = t

        t += step

    log(f"[auto-anchor] best_t={best_t:.2f}s  best_{HINT}={best_score:.3f}")
    return best_t


# ----------------------------- main logic -----------------------------

def extract(
    video: str,
    out: str,
    start: float,
    duration: float,
    fps: int,
    img_size: int,
    pad_frac: float,
    iou_thr: float,
    roi: Optional[Tuple[float,float]],
    sim_thr: float,
    overwrite: bool,
    stop_on_miss: bool,
    auto_anchor: bool,
    anchor_tmax: float,
    anchor_step: float,
    append: bool,
    ref_dir: Optional[str] = None,
) -> int:
    vid = Path(video)
    out_dir = Path(out)

    exists = vid.exists()
    log(f"[DEBUG] exists={exists} size={(vid.stat().st_size if exists else 'NA')} path={vid}")

    if not exists:
        raise RuntimeError(f"Cannot open video: {video}. exists={exists}")

    # xử lý out_dir
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    # append-safe index
    start_idx = 0
    if (not overwrite) or append:
        exts = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
        if exts:
            digits = []
            for p in exts:
                s = p.stem
                if s.isdigit():
                    digits.append(int(s))
            if digits:
                start_idx = max(digits) + 1
            else:
                start_idx = len(exts)

    # load ref vec nếu có
    ref_vec = None
    if ref_dir:
        ref_vec = load_ref_vec(Path(ref_dir), take=12)
        if ref_vec is None:
            log("WARN: ref_dir không có ảnh hợp lệ -> bỏ so khớp danh tính.")

    cfg = DetectConfig(iou_thr=iou_thr, pad_frac=pad_frac, roi=roi, sim_thr=sim_thr, stop_on_miss=stop_on_miss)

    reader = imageio.get_reader(str(vid), "ffmpeg")
    meta = reader.get_meta_data()
    vfps = float(meta.get("fps", fps))
    if vfps <= 1e-3:
        vfps = float(fps)

    detector = FaceDetector(model_selector=0, min_confidence=0.5)

    # auto-anchor nếu bật
    if auto_anchor:
        best_t = scan_anchor_time(
            reader=reader, fps=vfps, tmax=anchor_tmax, step=anchor_step,
            detector=detector, ref_vec=ref_vec, cfg=cfg
        )
        start = best_t

    s_idx = frame_index_from_time(start, vfps)
    e_idx = frame_index_from_time(start + duration, vfps)

    wrote = 0
    idx_write = start_idx
    last_box = None
    locked = False

    # duyệt khung
    for fi in range(s_idx, e_idx + 1):
        bgr = read_frame(reader, fi)
        if bgr is None:
            if cfg.stop_on_miss:
                log(f"[STOP] read_frame None at fi={fi}")
                break
            else:
                continue

        boxes = detector.detect(bgr)

        if not boxes:
            if cfg.stop_on_miss and not locked:
                log(f"[STOP] no face at fi={fi} before lock")
                break
            # cho qua khung miss sau khi đã lock (nếu stop_on_miss=False)
            continue

        chosen = pick_face(boxes, last_box, bgr, cfg, ref_vec)

        if chosen is None:
            if cfg.stop_on_miss and not locked:
                log(f"[STOP] cannot pick face at fi={fi} before lock")
                break
            else:
                continue

        # nếu đã lock: kiểm IoU tối thiểu để giữ cùng người
        if locked and last_box is not None:
            if iou_xyxy(chosen, last_box) < cfg.iou_thr:
                # nếu có ref_vec, thử xác thực thêm bằng sim để cho qua
                if ref_vec is not None:
                    crop_chk = crop_with_pad(bgr, chosen, cfg.pad_frac, 128)
                    if crop_chk is not None:
                        sim = cos_sim(to_vec_simple(crop_chk), ref_vec)
                        if sim < cfg.sim_thr:
                            if cfg.stop_on_miss:
                                log(f"[STOP] IoU/sim fail at fi={fi}")
                                break
                            else:
                                continue
                else:
                    if cfg.stop_on_miss:
                        log(f"[STOP] IoU fail at fi={fi}")
                        break
                    else:
                        continue

        # crop & save
        crop = crop_with_pad(bgr, chosen, cfg.pad_frac, img_size)
        if crop is None:
            if cfg.stop_on_miss and not locked:
                log(f"[STOP] empty crop at fi={fi} before lock")
                break
            else:
                continue

        if (ref_vec is not None) and (not locked):
            # re-check sim tại thời khắc lock lần đầu
            s = cos_sim(to_vec_simple(crop), ref_vec)
            if s < cfg.sim_thr:
                # chưa qua ngưỡng để lock
                if cfg.stop_on_miss:
                    log(f"[STOP] first-lock sim<{cfg.sim_thr:.2f} at fi={fi} (sim={s:.3f})")
                    break
                else:
                    # thử tiếp khung sau
                    continue

        # save
        out_path = out_dir / f"{zpad(idx_write)}.jpg"
        # tránh đè (append-safe)
        while out_path.exists():
            idx_write += 1
            out_path = out_dir / f"{zpad(idx_write)}.jpg"

        cv2.imwrite(str(out_path), crop)
        wrote += 1
        idx_write += 1

        # update lock
        last_box = chosen
        locked = True

    try:
        reader.close()
    except Exception:
        pass
    detector.close()

    log(f"✅ wrote {wrote} frames to {out_dir}")
    return wrote


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Đường dẫn video nguồn")
    ap.add_argument("--out", required=True, help="Thư mục đích để lưu ảnh")
    ap.add_argument("--ref_dir", default=None, help="Thư mục ảnh tham chiếu (để khóa đúng người). Khuyên dùng.")

    # thời gian/cửa sổ cắt
    ap.add_argument("--start", type=float, default=0.0, help="Thời điểm bắt đầu (giây). Bỏ qua nếu --auto_anchor bật.")
    ap.add_argument("--duration", type=float, default=4.0, help="Độ dài cửa sổ cần cắt (giây)")
    ap.add_argument("--fps", type=int, default=25, help="FPS fallback nếu video không có meta fps")

    # tham số crop/lock
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--pad_frac", type=float, default=0.25, help="Tăng khung crop theo tỉ lệ cạnh (mặc định 0.25)")
    ap.add_argument("--iou_thr", type=float, default=0.30, help="Ngưỡng IoU để giữ cùng người sau khi lock")
    ap.add_argument("--sim_thr", type=float, default=0.70, help="Ngưỡng cosine-sim khi lock bằng ref")
    ap.add_argument("--roi", nargs=2, type=float, default=None, help="Giới hạn ROI ngang (xmin xmax) dạng [0..1]")

    # flags
    ap.add_argument("--no_overwrite", action="store_true", help="Không xóa thư mục out; bật chế độ append an toàn")
    ap.add_argument("--no_stop_on_miss", action="store_true", help="Không dừng khi miss trước khi lock (cố đi tiếp)")
    ap.add_argument("--append", action="store_true", help="Ghi nối tiếp (ép bật append)")

    # auto-anchor
    ap.add_argument("--auto_anchor", action="store_true", help="Tự quét 0..tmax, chọn anchor tốt nhất để bắt đầu")
    ap.add_argument("--anchor_tmax", type=float, default=5.0, help="Khoảng thời gian quét anchor (giây)")
    ap.add_argument("--anchor_step", type=float, default=0.04, help="Bước quét anchor (giây)")

    return ap.parse_args()


def main():
    args = parse_args()

    roi = tuple(map(float, args.roi)) if args.roi is not None else None
    wrote = extract(
        video=args.video,
        out=args.out,
        start=args.start,
        duration=args.duration,
        fps=args.fps,
        img_size=args.img_size,
        pad_frac=args.pad_frac,
        iou_thr=args.iou_thr,
        roi=roi,
        sim_thr=args.sim_thr,
        overwrite=(not args.no_overwrite),
        stop_on_miss=(not args.no_stop_on_miss),
        auto_anchor=args.auto_anchor,
        anchor_tmax=args.anchor_tmax,
        anchor_step=args.anchor_step,
        append=(args.append or args.no_overwrite),
        ref_dir=args.ref_dir,
    )
    if wrote == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
