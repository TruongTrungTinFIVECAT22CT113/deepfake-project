# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, tempfile
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

from .utils import draw_box_with_label_np, render_verdict_text, average_threshold
from .face_detection import crop_largest_face

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_eval_transform(img_size: int):
    # exactly like backend_eval.py: ToPILImage() expects HxWxC array; we pass BGR array as-is
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

@torch.no_grad()
def _predict_image_tensor(x_chw: torch.Tensor, model, device):
    xb = x_chw.unsqueeze(0).to(device, non_blocking=True)
    lb, lm, *_ = model(xb)
    pbin = torch.softmax(lb, dim=1).squeeze(0).cpu().numpy()  # [fake_prob, real_prob] (fake_index=0)
    pmth = torch.softmax(lm, dim=1).squeeze(0).cpu().numpy()
    p_fake = float(pbin[0])
    p_real = float(pbin[1]) if pbin.shape[0] > 1 else float(1.0 - p_fake)
    return p_fake, p_real, pmth

@torch.no_grad()
def _ensemble_predict_bgr_crop(
    detectors_info: List[dict],
    crop_bgr: np.ndarray,
    tx: transforms.Compose,
    method_names: List[str]
):
    p_fake_sum, p_real_sum = 0.0, 0.0
    pm_sum = np.zeros(len(method_names), dtype=np.float64) if method_names else None
    # We apply a single eval-style transform (same as backend_eval.py) for all models
    x = tx(crop_bgr)  # BGR array -> ToPILImage (assumes RGB) -> (no channel swap, matching eval quirk)
    for info in detectors_info:
        model = info["model"]; device = info["device"]
        pf, pr, pm = _predict_image_tensor(x, model, device)
        p_fake_sum += pf; p_real_sum += pr
        if method_names:
            pm_sum += pm[:len(method_names)]
    n = max(1, len(detectors_info))
    if method_names:
        return p_fake_sum/n, p_real_sum/n, (pm_sum/n)
    else:
        return p_fake_sum/n, p_real_sum/n, np.zeros(1, dtype=np.float32)

@torch.no_grad()
def analyze_video(
    video_path: str,
    detectors_info: List[dict],
    method_names: List[str],
    # FE threshold override (honored only when a single model is enabled)
    fe_thr_override: Optional[float],
    # detector backend prefs
    detector_backend: str = "retinaface",
    bbox_scale: float = 1.10,
    det_thr: float = 0.5,
    # viz
    box_thickness: int = 3,
    # strict fallback control (match eval by default)
    allow_fallback: bool = False,
):
    # ---------- choose threshold ----------
    if len(detectors_info) <= 0:
        return None, "No enabled model.", {}, ""

    if len(detectors_info) == 1:
        thr_used = float(fe_thr_override) if (fe_thr_override is not None) else float(detectors_info[0].get("best_thr", 0.5))
        thr_override_ignored = False
        img_size = int(detectors_info[0].get("img_size", 384))
    else:
        thr_used = average_threshold(detectors_info)
        thr_override_ignored = True
        # if multiple models have different sizes, use the first's size for tx (matching eval uses a single size)
        img_size = int(detectors_info[0].get("img_size", 384))

    tx = build_eval_transform(img_size)

    # ---------- open video ----------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None, "Cannot open video.", {}, ""

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0: fps = 25.0
    except Exception:
        fps = 25.0
    if w <= 0 or h <= 0:
        cap.release()
        return None, "Invalid video.", {}, ""

    # ---------- writer ----------
    tmpdir = tempfile.mkdtemp(prefix="df_web_")
    out_path = os.path.join(tmpdir, "out.mp4")
    use_imageio = False
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7)
        use_imageio = True
        vout = None
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vout = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writer = None

    # ---------- loop ----------
    requested_backend = (detector_backend or "retinaface").strip().lower()

    frames_total = 0
    fake_frames = 0
    mnames = list(method_names or [])
    m_count = {m: 0 for m in mnames}
    backend_used_final = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames_total += 1

        # strict: largest face by requested backend
        # NOTE: crop_largest_face returns (PIL-crop, box, backend_used), but to match eval we rebuild BGR crop from box.
        try:
            _, box, backend_used = crop_largest_face(
                Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)),  # only for detector; we'll use frame_bgr + box for crop
                backend=requested_backend,
                device=detectors_info[0]["device"].type if hasattr(detectors_info[0]["device"], "type") else "cuda",
                det_thr=float(det_thr),
                det_size=640,
                bbox_scale=float(bbox_scale),
                allow_fallback=allow_fallback,
            )
        except Exception as e:
            # if detector fails and no fallback: mark as no-face -> behave like eval (ok=False) => don't increment fake, but n_frames vẫn tăng
            # To mimic eval more strictly, we'll just continue without counting fake
            continue

        if backend_used_final is None:
            backend_used_final = backend_used

        x1,y1,x2,y2 = [int(v) for v in box]
        # rebuild BGR crop exactly like backend_eval.square_crop_from_bbox
        crop_bgr = frame_bgr[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if crop_bgr.size == 0:
            # behave like eval: skip counting this frame as fake
            continue

        pf, pr, pm = _ensemble_predict_bgr_crop(detectors_info, crop_bgr, tx, mnames)
        is_fake = (pf >= thr_used)
        if is_fake:
            fake_frames += 1
            if len(mnames) > 0:
                m_idx = int(np.argmax(pm))
                m_count[mnames[m_idx]] += 1

        # draw box+label on RGB frame for output video only
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if is_fake and len(mnames) > 0:
            m_idx = int(np.argmax(pm))
            label = mnames[m_idx]   # show method name instead of just "Fake"
        else:
            label = "Real"

        draw_box_with_label_np(
            frame_rgb, [x1,y1,x2,y2], label,
            color=(223,64,64) if is_fake else (64,208,120),
            thickness=int(box_thickness)
        )
        out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if use_imageio:
            try:
                import imageio.v2 as imageio  # type: ignore
                writer.append_data(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
            except Exception:
                pass
        else:
            vout.write(out_bgr)

    cap.release()
    if use_imageio and writer is not None:
        writer.close()
    elif vout is not None:
        vout.release()

    duration_sec = frames_total / fps if fps > 0 else 0.0
    verdict = render_verdict_text(frames_total, fake_frames)

    # method distribution tables
    method_rows_fake: List[Tuple[str, float]] = []
    method_rows_total: List[Tuple[str, float]] = []

    if len(mnames) > 0:
        counts = np.array([m_count[m] for m in mnames], dtype=np.float64)

        # A) % by fake frames (giữ để tương thích)
        if fake_frames > 0 and counts.sum() > 0:
            perc_fake = 100.0 * counts / counts.sum()
            idx = np.argsort(-perc_fake)
            method_rows_fake = [(mnames[int(i)], float(perc_fake[int(i)])) for i in idx]

        # B) % by total frames (mới – tránh hiểu lầm)
        if frames_total > 0:
            perc_total = 100.0 * counts / float(frames_total)
            idx2 = np.argsort(-perc_total)
            method_rows_total = [(mnames[int(i)], float(perc_total[int(i)])) for i in idx2]

    stats = {
        "frames_total": int(frames_total),
        "fake_frames": int(fake_frames),
        "fake_ratio": float(fake_frames / max(1, frames_total)),
        "fps": float(fps),
        "duration_sec": float(duration_sec),
        "threshold_used": float(thr_used),
        "thr_override_ignored": bool(thr_override_ignored),
        "detector_backend_used": backend_used_final or requested_backend,
        "method_distribution": {k: int(v) for k, v in m_count.items()},
    }

    # Trả cả 2: FE sẽ ưu tiên *_total
    return out_path, verdict, stats, method_rows_total or method_rows_fake
