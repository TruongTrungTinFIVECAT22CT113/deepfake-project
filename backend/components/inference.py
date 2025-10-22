# -*- coding: utf-8 -*-
import os, tempfile
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import torch
from PIL import Image

from .filters import apply_method_filters
from .utils import (
    draw_box_np, draw_box_with_label_np, draw_red_spot_np,
    overlay_heatmap_in_box, draw_saliency_dots_in_box,
    _render_fake_real_bar, _make_method_table
)

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

def _ensure_playable_mp4(src_path: str) -> str:
    try:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src_path, apiPreference=cv2.CAP_FFMPEG)
            if not cap.isOpened():
                return src_path
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if w == 0 or h == 0:
            cap.release(); return src_path
        tmp = tempfile.mkdtemp(prefix="df_transcode_")
        out = os.path.join(tmp, "reencoded.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vout = cv2.VideoWriter(out, fourcc, float(fps if fps>0 else 25.0), (w, h))
        ok_any = False
        while True:
            ok, frm = cap.read()
            if not ok: break
            ok_any = True; vout.write(frm)
        cap.release(); vout.release()
        return out if ok_any else src_path
    except Exception:
        return src_path

@torch.no_grad()
def _predict_image_tensor(x_chw: torch.Tensor, detector, device, tta: int = 2):
    xb = x_chw.unsqueeze(0).to(device, non_blocking=True)
    use_amp = (device.type == "cuda")
    if use_amp:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            lb, lm = detector(xb)
            if tta and tta >= 2:
                lb2, lm2 = detector(torch.flip(xb, dims=[3]))
                lb = (lb + lb2) / 2; lm = (lm + lm2) / 2
    else:
        lb, lm = detector(xb)
        if tta and tta >= 2:
            lb2, lm2 = detector(torch.flip(xb, dims=[3]))
            lb = (lb + lb2) / 2; lm = (lm + lm2) / 2
    pbin = torch.softmax(lb, dim=1).squeeze(0).cpu().numpy()
    pmth = torch.softmax(lm, dim=1).squeeze(0).cpu().numpy()
    return float(pbin[0]), float(pbin[1]), pmth

@torch.no_grad()
def _ensemble_predict_image_pil(detectors_info, pil_img: Image.Image, method_names: List[str], tta: int = 2):
    p_fake_sum, p_real_sum = 0.0, 0.0
    pm_sum = np.zeros(len(method_names), dtype=np.float64)
    for detector, tfm, device, *_ in detectors_info:
        x = tfm(pil_img)
        pf, pr, pm = _predict_image_tensor(x, detector, device, tta=tta)
        p_fake_sum += pf; p_real_sum += pr; pm_sum += pm[:len(method_names)]
    n = max(1, len(detectors_info))
    return p_fake_sum/n, p_real_sum/n, pm_sum/n

def _saliency_map_point(detector, tfm, pil_img, device, target="fake", m_idx=None):
    detector.eval()
    x = tfm(pil_img).unsqueeze(0).to(device)
    x.requires_grad_(True)
    lb, lm = detector(x)
    if target == "method" and m_idx is not None:
        lm[:, int(m_idx)].sum().backward()
    else:
        lb[:, 0].sum().backward()
    grad = x.grad.detach().abs().squeeze(0)
    sal = grad.sum(dim=0); sal = sal/(sal.max()+1e-8)
    yy, xx = torch.nonzero(sal == sal.max(), as_tuple=False)[0]
    yy=int(yy.item()); xx=int(xx.item())
    H,W = sal.shape
    peak = (int(round(xx*pil_img.width/float(W))), int(round(yy*pil_img.height/float(H))))
    return sal.cpu().numpy().astype(np.float32), peak

@torch.no_grad()
def analyze_video(
    video_path: str, detectors_info, method_names: List[str], det_thr: float,
    use_face_crop=True, override_thr: Optional[float]=None, tta=2,
    box_thickness=3, method_gate=0.55, enable_filters=True,
    saliency_density=0.02, saliency_mode="method",
):
    video_path = _ensure_playable_mp4(video_path)
    thr_global = float(det_thr if override_thr is None else override_thr)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None, "Không mở được video.", "", [], {}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        fps = float(fps); 
        if not np.isfinite(fps) or fps<=0: fps = 25.0
    except Exception:
        fps = 25.0
    if w==0 or h==0:
        return None, "Video không hợp lệ.", "", [], {}

    tmpdir = tempfile.mkdtemp(prefix="df_web_")
    out_path = os.path.join(tmpdir, "out.mp4")
    use_imageio = False
    try:
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7)
        use_imageio = True
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vout = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    from .face_detection import crop_faces
    det0, tfm0, dev0, *_ = detectors_info[0]

    frames_total = 0
    fake_frames = 0
    sum_pfake, sum_preal = 0.0, 0.0

    mnames = list(method_names or [])
    m_count = {m: 0 for m in mnames}
    m_segments = {m: [] for m in mnames}
    m_open = {m: None for m in mnames}

    def _close_all(end_sec: float):
        for m in mnames:
            if m_open[m] is not None:
                m_segments[m].append([float(m_open[m]), float(end_sec)])
                m_open[m] = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        frames_total += 1
        curr_sec = (frames_total - 1) / fps

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)

        if use_face_crop:
            crops, boxes = crop_faces(pil, max_faces=5)
        else:
            crops, boxes = [pil], [[0,0,w,h]]

        best = {"p_fake":-1.0,"p_real":-1.0,"box":None,"m_idx":None,"pm":None,"spot":None,"heat":None}

        for crop, box in zip(crops, boxes):
            if len(detectors_info) > 1:
                pf, pr, pm = _ensemble_predict_image_pil(detectors_info, crop, mnames, tta=tta)
            else:
                pf, pr, pm = _predict_image_tensor(tfm0(crop), det0, dev0, tta=tta)
            pm = pm[:len(mnames)] if len(mnames)>0 else pm
            m_idx = int(np.argmax(pm)) if len(mnames)>0 else None
            m_name = (mnames[m_idx] if (len(mnames)>0 and m_idx is not None) else "")
            m_conf = float(pm[m_idx]) if (len(mnames)>0 and m_idx is not None) else 0.0

            if enable_filters and len(mnames)>0 and m_conf >= float(method_gate):
                crop_f = apply_method_filters(crop, m_name)
                if len(detectors_info) > 1:
                    pf2, pr2, pm2 = _ensemble_predict_image_pil(detectors_info, crop_f, mnames, tta=tta)
                else:
                    pf2, pr2, pm2 = _predict_image_tensor(tfm0(crop_f), det0, dev0, tta=tta)
                if pf2 > pf:
                    pf, pr, pm = pf2, pr2, pm2[:len(mnames)] if pm2 is not None else pm
                    crop = crop_f

            if pf > best["p_fake"]:
                spot=None; heat=None
                try:
                    with torch.enable_grad():
                        target = "method" if (saliency_mode=="method" and len(mnames)>0 and m_idx is not None) else "fake"
                        heat,(sx,sy) = _saliency_map_point(det0, tfm0, crop, dev0, target=target, m_idx=m_idx)
                        spot = (box[0]+sx, box[1]+sy)
                except Exception:
                    pass
                best.update({"p_fake":pf,"p_real":pr,"box":box,"m_idx":m_idx,"pm":pm,"spot":spot,"heat":heat})

        sum_pfake += best["p_fake"]; sum_preal += best["p_real"]
        is_fake = (best["p_fake"] >= thr_global)

        color_fake=(223,64,64); color_real=(64,208,120)
        if best["box"] is not None:
            if is_fake:
                if best["heat"] is not None:
                    overlay_heatmap_in_box(frame_rgb, best["box"], best["heat"], alpha=0.5)
                    draw_saliency_dots_in_box(frame_rgb, best["box"], best["heat"], density=0.02)
                label_txt = (mnames[best["m_idx"]] if (len(mnames)>0 and best["m_idx"] is not None) else "Fake")
                draw_box_with_label_np(frame_rgb, best["box"], label_txt, color=color_fake, thickness=int(box_thickness))
                if best["spot"] is not None:
                    draw_red_spot_np(frame_rgb, best["spot"], radius=14)
            else:
                draw_box_with_label_np(frame_rgb, best["box"], "Real", color=color_real, thickness=int(box_thickness))

        if is_fake:
            fake_frames += 1
            if len(mnames)>0 and best["m_idx"] is not None:
                mname = mnames[best["m_idx"]]
                m_count[mname] += 1
                if m_open[mname] is None:
                    m_open[mname] = curr_sec
                for other in mnames:
                    if other != mname and m_open[other] is not None:
                        m_segments[other].append([float(m_open[other]), float(curr_sec)])
                        m_open[other] = None
            else:
                _close_all(curr_sec)
        else:
            _close_all(curr_sec)

        out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if HAS_IMAGEIO:
            try: imageio.get_writer  # keep linter calm
            except Exception: pass
        if HAS_IMAGEIO and 'writer' in locals():
            try:
                writer.append_data(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
            except Exception:
                pass
        else:
            vout.write(out_bgr)

    cap.release()
    if HAS_IMAGEIO and 'writer' in locals(): writer.close()
    elif 'vout' in locals(): vout.release()

    duration_sec = frames_total / fps if fps > 0 else 0.0
    for m in mnames:
        if m_open[m] is not None:
            m_segments[m].append([float(m_open[m]), float(duration_sec)])
            m_open[m] = None

    pf = sum_pfake / max(1, frames_total)
    pr = sum_preal / max(1, frames_total)
    fr_bar_html = _render_fake_real_bar(pf, pr)

    # method_rows từ đúng method_names của model
    method_rows = []
    if len(mnames)>0 and fake_frames>0:
        counts = np.array([m_count[m] for m in mnames], dtype=np.float64)
        if counts.sum() > 0:
            perc = 100.0 * counts / counts.sum()
            # sort desc như UI cũ
            idx = np.argsort(-perc)
            method_rows = [(mnames[i], float(perc[i])) for i in idx]

    method_distribution = {}
    for m in mnames:
        frames_m = int(m_count[m])
        percent = (100.0 * frames_m / fake_frames) if fake_frames>0 else 0.0
        secs_total = 0.0; segs_out=[]
        for s,e in m_segments[m]:
            if e > s:
                secs_total += (e - s)
                segs_out.append([round(float(s),3), round(float(e),3)])
        method_distribution[m] = {
            "percent": round(float(percent),4),
            "frames": frames_m,
            "seconds_total": round(float(secs_total),3),
            "segments": segs_out,
        }

    verdict = f"Frames: {frames_total} | Fake-frames: {fake_frames} ({(100.0*fake_frames/max(1,frames_total)):.1f}%)"
    stats = {
        "frames_total": int(frames_total),
        "fake_frames": int(fake_frames),
        "fake_ratio": float(fake_frames / max(1, frames_total)),
        "fps": float(fps),
        "duration_sec": float(duration_sec),
        "threshold_used": float(thr_global),
        "method_distribution": method_distribution,
    }
    return out_path, verdict, fr_bar_html, method_rows, stats
