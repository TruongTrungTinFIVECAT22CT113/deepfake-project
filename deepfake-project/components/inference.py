import os
import tempfile
import numpy as np
import cv2
import torch
from PIL import Image
from .filters import apply_method_filters
from .utils import (
    draw_box_np, draw_box_with_label_np, draw_red_spot_np, overlay_heatmap_in_box, draw_saliency_dots_in_box,
    _render_fake_real_bar, _make_method_table
)
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

# ---------- helpers ----------
def _ensure_playable_mp4(src_path):
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
        vout = cv2.VideoWriter(out, fourcc, float(fps if fps > 0 else 25.0), (w, h))
        ok_any = False
        while True:
            ok, frm = cap.read()
            if not ok: break
            ok_any = True
            vout.write(frm)
        cap.release(); vout.release()
        return out if ok_any else src_path
    except Exception:
        return src_path

@torch.no_grad()
def predict_image_tensor(x_chw, detector, device, tta=2):
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
def ensemble_predict_image_pil(detectors_info, pil_img, tta=2):
    p_fake_sum, p_real_sum = 0.0, 0.0
    pm_sum = np.zeros(len(detectors_info[0][4]), dtype=np.float64)
    for detector, tfm, device, *_ in detectors_info:
        x = tfm(pil_img)
        p_fake, p_real, pm = predict_image_tensor(x, detector, device, tta=tta)
        p_fake_sum += p_fake; p_real_sum += p_real; pm_sum += pm
    n = max(1, len(detectors_info))
    return p_fake_sum / n, p_real_sum / n, pm_sum / n

def _saliency_map_point(detector, tfm, pil_img, device, target="fake", m_idx=None):
    detector.eval()
    x = tfm(pil_img).unsqueeze(0).to(device)
    x.requires_grad_(True)
    lb, lm = detector(x)
    if target == "method" and m_idx is not None:
        lm[:, int(m_idx)].sum().backward()
    else:
        lb[:, 0].sum().backward()
    grad = x.grad.detach().abs().squeeze(0)  # [C,H,W]
    sal = grad.sum(dim=0)
    sal = sal / (sal.max() + 1e-8)
    yy, xx = torch.nonzero(sal == sal.max(), as_tuple=False)[0]
    yy = int(yy.item()); xx = int(xx.item())
    H, W = sal.shape
    peak = (int(round(xx * pil_img.width / float(W))), int(round(yy * pil_img.height / float(H))))
    return sal.cpu().numpy().astype(np.float32), peak

@torch.no_grad()
def analyze_video(video_path, detectors_info, method_names, det_thr, use_face_crop=True, override_thr=None, tta=2,
                  box_thickness=3, method_gate=0.55, enable_filters=True,
                  saliency_density=0.02, saliency_mode="fake"):
    # đảm bảo phát được
    video_path = _ensure_playable_mp4(video_path)

    # Ngưỡng chung (auto từ models nếu override_thr=None)
    if override_thr is None:
        thrs = [info[6] for info in detectors_info]
        thr_global = float(np.mean(thrs)) if len(thrs) > 1 else float(thrs[0])
    else:
        thr_global = float(override_thr)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None, "Không mở được video (codec không hỗ trợ).", "", []
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
    except Exception:
        fps = 25.0
    if w == 0 or h == 0:
        return None, "Video không có kích thước hợp lệ.", "", []

    tmpdir = tempfile.mkdtemp(prefix="df_web_")
    out_path = os.path.join(tmpdir, "out.mp4")
    if HAS_IMAGEIO:
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vout = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    sum_pfake = 0.0; sum_preal = 0.0; n_frames = 0
    pm_sum_fake = np.zeros(len(method_names), dtype=np.float64); n_fake_frames = 0

    det0, tfm0, dev0, *_ = detectors_info[0]
    from .face_detection import crop_faces

    while True:
        try:
            ok, frame_bgr = cap.read()
        except Exception:
            ok = False
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)

        if use_face_crop:
            crops, boxes = crop_faces(pil, max_faces=5)
        else:
            crops, boxes = [pil], [[0, 0, w, h]]

        best = {"p_fake": -1.0, "p_real": -1.0, "box": None, "m_idx": None, "pm": None, "spot": None, "heat": None}
        for crop, box in zip(crops, boxes):
            # dự đoán
            if len(detectors_info) > 1:
                p_fake0, p_real0, pm0 = ensemble_predict_image_pil(detectors_info, crop, tta=tta)
            else:
                p_fake0, p_real0, pm0 = predict_image_tensor(tfm0(crop), det0, dev0, tta=tta)
            m_idx0 = int(np.argmax(pm0)); m_name0 = method_names[m_idx0]; m_conf0 = float(pm0[m_idx0])

            # filter theo method (nếu tự tin)
            p_fake_filt, p_real_filt, pm_filt, crop_use = -1.0, -1.0, None, crop
            if enable_filters and m_conf0 >= float(method_gate):
                crop_f = apply_method_filters(crop, m_name0)
                if len(detectors_info) > 1:
                    p_fake_filt, p_real_filt, pm_filt = ensemble_predict_image_pil(detectors_info, crop_f, tta=tta)
                else:
                    p_fake_filt, p_real_filt, pm_filt = predict_image_tensor(tfm0(crop_f), det0, dev0, tta=tta)
                if p_fake_filt > p_fake0:
                    p_fake0, p_real0, pm0, crop_use = p_fake_filt, p_real_filt, (pm_filt if pm_filt is not None else pm0), crop_f

            if p_fake0 > best["p_fake"]:
                # Bật grad cục bộ để tính saliency
                try:
                    with torch.enable_grad():
                        target = "method" if saliency_mode == "method" else "fake"
                        heat, (sx, sy) = _saliency_map_point(det0, tfm0, crop_use, dev0, target=target, m_idx=m_idx0)
                    spot = (box[0] + sx, box[1] + sy)
                except Exception:
                    heat, spot = None, None
                best.update({"p_fake": p_fake0, "p_real": p_real0, "box": box, "m_idx": m_idx0, "pm": pm0, "spot": spot, "heat": heat})

        n_frames += 1
        sum_pfake += best["p_fake"]; sum_preal += best["p_real"]

        # QUYẾT ĐỊNH: chỉ dùng NGƯỠNG CHUNG
        thr_eff = thr_global
        is_fake = (best["p_fake"] >= thr_eff)

        color = (223, 64, 64) if is_fake else (64, 208, 120)
        if best["box"] is not None:
            if is_fake:
                if best["heat"] is not None:
                    overlay_heatmap_in_box(frame_rgb, best["box"], best["heat"], alpha=0.5)
                    draw_saliency_dots_in_box(frame_rgb, best["box"], best["heat"], density=saliency_density)
                draw_box_with_label_np(frame_rgb, best["box"], method_names[best["m_idx"]], color=color, thickness=int(box_thickness))
                draw_red_spot_np(frame_rgb, best["spot"], radius=14)
            else:
                draw_box_np(frame_rgb, best["box"], color=color, thickness=int(box_thickness))

        if is_fake:
            n_fake_frames += 1
            if best["pm"] is not None:
                pm_sum_fake += best["pm"]

        out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if HAS_IMAGEIO:
            try:
                writer.append_data(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
            except Exception:
                pass
        else:
            vout.write(out_bgr)

    cap.release()
    if HAS_IMAGEIO: writer.close()
    else: vout.release()

    pf = sum_pfake / max(1, n_frames); pr = sum_preal / max(1, n_frames)
    fr_bar_html = _render_fake_real_bar(pf, pr)

    method_rows = []
    if n_fake_frames > 0 and pm_sum_fake.sum() > 0:
        pm = pm_sum_fake / pm_sum_fake.sum()
        method_rows = _make_method_table(pm, method_names)

    verdict = f"Frames: {n_frames} | Fake-frames: {n_fake_frames} ({(100.0 * n_fake_frames / max(1, n_frames)):.1f}%)"
    return out_path, verdict, fr_bar_html, method_rows
