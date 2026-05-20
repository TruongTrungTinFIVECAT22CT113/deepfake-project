# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, tempfile, time
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

from .utils import draw_box_with_label_np, render_verdict_text
from .face_detection import (
    crop_largest_face,
    retina_detect_batch,
    _try_init_retinaface,
    _try_init_mediapipe,
    _mp_detect_all,
    _square_crop_from_bbox,
    _retina_det_thr,
)
from .xai_cam import generate_cam_vit, generate_cam_cnn, generate_cam_swin, overlay_heatmap_on_bgr

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
ENSEMBLE_THR_DEFAULT = 0.58

BASE_BATCH_SIZE = 64
XAI_BATCH_SIZE  = 48

# Số frame đọc vào RAM trước khi detect + infer cùng lúc.
# Tăng nếu GPU VRAM > 8 GB, giảm nếu RAM thấp.
READ_AHEAD = 128


def build_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def _ensemble_predict_batch(detectors_info, crops_bgr, tx, method_names):
    if not crops_bgr:
        return [], [], None

    device = detectors_info[0]["device"]
    batch_tensors = torch.stack([tx(crop) for crop in crops_bgr]).to(device)

    # Tự động ép FP16 nếu model đang ở FP16
    first_param = next(detectors_info[0]["model"].parameters())
    if batch_tensors.dtype == torch.float32 and first_param.dtype == torch.float16:
        batch_tensors = batch_tensors.half()

    p_fake_list = []
    pm_list = []

    for info in detectors_info:
        lb, lm, *_ = info["model"](batch_tensors)
        pbin = torch.softmax(lb, dim=1)
        p_fake_list.append(pbin[:, 0])
        if method_names:
            pm_list.append(torch.softmax(lm, dim=1))

    p_fake_avg = torch.mean(torch.stack(p_fake_list), dim=0).cpu().numpy()
    pm_avg = torch.mean(torch.stack(pm_list), dim=0).cpu().numpy() if pm_list else None

    return p_fake_avg.tolist(), None, pm_avg


def _guess_cnn_target_layer(backbone: nn.Module):
    """
    Đoán target layer cho GradCAM CNN — giữ nguyên logic bản gốc:
      - ConvNeXt: stages[-1][-1]  (check hasattr trước, không check class name)
      - EfficientNet: blocks[-2]  (không phải [-1], để map lan rộng hơn)
      - ResNet-like: layer4 / layers4 / block4 / stage4
      - Fallback: Conv2d cuối cùng
    """
    # 1) ConvNeXt: check hasattr("stages") — không dùng class name vì timm đặt tên khác nhau
    if hasattr(backbone, "stages"):
        try:
            stages = backbone.stages
            if hasattr(stages, "__len__") and len(stages) > 0:
                last_stage = stages[-1]
                if hasattr(last_stage, "__len__") and len(last_stage) > 0:
                    return last_stage[-1]
                return last_stage
        except Exception:
            pass

    # 2) EfficientNet / RegNet: có .blocks
    if hasattr(backbone, "blocks"):
        try:
            blocks = backbone.blocks
            if hasattr(blocks, "__len__") and len(blocks) > 0:
                cls_name = backbone.__class__.__name__.lower()
                if "efficientnet" in cls_name:
                    # Ưu tiên conv_head (Conv2d 1×1 sau blocks, spatial 12×12).
                    # Discriminative nhất, tránh artifact của blocks[-2] (resolution quá nhỏ + nhiễu).
                    if hasattr(backbone, "conv_head"):
                        return backbone.conv_head
                    # Fallback: blocks[-1] tốt hơn blocks[-2]
                    idx = len(blocks) - 1
                else:
                    idx = len(blocks) - 1
                last_block = blocks[idx]
                if hasattr(last_block, "__len__") and len(last_block) > 0:
                    return last_block[-1]
                return last_block
        except Exception:
            pass

    # 3) ResNet-like
    for attr in ["layer4", "layers4", "block4", "stage4"]:
        if hasattr(backbone, attr):
            try:
                layer = getattr(backbone, attr)
                if hasattr(layer, "__len__") and len(layer) > 0:
                    return layer[-1]
                return layer
            except Exception:
                pass

    # 4) Fallback: Conv2d cuối cùng
    last_conv = None
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def _detect_faces_batch(
    bgr_frames: List[np.ndarray],
    backend: str,
    device_type: str,
    det_thr: float,
    det_size: int,
    bbox_scale: float,
    allow_fallback: bool,
) -> List[Optional[Tuple[int, int, int, int, np.ndarray]]]:
    """
    Detect mặt cho một batch frame cùng lúc.
    Trả về list: mỗi phần tử là (x1,y1,x2,y2, crop_bgr) hoặc None nếu không detect được.
    """
    backend = (backend or "retinaface").strip().lower()
    n = len(bgr_frames)
    results: List[Optional[Tuple]] = [None] * n

    if backend == "retinaface":
        _try_init_retinaface(device_type=device_type, det_size=det_size, det_thr=det_thr)
        bboxes = retina_detect_batch(bgr_frames)          # true batch (hoặc fallback serial)
    else:
        # MediaPipe: không có batch API → chạy serial (nhanh hơn RetinaFace serial vì model nhỏ hơn)
        _try_init_mediapipe()
        bboxes = []
        for bgr in bgr_frames:
            dets = _mp_detect_all(bgr)
            if dets:
                dets.sort(key=lambda t: (t[2] - t[0]) * (t[3] - t[1]), reverse=True)
                x1, y1, x2, y2, _ = dets[0]
                bboxes.append((x1, y1, x2, y2))
            else:
                bboxes.append(None)

    for idx, (bgr, bb) in enumerate(zip(bgr_frames, bboxes)):
        if bb is None:
            if allow_fallback:
                # Thử backend kia
                try:
                    pil = Image.fromarray(bgr[:, :, ::-1].copy())
                    _, box, _ = crop_largest_face(
                        pil,
                        backend=("mediapipe" if backend == "retinaface" else "retinaface"),
                        device=device_type, det_thr=det_thr, det_size=det_size,
                        bbox_scale=bbox_scale, allow_fallback=False,
                    )
                    x1, y1, x2, y2 = map(int, box)
                    crop = bgr[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        results[idx] = (x1, y1, x2, y2, crop)
                except Exception:
                    pass
            continue

        crop_info = _square_crop_from_bbox(bgr, bb, scale=bbox_scale)
        if crop_info is None:
            continue
        crop_bgr, box_scaled = crop_info
        h_f, w_f = bgr.shape[:2]
        x1 = max(0, box_scaled[0]); y1 = max(0, box_scaled[1])
        x2 = min(w_f, box_scaled[2]); y2 = min(h_f, box_scaled[3])
        crop_clipped = bgr[y1:y2, x1:x2]
        if crop_clipped.size > 0:
            results[idx] = (x1, y1, x2, y2, crop_clipped)

    return results


def analyze_video(
    video_path: str,
    detectors_info: List[dict],
    method_names: List[str],
    fe_thr_override: Optional[float],
    detector_backend: str = "retinaface",
    bbox_scale: float = 1.10,
    det_thr: float = 0.5,
    box_thickness: int = 3,
    allow_fallback: bool = False,
    xai_mode: str = "none",
    xai_primary_index: Optional[int] = None,
):
    if not detectors_info:
        return None, "No enabled model.", {}, ""

    t0 = time.perf_counter()   # bắt đầu đo thời gian xử lý toàn bộ video

    if len(detectors_info) == 1:
        thr_used = float(fe_thr_override) if fe_thr_override is not None else float(detectors_info[0].get("best_thr", 0.5))
        img_size = int(detectors_info[0].get("img_size", 384))
    else:
        thr_used = ENSEMBLE_THR_DEFAULT
        img_size = int(detectors_info[0].get("img_size", 384))

    tx = build_eval_transform(img_size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    tmpdir = tempfile.mkdtemp(prefix="df_web_")
    out_path = os.path.join(tmpdir, "out.mp4")

    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7)
        use_imageio = True
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        use_imageio = False

    frames_total = fake_frames = 0
    mnames = list(method_names or [])
    m_count = {m: 0 for m in mnames}
    backend_used_final = None
    frame_tags: List[str] = []

    enable_xai = (xai_mode or "none").lower() in ("single", "full")
    batch_size = XAI_BATCH_SIZE if enable_xai else BASE_BATCH_SIZE
    # READ_AHEAD phải ≥ batch_size để lấp đầy batch tốt nhất
    read_ahead = max(READ_AHEAD, batch_size)

    # ── Setup XAI ──────────────────────────────────────────────
    primary_info = primary_model = primary_device = None
    is_vit = is_swin = False
    cnn_target_layer = None

    if enable_xai and detectors_info:
        if xai_primary_index is not None and 0 <= xai_primary_index < len(detectors_info):
            primary_info = detectors_info[xai_primary_index]
        else:
            for info in detectors_info:
                if str(info.get("arch_type", "")).lower() in ("vit", "swin"):
                    primary_info = info
                    break
            if not primary_info:
                primary_info = detectors_info[0]

        primary_model  = primary_info["model"]
        primary_device = primary_info["device"]
        arch_type = str(primary_info.get("arch_type", "cnn")).lower()
        if arch_type == "vit":
            is_vit = True
        elif arch_type == "swin":
            is_swin = True
        else:
            cnn_target_layer = _guess_cnn_target_layer(
                getattr(primary_model, "backbone", primary_model)
            )

    device_type_str = str(detectors_info[0]["device"].type)

    # ── Vòng lặp chính: đọc READ_AHEAD frame → detect batch → infer batch ──
    def _read_chunk(cap, n) -> List[np.ndarray]:
        """Đọc tối đa n frame từ cap, trả về list BGR."""
        chunk = []
        for _ in range(n):
            ret, frm = cap.read()
            if not ret:
                break
            chunk.append(frm)
        return chunk

    while True:
        chunk_bgr = _read_chunk(cap, read_ahead)
        if not chunk_bgr:
            break

        frames_total += len(chunk_bgr)

        # ── 1. Detect mặt cho cả chunk (true batch với RetinaFace) ──
        det_results = _detect_faces_batch(
            chunk_bgr,
            backend=detector_backend,
            device_type=device_type_str,
            det_thr=det_thr,
            det_size=640,
            bbox_scale=bbox_scale,
            allow_fallback=allow_fallback,
        )
        if backend_used_final is None:
            backend_used_final = detector_backend  # ghi nhận lần đầu

        # ── 2. Tách frame có mặt / không có mặt ──
        # face_items: (chunk_idx, frame_bgr, x1,y1,x2,y2, crop_bgr)
        face_items = []
        no_face_indices = []
        for ci, (frm, det) in enumerate(zip(chunk_bgr, det_results)):
            if det is None:
                no_face_indices.append(ci)
            else:
                x1, y1, x2, y2, crop = det
                face_items.append((ci, frm, x1, y1, x2, y2, crop))

        # ── 3. Inference batch theo batch_size ──
        # Kết quả: infer_results[ci] = (p_fake, label, p_methods_row)
        infer_results: Dict[int, Tuple[float, str, Optional[np.ndarray]]] = {}

        for b_start in range(0, len(face_items), batch_size):
            b_items = face_items[b_start : b_start + batch_size]
            crops = [item[6] for item in b_items]
            p_fakes, _, p_methods = _ensemble_predict_batch(detectors_info, crops, tx, mnames)

            for rel_i, item in enumerate(b_items):
                ci = item[0]
                pf = float(p_fakes[rel_i])
                is_fake = pf >= thr_used
                if is_fake:
                    if mnames and p_methods is not None:
                        label = mnames[int(np.argmax(p_methods[rel_i]))]
                    else:
                        label = "Fake"
                else:
                    label = "Real"
                pm_row = p_methods[rel_i] if p_methods is not None else None
                infer_results[ci] = (pf, label, pm_row)

        # ── 4. Render từng frame theo thứ tự gốc ──
        for ci, frm in enumerate(chunk_bgr):
            orig_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

            if ci in no_face_indices:
                frame_tags.append("Real")
                if use_imageio:
                    writer.append_data(orig_rgb)
                else:
                    writer.write(frm)
                continue

            pf, label, pm_row = infer_results.get(ci, (0.0, "Real", None))
            is_fake = label != "Real"

            if is_fake:
                fake_frames += 1
                m_count[label] = m_count.get(label, 0) + 1

            frame_tags.append(label)

            # Lấy lại thông tin detect để vẽ box và XAI
            det = det_results[ci]
            x1, y1, x2, y2, crop_bgr_item = det  # type: ignore

            draw_box_with_label_np(
                orig_rgb, [x1, y1, x2, y2], label,
                color=(223, 64, 64) if is_fake else (64, 208, 120),
                thickness=int(box_thickness),
            )

            # ── XAI ──
            if enable_xai and primary_model is not None and crop_bgr_item is not None:
                try:
                    x_tensor = tx(crop_bgr_item).unsqueeze(0).to(primary_device).float()

                    _was_half = next(primary_model.parameters()).dtype == torch.float16
                    if _was_half:
                        primary_model.float()
                    try:
                        if is_vit:
                            _is_beit = "beit" in str(primary_info.get("model_name", "")).lower()
                            cam = generate_cam_vit(
                                primary_model, x_tensor, target_index=0,
                                device=str(primary_device), smooth_output=_is_beit,
                            )
                        elif is_swin:
                            cam = generate_cam_swin(
                                primary_model, x_tensor, target_index=0,
                                device=str(primary_device),
                            )
                        elif cnn_target_layer is not None:
                            _is_eff = "efficientnet" in str(primary_info.get("model_name", "")).lower()
                            cam = generate_cam_cnn(
                                primary_model, x_tensor, target_index=0,
                                target_layer=cnn_target_layer,
                                device=str(primary_device),
                                extra_smooth=_is_eff, smooth_samples=5,
                            )
                        else:
                            cam = None
                    finally:
                        if _was_half:
                            primary_model.half()

                    if cam is not None and (x2 - x1) > 0 and (y2 - y1) > 0:
                        scale = max(0.08, min(1.0, float(pf)))
                        heat = overlay_heatmap_on_bgr(crop_bgr_item, cam * scale, alpha=0.65)
                        heat = heat.astype(np.uint8) if heat.dtype != np.uint8 else heat
                        heat_resized = cv2.resize(heat, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                        out_bgr_tmp = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
                        out_bgr_tmp[y1:y2, x1:x2] = heat_resized
                        orig_rgb = cv2.cvtColor(out_bgr_tmp, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"[XAI] Error: {type(e).__name__} - {e}")

            out_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
            if use_imageio:
                writer.append_data(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
            else:
                writer.write(out_bgr)

    cap.release()
    if use_imageio:
        writer.close()
    else:
        writer.release()

    duration_sec = frames_total / fps if fps > 0 else 0.0
    processing_time_sec = time.perf_counter() - t0
    verdict = render_verdict_text(frames_total, fake_frames)

    method_rows_total = []
    if mnames and frames_total > 0:
        counts = np.array([m_count.get(m, 0) for m in mnames], dtype=float)
        perc = 100.0 * counts / frames_total
        idx = np.argsort(-perc)
        method_rows_total = [(mnames[i], float(perc[i])) for i in idx]

    stats = {
        "frames_total": int(frames_total),
        "fake_frames": int(fake_frames),
        "fake_ratio": float(fake_frames / max(1, frames_total)),
        "fps": float(fps),
        "duration_sec": float(duration_sec),
        "processing_time_sec": float(processing_time_sec),
        "threshold_used": float(thr_used),
        "detector_backend_used": backend_used_final or detector_backend,
        "method_distribution": {k: int(v) for k, v in m_count.items()},
        "frame_tags": frame_tags,
    }

    return out_path, verdict, stats, method_rows_total