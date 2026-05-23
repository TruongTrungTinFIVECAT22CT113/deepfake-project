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
from .xai_cam import (
    generate_cam_vit, generate_cam_cnn, generate_cam_swin, overlay_heatmap_on_bgr,
    CnnHookContext, generate_cam_cnn_with_ctx,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
ENSEMBLE_THR_DEFAULT = 0.58

BASE_BATCH_SIZE = 48
XAI_BATCH_SIZE  = 24

# Số frame đọc vào RAM trước khi detect + infer cùng lúc.
# Tăng nếu GPU VRAM > 8 GB, giảm nếu RAM thấp.
READ_AHEAD = 64


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
    # batch_tensors ở FP32 — mỗi model tự cast sang dtype của nó trong vòng lặp bên dưới.
    # KHÔNG cast 1 lần theo model[0]: khi XAI đang giữ primary_model ở FP32 trong khi
    # các model khác vẫn FP16, cast chung sẽ gây mismatch dtype → RuntimeError.
    batch_tensors = torch.stack([tx(crop) for crop in crops_bgr]).to(device)  # FP32

    p_fake_list = []
    pm_list = []

    for info in detectors_info:
        first_param = next(info["model"].parameters())
        # Cast tensor sang đúng dtype của từng model
        t = batch_tensors.to(dtype=first_param.dtype) if batch_tensors.dtype != first_param.dtype else batch_tensors
        lb, lm, *_ = info["model"](t)
        # .float() trước softmax: đồng nhất với sweep_ensemble_thr_spatial.py
        # (sweep dùng softmax(lb.float())) — tránh sai lệch FP16 xung quanh biên threshold 0.58
        pbin = torch.softmax(lb.float(), dim=1)
        p_fake_list.append(pbin[:, 0])
        if method_names:
            pm_list.append(torch.softmax(lm.float(), dim=1))

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
    progress_callback=None,   # Callable[[int, int], None] — (frames_done, frames_total_hint)
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
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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
    _cnn_hook_ctx: Optional[CnnHookContext] = None   # hook cache cho CNN
    _xai_model_was_half = False                       # để restore FP16 sau khi xong

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

        # ── Tối ưu tốc độ XAI ──────────────────────────────────────────────
        # 1) Giữ model ở FP32 suốt cả video thay vì flip FP32↔FP16 mỗi frame.
        #    primary_model.float() / .half() mỗi frame tốn ~8-15ms/frame vì
        #    phải copy toàn bộ weight tensor trên GPU.
        # 2) CNN: register hook 1 lần → 0 overhead per-frame.
        #    ViT/Swin: hook đơn giản hơn, overhead nhỏ hơn, giữ nguyên per-call.
        if primary_model is not None:
            _xai_model_was_half = (
                next(primary_model.parameters()).dtype == torch.float16
            )
            if _xai_model_was_half:
                primary_model.float()   # FP32 cho cả video — restore sau khi xong

        if cnn_target_layer is not None:
            _cnn_hook_ctx = CnnHookContext(cnn_target_layer)

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

            # ── XAI — chỉ render trên fake frame ──
            # Real frame không cần heatmap → tiết kiệm ~(1 - fake_ratio) × XAI time
            if enable_xai and is_fake and primary_model is not None and crop_bgr_item is not None:
                try:
                    x_tensor = tx(crop_bgr_item).unsqueeze(0)
                    # Model đã ở FP32 suốt video (convert 1 lần ở setup) — không flip ở đây

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
                    elif _cnn_hook_ctx is not None:
                        # Fast path: hook đã cache, 1 pass, không SmoothGrad
                        _is_eff = "efficientnet" in str(primary_info.get("model_name", "")).lower()
                        cam = generate_cam_cnn_with_ctx(
                            primary_model, x_tensor, _cnn_hook_ctx,
                            target_index=0, device=str(primary_device),
                            extra_smooth=_is_eff,
                        )
                    else:
                        cam = None

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

        # ── Gọi progress callback sau mỗi chunk ──────────────────────────────
        if progress_callback is not None:
            try:
                progress_callback(frames_total, total_frames_hint)
            except Exception:
                pass

    cap.release()
    if use_imageio:
        writer.close()
    else:
        writer.release()

    # ── Cleanup XAI: remove hook cache, restore model dtype ──────────────────
    if _cnn_hook_ctx is not None:
        _cnn_hook_ctx.remove()
    if _xai_model_was_half and primary_model is not None:
        primary_model.half()   # restore FP16 để inference batch tiếp theo không bị chậm

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