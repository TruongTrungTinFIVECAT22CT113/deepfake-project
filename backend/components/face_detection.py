# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from PIL import Image

# -------- RetinaFace ----------
_RETINA_READY = False
_RETINA_ERR: Optional[Exception] = None
_retina_app = None
_retina_ctx_id = -1
_retina_det_size = (640, 640)
_retina_det_thr = 0.5

def retina_status() -> Dict[str, Any]:
    return {
        "ready": bool(_RETINA_READY),
        "error": (str(_RETINA_ERR) if _RETINA_ERR else None),
        "det_size": _retina_det_size,
        "det_thr": _retina_det_thr,
    }

def retinaface_available() -> bool:
    """Compat for older imports: True iff RetinaFace is initialized and ready."""
    return bool(_RETINA_READY and _retina_app is not None)

def _try_init_retinaface(device_type: str = "cuda",
                         det_size: int = 640,
                         det_thr: float = 0.5) -> bool:
    global _RETINA_READY, _RETINA_ERR, _retina_app, _retina_ctx_id
    global _retina_det_size, _retina_det_thr
    if _RETINA_READY:
        return True
    try:
        from insightface.app import FaceAnalysis  # type: ignore
        _retina_det_size = (int(det_size), int(det_size))
        _retina_det_thr = float(det_thr)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device_type == "cuda" else ["CPUExecutionProvider"]
        _retina_ctx_id = 0 if device_type == "cuda" else -1
        _retina_app = FaceAnalysis(name="buffalo_l", providers=providers)
        _retina_app.prepare(ctx_id=_retina_ctx_id, det_size=_retina_det_size)
        _RETINA_READY = True
        _RETINA_ERR = None
        return True
    except Exception as e:
        _RETINA_READY = False
        _RETINA_ERR = e
        _retina_app = None
        return False

def _retina_detect_largest(bgr: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    if not _RETINA_READY or _retina_app is None:
        return None
    faces = _retina_app.get(bgr)
    faces = [f for f in faces if getattr(f, "det_score", 1.0) >= _retina_det_thr]
    if not faces:
        return None
    def _area(f): x1,y1,x2,y2 = f.bbox; return (x2-x1)*(y2-y1)
    best = max(faces, key=_area)
    x1,y1,x2,y2 = [int(round(v)) for v in best.bbox]
    return (x1,y1,x2,y2)

# -------- MediaPipe ----------
_MP_READY = False
_MP_ERR = None
_mp_face = None

def _try_init_mediapipe() -> bool:
    global _MP_READY, _MP_ERR, _mp_face
    if _MP_READY: return True
    try:
        import mediapipe as mp  # type: ignore
        _mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        _MP_READY = True
        _MP_ERR = None
        return True
    except Exception as e:
        _MP_READY = False
        _MP_ERR = e
        _mp_face = None
        return False

def _mp_detect_all(bgr: np.ndarray):
    if not _MP_READY or _mp_face is None:
        return []
    H, W = bgr.shape[:2]
    res = _mp_face.process(bgr[:, :, ::-1])  # needs RGB
    out = []
    if res.detections:
        for det in res.detections:
            score = det.score[0] if det.score else 0.0
            box = det.location_data.relative_bounding_box
            x = int(round(box.xmin * W)); y = int(round(box.ymin * H))
            w = int(round(box.width * W)); h = int(round(box.height * H))
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(W, x + max(1, w)); y2 = min(H, y + max(1, h))
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2, float(score)))
    return out

# -------- Utilities ----------
def _square_crop_from_bbox(bgr: np.ndarray, bbox, scale: float = 1.10):
    H, W = bgr.shape[:2]
    if bbox is None: return None
    x1,y1,x2,y2 = bbox
    cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
    side = max(x2-x1, y2-y1) * float(scale)
    nx1 = max(0, int(round(cx - side/2))); ny1 = max(0, int(round(cy - side/2)))
    nx2 = min(W, int(round(cx + side/2))); ny2 = min(H, int(round(cy + side/2)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return bgr[ny1:ny2, nx1:nx2], [nx1, ny1, nx2, ny2]

def crop_largest_face(
    pil_img: Image.Image,
    backend: str = "retinaface",
    device: str = "cuda",
    det_thr: float = 0.5,
    det_size: int = 640,
    bbox_scale: float = 1.10,
    allow_fallback: bool = False,  # default NO fallback to match eval
):
    """
    Try requested backend first. If allow_fallback=True, fallback to the other detector when failed.
    Returns: (crop_pil, box, backend_used)
    Raises: Exception if backend requested fails and allow_fallback=False.
    """
    bgr = np.array(pil_img)[:, :, ::-1].copy()
    backend = (backend or "retinaface").strip().lower()

    if backend == "retinaface":
        ok = _try_init_retinaface(device_type=device, det_size=det_size, det_thr=det_thr)
        if ok:
            bb = _retina_detect_largest(bgr)
            crop = _square_crop_from_bbox(bgr, bb, scale=bbox_scale)
            if crop is not None:
                arr, box = crop
                return Image.fromarray(arr[:, :, ::-1].copy()), box, "retinaface"
        if not allow_fallback:
            reason = str(_RETINA_ERR) if _RETINA_ERR else "no-face-detected"
            raise RuntimeError(f"RetinaFace failed: {reason}")
        # fallback → mediapipe
        if _try_init_mediapipe():
            dets = _mp_detect_all(bgr)
            if dets:
                dets.sort(key=lambda t: (t[2]-t[0])*(t[3]-t[1]), reverse=True)
                x1,y1,x2,y2,_ = dets[0]
                crop = _square_crop_from_bbox(bgr, (x1,y1,x2,y2), bbox_scale)
                if crop is not None:
                    arr, box = crop
                    return Image.fromarray(arr[:, :, ::-1].copy()), box, "mediapipe"
        raise RuntimeError("Detector fallback failed.")

    # backend == mediapipe
    if _try_init_mediapipe():
        dets = _mp_detect_all(bgr)
        if dets:
            dets.sort(key=lambda t: (t[2]-t[0])*(t[3]-t[1]), reverse=True)
            x1,y1,x2,y2,_ = dets[0]
            crop = _square_crop_from_bbox(bgr, (x1,y1,x2,y2), bbox_scale)
            if crop is not None:
                arr, box = crop
                return Image.fromarray(arr[:, :, ::-1].copy()), box, "mediapipe"
    if not allow_fallback:
        raise RuntimeError("MediaPipe failed.")
    # fallback → retinaface
    if _try_init_retinaface(device_type=device, det_size=det_size, det_thr=det_thr):
        bb = _retina_detect_largest(bgr)
        crop = _square_crop_from_bbox(bgr, bb, scale=bbox_scale)
        if crop is not None:
            arr, box = crop
            return Image.fromarray(arr[:, :, ::-1].copy()), box, "retinaface"
    raise RuntimeError("Detector fallback failed.")
