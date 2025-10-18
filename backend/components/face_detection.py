import numpy as np
from PIL import Image

# Lazy initialization của MediaPipe Face Detection
_FACE_DET = None
def _lazy_face_det():
    global _FACE_DET
    if _FACE_DET is None:
        try:
            import mediapipe as mp
            _FACE_DET = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.3
            )
        except Exception:
            _FACE_DET = False
    return _FACE_DET

def detect_faces_xyxy(img_rgb):
    det = _lazy_face_det()
    H, W = img_rgb.shape[:2]
    boxes = []
    if not det:
        return boxes
    res = det.process(img_rgb)  # MediaPipe nhận RGB
    if not res or not res.detections:
        return boxes
    for d in res.detections:
        r = d.location_data.relative_bounding_box
        x0 = int(max(0, r.xmin * W)); y0 = int(max(0, r.ymin * H))
        x1 = int(min(W, (r.xmin + r.width) * W)); y1 = int(min(H, (r.ymin + r.height) * H))
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes

def crop_faces(pil_img, max_faces=5, expand=0.25):
    """
    Trả về (crops(list PIL), boxes(list [x0,y0,x1,y1])).
    Nếu không có mặt: trả 1 crop = full image.
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    H, W = img_rgb.shape[:2]
    boxes = detect_faces_xyxy(img_rgb)
    if not boxes:
        return [pil_img], [[0, 0, W, H]]
    out_crops, out_boxes = [], []
    for (x0, y0, x1, y1) in boxes[:max_faces]:
        cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
        w = (x1 - x0); h = (y1 - y0)
        s = int(round(max(w, h) * (1.0 + expand)))
        nx0 = int(max(0, cx - s / 2)); ny0 = int(max(0, cy - s / 2))
        nx1 = int(min(W, cx + s / 2)); ny1 = int(min(H, cy + s / 2))
        crop = Image.fromarray(img_rgb[ny0:ny1, nx0:nx1])
        out_crops.append(crop); out_boxes.append([nx0, ny0, nx1, ny1])
    return out_crops, out_boxes