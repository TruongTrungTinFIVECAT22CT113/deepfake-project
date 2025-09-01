# web.py
# Gradio UI cho Detect / Generate (Detect là chính). Chỉ sửa để chạy mượt, đúng yêu cầu.
import os, io, time, math, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch, torch.nn as nn
import timm
from torchvision import transforms
import gradio as gr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Utils ======
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

CHECK_IMG_DIR = ensure_dir("data/check_img")
CHECK_VID_DIR = ensure_dir("data/check_vid")

# Optional mediapipe face detection (fallback nếu không có => dùng full ảnh)
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
    """
    img_rgb: np.uint8 (H,W,3) RGB
    return: list of [x0,y0,x1,y1]
    """
    det = _lazy_face_det()
    H, W, _ = img_rgb.shape
    boxes = []
    if not det:
        return boxes
    res = det.process(img_rgb[..., ::-1])  # mediapipe expects BGR
    if not res or not res.detections:
        return boxes
    for d in res.detections:
        r = d.location_data.relative_bounding_box
        x0 = int(max(0, r.xmin * W))
        y0 = int(max(0, r.ymin * H))
        x1 = int(min(W, (r.xmin + r.width) * W))
        y1 = int(min(H, (r.ymin + r.height) * H))
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes

def crop_faces(pil_img, max_faces=5, expand=0.25):
    """
    Trả về (crops(list PIL), boxes(list [x0,y0,x1,y1])).
    Nếu không phát hiện mặt: fallback 1 crop = full image, 1 box full image.
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    H, W = img_rgb.shape[:2]
    boxes = detect_faces_xyxy(img_rgb)
    if not boxes:
        return [pil_img], [[0, 0, W, H]]

    # Sắp xếp theo diện tích lớn → nhỏ, giới hạn max_faces
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:max_faces]
    crops = []
    out_boxes = []
    for (x0,y0,x1,y1) in boxes:
        # expand
        dx = int((x1-x0) * expand)
        dy = int((y1-y0) * expand)
        xx0 = max(0, x0-dx); yy0 = max(0, y0-dy)
        xx1 = min(W, x1+dx); yy1 = min(H, y1+dy)
        crops.append(Image.fromarray(img_rgb[yy0:yy1, xx0:xx1]))
        out_boxes.append([xx0, yy0, xx1, yy1])
    return crops, out_boxes

# ====== Model ======
class MultiHeadViT(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_224", img_size=224, num_methods=6):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, img_size=img_size)
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(0.1)
        self.head_bin = nn.Linear(feat_dim, 2)       # fake/real
        self.head_m  = nn.Linear(feat_dim, num_methods)  # method classes

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head_bin(feat), self.head_m(feat)

def _remap_heads(state):
    """Remap tên head trong ckpt => khớp model hiện tại."""
    out = {}
    for k,v in state.items():
        if k.startswith("head_cls."):
            out[k.replace("head_cls.", "head_bin.")] = v
        elif k.startswith("head_mth."):
            out[k.replace("head_mth.", "head_m.")] = v
        else:
            out[k] = v
    return out

def load_detector(ckpt_path="deepfake_detector/checkpoints/detector_best_calib.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    classes = meta.get("classes", ["fake","real"])
    method_names = meta.get("method_names", ["Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures","Other"])
    mean = meta.get("norm_mean", [0.5,0.5,0.5])
    std  = meta.get("norm_std",  [0.5,0.5,0.5])
    thr  = float(meta.get("threshold", 0.5))
    model_name = meta.get("model_name", "vit_base_patch16_224")
    img_size = int(meta.get("img_size", 224))

    model = MultiHeadViT(model_name, img_size=img_size, num_methods=len(method_names))

    state = ckpt.get("model_ema") or ckpt.get("model") or ckpt
    state = _remap_heads(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict missing={list(missing)} unexpected={list(unexpected)}")

    model.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return model, tfm, classes, method_names, img_size, thr

DETECTOR, DET_TFM, CLASSES, METHOD_NAMES, IMG_SIZE, DET_THR = load_detector()

# ====== Inference helpers ======
@torch.no_grad()
def predict_image_tensor(x_chw, tta=2):
    """
    x_chw: torch.FloatTensor (C,H,W) đã normalize
    return: p_fake(float), p_real(float), pmth(np.ndarray length = num_methods)
    """
    xb = x_chw.unsqueeze(0).to(DEVICE, non_blocking=True)
    use_amp = (DEVICE.type == "cuda")
    if use_amp:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            lb, lm = DETECTOR(xb)
            if tta >= 2:
                lb2, lm2 = DETECTOR(torch.flip(xb, dims=[3]))
                lb = (lb + lb2) / 2
                lm = (lm + lm2) / 2
    else:
        lb, lm = DETECTOR(xb)
        if tta >= 2:
            lb2, lm2 = DETECTOR(torch.flip(xb, dims=[3]))
            lb = (lb + lb2) / 2
            lm = (lm + lm2) / 2

    pbin = torch.softmax(lb, dim=1).squeeze(0).detach().cpu().numpy()  # shape (2,)
    pmth = torch.softmax(lm, dim=1).squeeze(0).detach().cpu().numpy()  # shape (M,)
    # pbin[0] = P(fake), pbin[1] = P(real)
    return float(pbin[0]), float(pbin[1]), pmth

def draw_box_np(img_rgb, box, color=(255, 0, 0), thickness=3):
    x0,y0,x1,y1 = map(int, box)
    cv2.rectangle(img_rgb, (x0,y0), (x1,y1), color, thickness)

def analyze_image(pil_img, use_face_crop=True, override_thr=None, tta=2, box_thickness=3):
    """
    Trả về: (pil_img_annotated, verdict_text)
    - Nếu "Không có dấu hiệu deepfake": không vẽ khung.
    - Nếu "Có dấu hiệu deepfake": vẽ khung đỏ lớn quanh mặt bị flag.
    """
    img = pil_img.convert("RGB")

    if use_face_crop:
        crops, boxes = crop_faces(img, max_faces=5)
    else:
        W,H = img.size
        crops, boxes = [img], [[0,0,W,H]]

    thr = float(override_thr) if (override_thr is not None and not math.isnan(override_thr)) else float(DET_THR)
    img_rgb = np.array(img).copy()
    best = {"p_fake": -1.0, "p_real": -1.0, "box": None, "m_idx": None}

    for crop, box in zip(crops, boxes):
        x = DET_TFM(crop)
        p_fake, p_real, pm = predict_image_tensor(x, tta=tta)
        if p_fake > best["p_fake"]:
            best.update({"p_fake": p_fake, "p_real": p_real, "box": box, "m_idx": int(np.argmax(pm))})

    if best["p_fake"] >= thr:
        # Có dấu hiệu -> vẽ khung
        draw_box_np(img_rgb, best["box"], color=(255,0,0), thickness=int(box_thickness))
        verdict = f"Có dấu hiệu deepfake — fake {best['p_fake']*100:.1f}% / real {best['p_real']*100:.1f}% | Loại: {METHOD_NAMES[best['m_idx']]}"
        # Lưu kết quả
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(CHECK_IMG_DIR, f"det_{ts}.png")
        Image.fromarray(img_rgb).save(out_path)
    else:
        # Không khoanh, chỉ in tỉ lệ
        verdict = f"Không có dấu hiệu deepfake — real {best['p_real']*100:.1f}% / fake {best['p_fake']*100:.1f}%"

    return Image.fromarray(img_rgb), verdict

def analyze_video(video_path, use_face_crop=True, override_thr=None, tta=2, box_thickness=3):
    """
    Xử lý từng frame, không tua nhanh. Trả đường dẫn video đã annotate.
    Chỉ vẽ khung khi vượt ngưỡng. Lưu vào data/check_vid.
    """
    thr = float(override_thr) if (override_thr is not None and not math.isnan(override_thr)) else float(DET_THR)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Không mở được video."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(CHECK_VID_DIR, f"det_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_file, fourcc, fps, (w, h))

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)

        # faces or full
        if use_face_crop:
            crops, boxes = crop_faces(pil, max_faces=5)
        else:
            crops, boxes = [pil], [[0,0,w,h]]

        # choose best
        best = {"p_fake": -1.0, "p_real": -1.0, "box": None, "m_idx": None}
        for crop, box in zip(crops, boxes):
            x = DET_TFM(crop)
            p_fake, p_real, pm = predict_image_tensor(x, tta=tta)
            if p_fake > best["p_fake"]:
                best.update({"p_fake": p_fake, "p_real": p_real, "box": box, "m_idx": int(np.argmax(pm))})

        # draw if fake
        if best["p_fake"] >= thr and best["box"] is not None:
            draw_box_np(frame_rgb, best["box"], color=(255,0,0), thickness=int(box_thickness))

        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    cap.release(); writer.release()
    verdict = f"Video đã phân tích: lưu {out_file}"
    return out_file, verdict

# ====== Gradio UI ======
def _img_wrap(img: Image.Image, fc: bool, thr: float, tta: int, thick: int):
    if img is None:
        return None, "Chưa chọn ảnh."
    out_img, verdict = analyze_image(img, use_face_crop=bool(fc), override_thr=float(thr), tta=int(tta), box_thickness=int(thick))
    return out_img, verdict

def _vid_wrap(vid_path: str, fc: bool, thr: float, tta: int, thick: int):
    if not vid_path:
        return None, "Chưa chọn video."
    out_path, verdict = analyze_video(vid_path, use_face_crop=bool(fc), override_thr=float(thr), tta=int(tta), box_thickness=int(thick))
    return out_path, verdict

with gr.Blocks(title="Deepfake Detect") as demo:
    gr.Markdown("## Deepfake Detect (UI)")
    with gr.Row():
        with gr.Column():
            face_crop = gr.Checkbox(label="Face crop", value=True)
            thr = gr.Slider(0.0, 0.99, value=float(DET_THR), step=0.005, label=f"Ngưỡng kết luận (default từ ckpt={DET_THR:.3f})")
            tta = gr.Slider(1, 4, value=2, step=1, label="TTA (1 hoặc 2/3/4)")
            thick = gr.Slider(1, 8, value=3, step=1, label="Độ dày khung")
        with gr.Column():
            gr.Markdown(f"**Model**: {type(DETECTOR.backbone).__name__} | **img={IMG_SIZE}** | **Classes**: {CLASSES} | **Methods**: {METHOD_NAMES}")

    gr.Markdown("### Detect ảnh")
    with gr.Row():
        in_img = gr.Image(type="pil", label="Ảnh nguồn")
        out_img = gr.Image(type="pil", label="Ảnh đã phân tích")
    img_text = gr.Textbox(label="Kết luận", interactive=False)
    btn_img = gr.Button("Phân tích Ảnh")
    btn_img.click(_img_wrap, [in_img, face_crop, thr, tta, thick], [out_img, img_text])

    gr.Markdown("### Detect video")
    with gr.Row():
        in_vid = gr.Video(label="Video nguồn")
        out_vid = gr.Video(label="Video đã phân tích")
    vid_text = gr.Textbox(label="Nhật ký", interactive=False)
    btn_vid = gr.Button("Phân tích Video")
    btn_vid.click(_vid_wrap, [in_vid, face_crop, thr, tta, thick], [out_vid, vid_text])

demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
