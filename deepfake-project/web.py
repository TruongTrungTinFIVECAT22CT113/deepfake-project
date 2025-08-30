# web.py
import os, time, math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import gradio as gr

import torch
import torch.nn as nn
from torchvision import transforms
import timm

import cv2
import mediapipe as mp
import imageio

# ===================== PATHS =====================
ROOT = os.getcwd()
SAVE_IMG_DIR = os.path.join(ROOT, "data", "check_img")
SAVE_VID_DIR = os.path.join(ROOT, "data", "check_vid")
os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_VID_DIR, exist_ok=True)

# ===================== MODEL =====================
DET_CKPT = "deepfake_detector/checkpoints/detector_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_METHODS = ["Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures","Other"]

class MultiHeadViT(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_224", img_size=224, num_methods=len(DEFAULT_METHODS)):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, img_size=img_size)
        feat_dim = self.backbone.num_features
        self.head_cls = nn.Linear(feat_dim, 2)              # [fake, real]
        self.head_mth = nn.Linear(feat_dim, num_methods)    # method
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head_cls(feat), self.head_mth(feat)

def load_detector():
    ckpt = torch.load(DET_CKPT, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    model_name = meta.get("model_name", "vit_base_patch16_224")
    img_size   = int(meta.get("img_size", 384))
    classes    = meta.get("classes", ["fake","real"])
    method_names = meta.get("method_names", DEFAULT_METHODS)
    mean = meta.get("norm_mean", [0.5,0.5,0.5])
    std  = meta.get("norm_std",  [0.5,0.5,0.5])

    model = MultiHeadViT(backbone_name=model_name, img_size=img_size, num_methods=len(method_names))
    state = ckpt.get("ema") or ckpt.get("model")
    if state is None:
        raise RuntimeError("Checkpoint không có khóa 'ema' hoặc 'model'.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[WARN] state_dict mismatch:",
              "missing:", missing, "| unexpected:", unexpected)

    model.to(DEVICE).eval()
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, tfm, classes, method_names, img_size

DETECTOR, DET_TFM, CLASSES, METHOD_NAMES, IMG_SIZE = load_detector()

@torch.no_grad()
def predict_tensor(x_tensor):
    """TTA flip ngang đơn giản."""
    logits_cls, logits_mth = DETECTOR(x_tensor)
    logits_cls_f, logits_mth_f = DETECTOR(torch.flip(x_tensor, dims=[3]))
    logits_cls = (logits_cls + logits_cls_f) / 2
    logits_mth = (logits_mth + logits_mth_f) / 2
    p_cls = torch.softmax(logits_cls, dim=1)  # [:,0]=fake, [:,1]=real
    p_mth = torch.softmax(logits_mth, dim=1)
    return p_cls, p_mth

# ======== Saliency (model-based suspicion over full face) ========
def compute_saliency_map(img_pil: Image.Image) -> np.ndarray:
    """
    Saliency theo gradient của logit 'fake' w.r.t input.
    Trả về heatmap [H,W] (chuẩn hóa 0..1) ở kích thước IMG_SIZE,
    caller sẽ resize lên hình gốc.
    """
    DETECTOR.eval()
    x = DET_TFM(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)
    logits_cls, _ = DETECTOR(x)
    fake_logit = logits_cls[:, 0].sum()
    DETECTOR.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    fake_logit.backward()
    grad = x.grad.detach().abs().mean(dim=1)[0]           # [H,W]
    g = grad.cpu().numpy()
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g

def pick_topk_from_heat(heat: np.ndarray, box: Tuple[int,int,int,int], top_k:int=20) -> List[Tuple[int,int,float]]:
    """
    Chọn top-k điểm (x,y,score) trong box theo giá trị heatmap.
    heat là [H,W] (đã resize về kích thước frame).
    """
    x1,y1,x2,y2 = box
    x1,x2 = max(0,x1), max(0,x2)
    y1,y2 = max(0,y1), max(0,y2)
    crop = heat[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    flat = crop.reshape(-1)
    if flat.size == 0:
        return []
    k = min(top_k, flat.size)
    idxs = np.argpartition(-flat, k-1)[:k]  # top-k nhanh
    pts = []
    Hc, Wc = crop.shape
    for idx in idxs:
        yy, xx = divmod(int(idx), Wc)
        score = float(crop[yy, xx])
        pts.append((x1 + xx, y1 + yy, score))
    pts.sort(key=lambda t: t[2], reverse=True)
    return pts

# ===================== FACE (MediaPipe FaceMesh) =====================
mp_face = mp.solutions.face_mesh

def landmarks_to_box(landmarks: np.ndarray, w: int, h: int, pad: float=0.08) -> Tuple[int,int,int,int]:
    xs = (landmarks[:,0] * w).astype(int)
    ys = (landmarks[:,1] * h).astype(int)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    dx = int((x2 - x1) * pad); dy = int((y2 - y1) * pad)
    x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
    x2 = min(w-1, x2 + dx); y2 = min(h-1, y2 + dy)
    return x1,y1,x2,y2

def draw_green_box(img_bgr: np.ndarray, box: Tuple[int,int,int,int], thickness:int=2):
    x1,y1,x2,y2 = box
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), thickness)

def draw_red_dots(img_bgr: np.ndarray, pts: List[Tuple[int,int,float]], radius:int=3):
    for (x,y,_) in pts:
        cv2.circle(img_bgr, (int(x),int(y)), radius, (0,0,255), thickness=-1, lineType=cv2.LINE_AA)

# ===================== VERDICT (ngưỡng 50%) =====================
def make_verdict_and_method(p_fake: float, p_real: float, p_mth_vec: Optional[np.ndarray]):
    if p_fake >= 0.5:
        verdict = f"⚠️ Có dấu hiệu deepfake — fake {p_fake*100:.1f}% / real {p_real*100:.1f}%"
        if p_mth_vec is not None:
            mi = int(np.argmax(p_mth_vec))
            mpct = float(p_mth_vec[mi]) * 100.0
            method_txt = f"Loại: {METHOD_NAMES[mi]} ({mpct:.1f}%)"
        else:
            method_txt = "Loại: (không xác định)"
    else:
        verdict = f"✅ Không có dấu hiệu deepfake — real {p_real*100:.1f}% / fake {p_fake*100:.1f}%"
        method_txt = "Loại: (N/A)"
    return verdict, method_txt

# ===================== MODEL PROBS =====================
@torch.no_grad()
def _model_probs(img_pil: Image.Image):
    x = DET_TFM(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    p_cls, p_mth = predict_tensor(x)
    p_fake = float(p_cls[0,0].item())
    p_real = float(p_cls[0,1].item())
    mth = p_mth[0].detach().cpu().numpy()
    return p_fake, p_real, mth

# ===================== IMAGE PIPE =====================
def analyze_image(
    img_pil: Image.Image,
    draw_dots: bool,
    dots_topk: int,
    dot_radius: int,
    box_thickness: int
):
    if img_pil is None:
        return None, {"fake":0.0,"real":0.0}, {}, "Vui lòng chọn ảnh.", ""

    p_fake, p_real, mth_vec = _model_probs(img_pil)
    verdict, method_txt = make_verdict_and_method(p_fake, p_real, mth_vec)

    cls_map = {"fake": p_fake, "real": p_real}
    mth_map = {METHOD_NAMES[i]: float(mth_vec[i]) for i in range(len(METHOD_NAMES))}

    img_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    ann = img_bgr.copy()
    H, W = ann.shape[:2]

    if p_fake >= 0.5:
        heat = compute_saliency_map(img_pil)  # [IMG_SIZE, IMG_SIZE]
        heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)

        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True) as fm:
            res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                for lm in res.multi_face_landmarks:
                    pts = np.array([(p.x, p.y) for p in lm.landmark], dtype=np.float32)
                    box = landmarks_to_box(pts, W, H, pad=0.10)
                    draw_green_box(ann, box, thickness=box_thickness)
                    if draw_dots:
                        topk = pick_topk_from_heat(heat, box, dots_topk)
                        draw_red_dots(ann, topk, radius=dot_radius)

        ts = time.strftime("%Y%m%d_%H%M%S")
        Image.fromarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)).save(
            os.path.join(SAVE_IMG_DIR, f"det_{ts}.png")
        )

    out_pil = Image.fromarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
    return out_pil, cls_map, mth_map, verdict, method_txt

# ===================== VIDEO PIPE =====================
def analyze_video(
    video_path: str,
    analyze_stride: int,
    smooth_win: int,
    saliency_stride: int,
    draw_dots: bool,
    dots_topk: int,
    dot_radius: int,
    box_thickness: int
):
    if not video_path:
        return None, {"fake":0.0,"real":0.0}, {}, "Vui lòng chọn video.", ""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"fake":0.0,"real":0.0}, {}, "Không mở được video.", ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pass 1: ước lượng điểm fake + method theo stride
    fake_scores = []
    mth_collect = []
    frames_rgb = []
    use_buffer = (W*H <= 1280*720)

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if use_buffer:
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if i % analyze_stride == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = DET_TFM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p_cls, p_mth = predict_tensor(x)
            fake_scores.append(float(p_cls[0,0].item()))
            mth_collect.append(p_mth[0].detach().cpu().numpy())
        i += 1
    cap.release()

    if not fake_scores:
        return None, {"fake":0.0,"real":0.0}, {}, "Không có khung hình hợp lệ.", ""

    # EMA smoothing
    alpha = 2/(smooth_win+1)
    ema = 0.0
    smoothed = []
    for v in fake_scores:
        ema = alpha*v + (1-alpha)*ema
        smoothed.append(ema)
    p_fake = float(np.mean(smoothed))
    p_real = 1.0 - p_fake
    mth_mean = np.mean(np.stack(mth_collect, axis=0), axis=0)
    verdict, method_txt = make_verdict_and_method(p_fake, p_real, mth_mean)

    deepfake = (p_fake >= 0.5)

    # Nếu KHÔNG deepfake → không ghi file nào cả, không preview
    if not deepfake:
        cls_map = {"fake": p_fake, "real": p_real}
        mth_map = {METHOD_NAMES[i]: float(mth_mean[i]) for i in range(len(METHOD_NAMES))}
        return None, cls_map, mth_map, verdict, method_txt

    # Deepfake → ghi H.264 vào check_vid và phát trực tiếp
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(SAVE_VID_DIR, f"det_{ts}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # saliency cache
    saliency_cache = None
    cap = None if use_buffer else cv2.VideoCapture(video_path)

    # imageio-ffmpeg (ưu tiên libx264 + yuv420p cho trình duyệt)
    writer = None
    try:
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")
    except Exception:
        writer = imageio.get_writer(out_path, fps=fps, codec="mpeg4", quality=8)

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True) as fm:
        fidx = 0
        while True:
            if use_buffer:
                if fidx >= len(frames_rgb): break
                frame_rgb = frames_rgb[fidx]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame_bgr = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            annotated = frame_bgr.copy()

            res = fm.process(frame_rgb)
            if (fidx % max(1, saliency_stride) == 0):
                img_pil = Image.fromarray(frame_rgb)
                heat = compute_saliency_map(img_pil)
                saliency_cache = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)

            if res.multi_face_landmarks and saliency_cache is not None:
                for lm in res.multi_face_landmarks:
                    pts = np.array([(p.x, p.y) for p in lm.landmark], dtype=np.float32)
                    box = landmarks_to_box(pts, W, H, pad=0.10)
                    draw_green_box(annotated, box, thickness=box_thickness)
                    if draw_dots:
                        topk = pick_topk_from_heat(saliency_cache, box, dots_topk)
                        draw_red_dots(annotated, topk, radius=dot_radius)

            # ghi RGB cho imageio
            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            fidx += 1

    if cap is not None: cap.release()
    writer.close()

    # Trả ra đường dẫn để Gradio phát trực tiếp
    # (dùng slash chuẩn để tránh vấn đề backslash trên Windows)
    out_path_norm = out_path.replace("\\", "/")

    cls_map = {"fake": p_fake, "real": p_real}
    mth_map = {METHOD_NAMES[i]: float(mth_mean[i]) for i in range(len(METHOD_NAMES))}
    return out_path_norm, cls_map, mth_map, verdict, method_txt

# ===================== GRADIO UI =====================
with gr.Blocks(title="Deepfake Detector") as demo:
    gr.Markdown("## 🔍 Deepfake Detector — Ảnh & Video")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Tuỳ chỉnh")
            analyze_stride = gr.Slider(1, 10, 5, step=1, label="Phân tích mỗi N khung (video)")
            smooth_win     = gr.Slider(1, 21, 5, step=2, label="EMA smoothing (video)")
            saliency_stride= gr.Slider(1, 20, 5, step=1, label="Tính saliency mỗi N khung (video)")
            draw_dots      = gr.Checkbox(True, label="Chấm điểm nghi ngờ (đỏ) theo saliency của model")
            dots_topk      = gr.Slider(5, 50, 20, step=1, label="Số điểm nghi ngờ tối đa")
            dot_radius     = gr.Slider(2, 6, 3, step=1, label="Bán kính chấm đỏ")
            box_thickness  = gr.Slider(1, 6, 2, step=1, label="Độ dày khung xanh")

        with gr.Column():
            gr.Markdown("### Nguồn")
            img_in = gr.Image(type="pil", label="Ảnh")
            vid_in = gr.Video(label="Hoặc Video", format="mp4")

    with gr.Row():
        btn_img = gr.Button("Phân tích Ảnh", variant="primary")
        btn_vid = gr.Button("Phân tích Video", variant="primary")

    with gr.Row():
        out_img = gr.Image(type="pil", label="Ảnh đã phân tích")
        out_vid = gr.Video(label="Video đã phân tích (phát trực tiếp)", format="mp4")
    with gr.Row():
        out_label_img = gr.Label(label="Kết quả Ảnh (fake/real)")
        out_label_mth_img = gr.Label(label="Loại deepfake (Ảnh)")
    with gr.Row():
        out_label_vid = gr.Label(label="Kết quả Video (fake/real)")
        out_label_mth_vid = gr.Label(label="Loại deepfake (Video)")
    verdict = gr.Textbox(label="Nhận định", interactive=False)
    method_txt = gr.Textbox(label="Phân loại kỹ thuật", interactive=False)

    def _img_wrap(img, draw, k, r, thick):
        anno, cls_map, mth_map, text, mtxt = analyze_image(img, bool(draw), int(k), int(r), int(thick))
        return anno, cls_map, mth_map, text, mtxt

    def _vid_wrap(vid, stride, smooth, sstride, draw, k, r, thick):
        path, cls_map, mth_map, text, mtxt = analyze_video(
            vid, int(stride), int(smooth), int(sstride), bool(draw), int(k), int(r), int(thick)
        )
        return path, cls_map, mth_map, text, mtxt

    btn_img.click(
        fn=_img_wrap,
        inputs=[img_in, draw_dots, dots_topk, dot_radius, box_thickness],
        outputs=[out_img, out_label_img, out_label_mth_img, verdict, method_txt]
    )
    btn_vid.click(
        fn=_vid_wrap,
        inputs=[vid_in, analyze_stride, smooth_win, saliency_stride, draw_dots, dots_topk, dot_radius, box_thickness],
        outputs=[out_vid, out_label_vid, out_label_mth_vid, verdict, method_txt]
    )

if __name__ == "__main__":
    print(f"Classes: {CLASSES} | Methods: {METHOD_NAMES} | Model: {type(DETECTOR.backbone).__name__} | img_size={IMG_SIZE}")
    demo.launch()
