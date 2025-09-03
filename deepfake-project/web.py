# web.py — Detect ảnh/video + hiển thị thanh % Fake/Real và bảng % theo kỹ thuật
# - Ảnh: vẽ khung, lưu vào data/check_img/<MethodName> nếu vượt ngưỡng
# - Video: vẽ khung từng frame, xuất MP4 xem trực tiếp trên web; nếu có frame fake thì lưu vào data/check_vid/<MethodName>
# - Thêm thanh % Fake/Real + bảng % theo kỹ thuật cho cả Ảnh & Video

import os, io, time, math, tempfile, shutil
import numpy as np
from PIL import Image
import cv2
import torch, torch.nn as nn
import timm
from torchvision import transforms
import gradio as gr

# ====== Optional: dùng imageio (ffmpeg) để ghi MP4 H.264 cho HTML5 dễ phát ======
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

CHECK_IMG_DIR = ensure_dir("data/check_img")
CHECK_VID_DIR = ensure_dir("data/check_vid")

# ========== Face detection (MediaPipe) ==========
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
    img_rgb: np.uint8 RGB, shape (H,W,3)
    return: list [[x0,y0,x1,y1], ...] (pixel)
    """
    det = _lazy_face_det()
    H, W = img_rgb.shape[:2]
    boxes = []
    if not det:
        return boxes
    # MediaPipe FaceDetection nhận RGB
    res = det.process(img_rgb)
    if not res or not res.detections:
        return boxes
    for d in res.detections:
        r = d.location_data.relative_bounding_box
        x0 = int(max(0, r.xmin * W)); y0 = int(max(0, r.ymin * H))
        x1 = int(min(W, (r.xmin + r.width) * W)); y1 = int(min(H, (r.ymin + r.height) * H))
        if x1 > x0 and y1 > y0:
            boxes.append([x0,y0,x1,y1])
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
        return [pil_img], [[0,0,W,H]]

    boxes = sorted(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)[:max_faces]
    crops, out_boxes = [], []
    for (x0,y0,x1,y1) in boxes:
        dx = int((x1-x0)*expand); dy = int((y1-y0)*expand)
        xx0 = max(0, x0-dx); yy0 = max(0, y0-dy)
        xx1 = min(W, x1+dx); yy1 = min(H, y1+dy)
        crops.append(Image.fromarray(img_rgb[yy0:yy1, xx0:xx1]))
        out_boxes.append([xx0,yy0,xx1,yy1])
    return crops, out_boxes

# ========== Model ==========
class MultiHeadViT(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_224", img_size=224, num_methods=6):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, img_size=img_size)
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(0.1)
        self.head_bin = nn.Linear(feat_dim, 2)           # fake/real
        self.head_m  = nn.Linear(feat_dim, num_methods)  # method

    def forward(self, x):
        f = self.backbone(x)
        f = self.dropout(f)
        return self.head_bin(f), self.head_m(f)

def _remap_heads(state):
    out = {}
    for k,v in state.items():
        if k.startswith("head_cls."):
            out[k.replace("head_cls.","head_bin.")] = v
        elif k.startswith("head_mth."):
            out[k.replace("head_mth.","head_m.")] = v
        else:
            out[k] = v
    return out

def load_detector(ckpt_path="deepfake_detector/checkpoints/detector_best_calib.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    classes = meta.get("classes", ["fake","real"])
    method_names = meta.get("method_names", ["Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures","Other"])
    mean = meta.get("norm_mean", [0.5,0.5,0.5]); std = meta.get("norm_std", [0.5,0.5,0.5])
    thr  = float(meta.get("threshold", 0.5))
    model_name = meta.get("model_name", "vit_base_patch16_224")
    img_size   = int(meta.get("img_size", 224))

    model = MultiHeadViT(model_name, img_size=img_size, num_methods=len(method_names))
    state = ckpt.get("model_ema") or ckpt.get("model") or ckpt
    state = _remap_heads(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict missing= {list(missing)} unexpected= {list(unexpected)}")
    model.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, tfm, classes, method_names, img_size, thr

DETECTOR, DET_TFM, CLASSES, METHOD_NAMES, IMG_SIZE, DET_THR = load_detector()
METHOD_DIRS     = {i: ensure_dir(os.path.join(CHECK_IMG_DIR, n)) for i,n in enumerate(METHOD_NAMES)}
METHOD_DIRS_VID = {i: ensure_dir(os.path.join(CHECK_VID_DIR, n)) for i,n in enumerate(METHOD_NAMES)}

# ========== Inference helpers ==========
@torch.no_grad()
def predict_image_tensor(x_chw, tta=2):
    """
    x_chw: torch.FloatTensor (C,H,W) normalized
    return: p_fake(float), p_real(float), pmth(np.ndarray, len=M)
    """
    xb = x_chw.unsqueeze(0).to(DEVICE, non_blocking=True)
    use_amp = (DEVICE.type == "cuda")
    if use_amp:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            lb, lm = DETECTOR(xb)
            if tta and tta >= 2:
                lb2, lm2 = DETECTOR(torch.flip(xb, dims=[3]))
                lb = (lb + lb2)/2; lm = (lm + lm2)/2
    else:
        lb, lm = DETECTOR(xb)
        if tta and tta >= 2:
            lb2, lm2 = DETECTOR(torch.flip(xb, dims=[3]))
            lb = (lb + lb2)/2; lm = (lm + lm2)/2

    pbin = torch.softmax(lb, dim=1).squeeze(0).cpu().numpy()   # (2,)
    pmth = torch.softmax(lm, dim=1).squeeze(0).cpu().numpy()   # (M,)
    return float(pbin[0]), float(pbin[1]), pmth  # P(fake), P(real), method probs

def draw_box_np(img_rgb, box, color=(255,0,0), thickness=3):
    x0,y0,x1,y1 = map(int, box)
    cv2.rectangle(img_rgb, (x0,y0), (x1,y1), color, thickness)

def _render_fake_real_bar(p_fake, p_real):
    pf = max(0, min(100, int(round(p_fake*100))))
    pr = 100 - pf
    return f"""
<div style="font-family: system-ui, -apple-system, Segoe UI, Roboto; max-width: 680px;">
  <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:6px;">
    <span>Fake: <b>{pf}%</b></span>
    <span>Real: <b>{pr}%</b></span>
  </div>
  <div style="width:100%; height:16px; background:#eee; border-radius:8px; overflow:hidden; border:1px solid #ddd;">
    <div style="width:{pf}%; height:100%; background:#e74c3c; float:left;"></div>
    <div style="width:{pr}%; height:100%; background:#2ecc71; float:left;"></div>
  </div>
</div>
"""

def _make_method_table(pm: np.ndarray):
    order = np.argsort(-pm)
    return [[METHOD_NAMES[i], round(float(pm[i]*100.0), 1)] for i in order]

# ========== Analyze IMAGE ==========
def analyze_image(pil_img, use_face_crop=True, override_thr=None, tta=2, box_thickness=3):
    img = pil_img.convert("RGB")
    if use_face_crop:
        crops, boxes = crop_faces(img, max_faces=5)
    else:
        W,H = img.size
        crops, boxes = [img], [[0,0,W,H]]

    thr = float(override_thr) if (override_thr is not None and not math.isnan(override_thr)) else float(DET_THR)
    img_rgb = np.array(img).copy()
    best = {"p_fake":-1.0, "p_real":-1.0, "box":None, "m_idx":None, "pm":None}

    for crop, box in zip(crops, boxes):
        x = DET_TFM(crop)
        p_fake, p_real, pm = predict_image_tensor(x, tta=tta)
        if p_fake > best["p_fake"]:
            best.update({"p_fake":p_fake, "p_real":p_real, "box":box, "m_idx":int(np.argmax(pm)), "pm":pm})

    fr_bar_html = _render_fake_real_bar(best["p_fake"], best["p_real"])
    method_rows = _make_method_table(best["pm"]) if best["pm"] is not None else []

    if best["p_fake"] >= thr and best["box"] is not None:
        draw_box_np(img_rgb, best["box"], color=(255,0,0), thickness=int(box_thickness))
        order = np.argsort(-best["pm"])
        topk = [f"{METHOD_NAMES[i]}:{best['pm'][i]*100:.1f}%" for i in order[:3]]
        verdict = (
            f"Có dấu hiệu deepfake — fake {best['p_fake']*100:.1f}% / real {best['p_real']*100:.1f}%"
            f" | Loại (best): {METHOD_NAMES[best['m_idx']]}\nTop-3: " + ", ".join(topk)
        )
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = METHOD_DIRS.get(best["m_idx"], CHECK_IMG_DIR)
        Image.fromarray(img_rgb).save(os.path.join(out_dir, f"det_{ts}.png"))
    else:
        verdict = f"Không có dấu hiệu deepfake — real {best['p_real']*100:.1f}% / fake {best['p_fake']*100:.1f}%"

    return Image.fromarray(img_rgb), verdict, fr_bar_html, method_rows

# ========== Video writer helpers ==========
class _VideoWriterImageIO:
    def __init__(self, out_path, fps):
        self.writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)  # H.264
    def write(self, frame_rgb):
        self.writer.append_data(frame_rgb)  # RGB
    def release(self):
        self.writer.close()

class _VideoWriterOpenCV:
    def __init__(self, out_path, fps, w, h):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # fallback (ít phổ biến hơn H.264 nhưng không cần ffmpeg)
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
    def write(self, frame_rgb):
        self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    def release(self):
        self.writer.release()

def _make_writer(path, fps, w, h):
    if HAS_IMAGEIO:
        return _VideoWriterImageIO(path, fps)
    return _VideoWriterOpenCV(path, fps, w, h)

# ========== Analyze VIDEO ==========
def analyze_video(video_path, use_face_crop=True, override_thr=None, tta=2, box_thickness=3):
    thr = float(override_thr) if (override_thr is not None and not math.isnan(override_thr)) else float(DET_THR)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Không mở được video.", "", []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_dir  = tempfile.mkdtemp(prefix="df_tmp_")
    tmp_file = os.path.join(tmp_dir, f"det_{ts}.mp4")

    writer = _make_writer(tmp_file, fps, w, h)

    any_fake = False
    votes = np.zeros(len(METHOD_NAMES), dtype=int)

    # Tổng hợp để hiển thị thanh + bảng
    sum_pfake, sum_preal, n_frames = 0.0, 0.0, 0
    pm_sum_all   = np.zeros(len(METHOD_NAMES), dtype=float)
    pm_sum_fake  = np.zeros(len(METHOD_NAMES), dtype=float)
    n_fake_frames = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)

        if use_face_crop:
            crops, boxes = crop_faces(pil, max_faces=5)
        else:
            crops, boxes = [pil], [[0,0,w,h]]

        best = {"p_fake":-1.0, "p_real":-1.0, "box":None, "m_idx":None, "pm":None}
        for crop, box in zip(crops, boxes):
            x = DET_TFM(crop)
            p_fake, p_real, pm = predict_image_tensor(x, tta=tta)
            if p_fake > best["p_fake"]:
                best.update({"p_fake":p_fake, "p_real":p_real, "box":box, "m_idx":int(np.argmax(pm)), "pm":pm})

        # accumulate stats
        sum_pfake += best["p_fake"]; sum_preal += best["p_real"]; n_frames += 1
        if best["pm"] is not None:
            pm_sum_all += best["pm"]
        if best["p_fake"] >= thr:
            n_fake_frames += 1
            if best["pm"] is not None: pm_sum_fake += best["pm"]

        # draw & write
        if best["p_fake"] >= thr and best["box"] is not None:
            any_fake = True
            votes[best["m_idx"]] += 1
            draw_box_np(frame_rgb, best["box"], color=(255,0,0), thickness=int(box_thickness))

        writer.write(frame_rgb)

    cap.release(); writer.release()

    # Chọn method & nơi lưu (nếu có fake)
    if any_fake:
        m_idx = int(np.argmax(votes))
        out_dir = METHOD_DIRS_VID.get(m_idx, CHECK_VID_DIR)
        ensure_dir(out_dir)
        final_path = os.path.join(out_dir, f"det_{ts}.mp4")
        try: shutil.move(tmp_file, final_path)
        except Exception: shutil.copy2(tmp_file, final_path)
        try: shutil.rmtree(tmp_dir, ignore_errors=True)
        except: pass
        out_path = final_path
        verdict = f"Video: Có dấu hiệu deepfake | Loại: {METHOD_NAMES[m_idx]} | lưu: {final_path}"
    else:
        # không có fake frame nào => vẫn trả video tạm để xem
        out_path = tmp_file
        verdict  = "Video: Không có dấu hiệu deepfake — đã render tạm để xem (không lưu vào check_vid)."

    # Thanh % Fake/Real (lấy trung bình theo frame tốt nhất)
    if n_frames > 0:
        p_fake_avg = sum_pfake / n_frames
        p_real_avg = sum_preal / n_frames
    else:
        p_fake_avg = 0.0; p_real_avg = 1.0
    fr_bar_html = _render_fake_real_bar(p_fake_avg, p_real_avg)

    # Bảng % theo kỹ thuật: ưu tiên thống kê trên các frame được đánh fake; nếu không có, dùng trung bình toàn bộ
    if n_fake_frames > 0:
        pm = pm_sum_fake / max(1, pm_sum_fake.sum()) if pm_sum_fake.sum() > 0 else np.zeros_like(pm_sum_fake)
    else:
        pm = pm_sum_all / max(1, pm_sum_all.sum()) if pm_sum_all.sum() > 0 else np.zeros_like(pm_sum_all)
    method_rows = _make_method_table(pm)

    return out_path, verdict, fr_bar_html, method_rows

# ========== Gradio UI ==========
def _img_wrap(img: Image.Image, fc: bool, thr: float, tta: int, thick: int):
    if img is None:
        return None, "Chưa chọn ảnh.", "", []
    out_img, verdict, fr_bar_html, method_rows = analyze_image(
        img, use_face_crop=bool(fc), override_thr=float(thr), tta=int(tta), box_thickness=int(thick)
    )
    return out_img, verdict, fr_bar_html, method_rows

def _vid_wrap(vid_path: str, fc: bool, thr: float, tta: int, thick: int):
    if not vid_path:
        return None, "Chưa chọn video.", "", []
    out_path, verdict, fr_bar_html, method_rows = analyze_video(
        vid_path, use_face_crop=bool(fc), override_thr=float(thr), tta=int(tta), box_thickness=int(thick)
    )
    return out_path, verdict, fr_bar_html, method_rows

with gr.Blocks(title="Deepfake Detect") as demo:
    gr.Markdown("## Deepfake Detect (UI)")
    with gr.Row():
        with gr.Column():
            face_crop = gr.Checkbox(label="Face crop", value=True)
            thr = gr.Slider(0.0, 0.99, value=float(DET_THR), step=0.005, label=f"Ngưỡng kết luận (default từ ckpt={DET_THR:.3f})")
            tta = gr.Slider(1, 4, value=2, step=1, label="TTA (1 hoặc 2/3/4)")
            thick = gr.Slider(1, 8, value=3, step=1, label="Độ dày khung")
        with gr.Column():
            gr.Markdown(f"**Model**: {type(DETECTOR.backbone).__name__} | **img={IMG_SIZE}**")
            gr.Markdown(f"**Classes**: {CLASSES} | **Methods**: {METHOD_NAMES}")

    gr.Markdown("### Detect ảnh")
    with gr.Row():
        in_img = gr.Image(type="pil", label="Ảnh nguồn")
        out_img = gr.Image(type="pil", label="Ảnh đã phân tích")
    img_text = gr.Textbox(label="Kết luận", interactive=False)
    fr_bar_img = gr.HTML(label="Tỉ lệ Fake/Real")
    method_df_img = gr.Dataframe(headers=["Method", "%"], datatype=["str","number"], interactive=False, label="Tỉ lệ theo phương pháp (descending)")
    btn_img = gr.Button("Phân tích Ảnh")
    btn_img.click(_img_wrap, [in_img, face_crop, thr, tta, thick], [out_img, img_text, fr_bar_img, method_df_img])

    gr.Markdown("### Detect video")
    with gr.Row():
        in_vid = gr.Video(label="Video nguồn")
        out_vid = gr.Video(label="Video đã phân tích (xem trực tiếp)")
    vid_text = gr.Textbox(label="Nhật ký", interactive=False)
    fr_bar_vid = gr.HTML(label="Tỉ lệ Fake/Real (video)")
    method_df_vid = gr.Dataframe(headers=["Method", "%"], datatype=["str","number"], interactive=False, label="Tỉ lệ theo phương pháp (descending)")
    btn_vid = gr.Button("Phân tích Video")
    btn_vid.click(_vid_wrap, [in_vid, face_crop, thr, tta, thick], [out_vid, vid_text, fr_bar_vid, method_df_vid])

demo.launch(server_name="127.0.0.1", server_port=7860, share=False)