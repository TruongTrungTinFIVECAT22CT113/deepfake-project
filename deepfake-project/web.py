# web.py — Detect ảnh/video + hiển thị thanh % Fake/Real, bảng % theo kỹ thuật
# + Hậu xử lý theo từng method (filter) và ngưỡng riêng cho từng method.
# ----------------------------------------------------------------------
# Cách chạy:
#   py web.py --thr 0.75 --per_method_thr "Deepfakes=0.72,Face2Face=0.78,FaceShifter=0.76,FaceSwap=0.78,NeuralTextures=0.62" --method_gate 0.55 --enable_filters 1
#
# UI có ô để đổi per-method threshold, method gate, bật/tắt filters mà không cần restart.

import os, io, time, math, tempfile, argparse, shutil
import numpy as np
from PIL import Image
import cv2
import torch, torch.nn as nn
import timm
from torchvision import transforms
import gradio as gr

# ====================== CLI ======================
parser = argparse.ArgumentParser()
parser.add_argument('--thr', type=float, default=None)
parser.add_argument('--per_method_thr', type=str, default="")  # "Deepfakes=0.72,Face2Face=0.78,..."
parser.add_argument('--method_gate', type=float, default=0.55)
parser.add_argument('--enable_filters', type=int, default=1)
args, _ = parser.parse_known_args()
CLI_THR = args.thr
CLI_PM_THR_STR = args.per_method_thr or ""
CLI_METHOD_GATE = float(args.method_gate)
CLI_ENABLE_FILTERS = bool(args.enable_filters)

# ====== Optional: dùng imageio (ffmpeg) để ghi MP4 cho HTML5) ======
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ====================== Face detection (MediaPipe) ======================
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
    res = det.process(img_rgb)  # mediapipe nhận RGB
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
    out_crops, out_boxes = [], []
    for (x0,y0,x1,y1) in boxes[:max_faces]:
        cx = (x0+x1)/2; cy = (y0+y1)/2
        w  = (x1-x0);   h  = (y1-y0)
        s  = int(round(max(w,h)*(1.0+expand)))
        nx0 = int(max(0, cx - s/2)); ny0 = int(max(0, cy - s/2))
        nx1 = int(min(W, cx + s/2)); ny1 = int(min(H, cy + s/2))
        crop = Image.fromarray(img_rgb[ny0:ny1, nx0:nx1])
        out_crops.append(crop); out_boxes.append([nx0,ny0,nx1,ny1])
    return out_crops, out_boxes

# ====================== Model ======================
class MultiHeadViT(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", num_methods=6, img_size=512, drop_rate=0.0):
        super().__init__()
        # Một số version timm không nhận 'img_size' hoặc 'drop_rate' qua create_model => thử tuần tự.
        backbone = None
        last_err = None
        for kwargs in [
            dict(pretrained=False, num_classes=0, drop_rate=drop_rate, img_size=img_size),
            dict(pretrained=False, num_classes=0, drop_rate=drop_rate),
            dict(pretrained=False, num_classes=0),
        ]:
            try:
                backbone = timm.create_model(model_name, **kwargs)
                break
            except TypeError as e:
                last_err = e
                continue
        if backbone is None:
            raise RuntimeError(f"Cannot create backbone for {model_name}: {last_err}")

        self.backbone = backbone
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(p=0.0)
        self.head_bin = nn.Linear(feat_dim, 2)           # fake/real
        self.head_m  = nn.Linear(feat_dim, num_methods)  # method

    def forward(self, x):
        f = self.backbone(x)
        f = self.dropout(f)
        return self.head_bin(f), self.head_m(f)

def _remap_heads(state):
    out = {}
    for k,v in state.items():
        if k.startswith("head_cls."):       # tương thích checkpoint cũ
            out[k.replace("head_cls.","head_bin.")] = v
        elif k.startswith("head_mth."):
            out[k.replace("head_mth.","head_m.")] = v
        else:
            out[k] = v
    return out

def load_detector(ckpt_path="deepfake_detector/checkpoints/detector_best.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    classes = meta.get("classes", ["fake","real"])
    method_names = meta.get("method_names", ["Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures","Other"])
    mean = meta.get("norm_mean", [0.5,0.5,0.5]); std = meta.get("norm_std", [0.5,0.5,0.5])
    thr  = float(meta.get("threshold", 0.5))
    model_name = meta.get("model_name", "vit_base_patch16_224")
    img_size   = int(meta.get("img_size", 224))

    model = MultiHeadViT(model_name, img_size=img_size, num_methods=len(method_names), drop_rate=0.0)
    state = ckpt.get("model", ckpt)
    state = _remap_heads(state)
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, tfm, classes, method_names, img_size, thr

DETECTOR, DET_TFM, CLASSES, METHOD_NAMES, IMG_SIZE, DET_THR = load_detector()

# ====================== Inference helpers ======================
@torch.no_grad()
def predict_image_tensor(x_chw, tta=2):
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

    pbin = torch.softmax(lb, dim=1).squeeze(0).cpu().numpy()
    pmth = torch.softmax(lm, dim=1).squeeze(0).cpu().numpy()
    return float(pbin[0]), float(pbin[1]), pmth  # P(fake), P(real), method probs

def draw_box_np(img_rgb, box, color=(255,0,0), thickness=3):
    x0,y0,x1,y1 = map(int, box)
    cv2.rectangle(img_rgb, (x0,y0), (x1,y1), color, thickness)

def _render_fake_real_bar(p_fake, p_real):
    pf = max(0, min(100, int(round(p_fake*100))))
    pr = max(0, min(100, int(round(p_real*100))))
    w = 300
    w_f = int(w * pf/100.0); w_r = w - w_f
    html = f"""
    <div style="display:flex;gap:8px;align-items:center">
      <div style="width:{w}px;height:16px;border-radius:8px;overflow:hidden;border:1px solid #999;display:flex">
        <div style="width:{w_f}px;background:#d33"></div>
        <div style="width:{w_r}px;background:#3b7"></div>
      </div>
      <div><b>Fake</b> {pf}% &nbsp;|&nbsp; <b>Real</b> {pr}%</div>
    </div>
    """
    return html

def _make_method_table(pm: np.ndarray):
    order = np.argsort(-pm)
    return [[METHOD_NAMES[i], round(float(pm[i]*100.0), 1)] for i in order]

# ====================== Filters per method ======================
def _unsharp_np(img_rgb, amount=0.6, radius=3):
    blur = cv2.GaussianBlur(img_rgb, (0,0), sigmaX=radius)
    sharp = cv2.addWeighted(img_rgb, 1+amount, blur, -amount, 0)
    return sharp

def _deblock_np(img_rgb, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, h, hColor, templateWindowSize, searchWindowSize)

def _denoise_np(img_rgb, strength=5):
    return cv2.bilateralFilter(img_rgb, d=0, sigmaColor=strength, sigmaSpace=strength)

DEFAULT_FILTERS = {
    "Deepfakes":      ["deblock_light"],
    "Face2Face":      ["unsharp_light"],
    "FaceShifter":    ["unsharp_light"],
    "FaceSwap":       ["unsharp_mid"],
    "NeuralTextures": ["denoise_verylight", "unsharp_verylight"],
}

def apply_method_filters(pil_img, method_name):
    img = np.array(pil_img.convert("RGB"))
    ops = DEFAULT_FILTERS.get(method_name, [])
    out = img
    for op in ops:
        if op == "unsharp_verylight": out = _unsharp_np(out, amount=0.25, radius=1.5)
        elif op == "unsharp_light":   out = _unsharp_np(out, amount=0.45, radius=2.0)
        elif op == "unsharp_mid":     out = _unsharp_np(out, amount=0.7,  radius=2.0)
        elif op == "deblock_light":   out = _deblock_np(out, h=2, hColor=2, templateWindowSize=7, searchWindowSize=21)
        elif op == "denoise_verylight": out = _denoise_np(out, strength=4)
    return Image.fromarray(out)

# ====================== Per-method threshold helpers ======================
def parse_thr_map(s: str, method_names):
    out = {}
    if not s: return out
    for it in s.split(","):
        it = it.strip()
        if not it or "=" not in it: 
            continue
        k, v = it.split("=", 1)
        k = k.strip(); v = v.strip()
        try:
            vf = float(v)
        except:
            continue
        for name in method_names:
            if name.lower() == k.lower():
                out[name] = vf
                break
    return out

def get_thr_for_method(global_thr: float, method_name: str, thr_map: dict, gate_ok: bool):
    if gate_ok and method_name in thr_map:
        return float(thr_map[method_name])
    return float(global_thr)

# ====================== Analyze IMAGE ======================
@torch.no_grad()
def analyze_image(pil_img, use_face_crop=True, override_thr=None, tta=2, box_thickness=3,
                  per_method_thr_map=None, method_gate=0.55, enable_filters=True):
    img = pil_img.convert("RGB")
    if use_face_crop:
        crops, boxes = crop_faces(img, max_faces=5)
    else:
        W, H = img.size
        crops, boxes = [img], [[0, 0, W, H]]

    thr_global = float(override_thr) if (override_thr is not None and not math.isnan(override_thr)) else float(DET_THR)
    thr_map = per_method_thr_map or {}
    img_rgb = np.array(img).copy()

    best = {
        "p_fake": -1.0, "p_real": -1.0, "box": None, "m_idx": None, "pm": None,
        "p_fake_orig": -1.0, "p_fake_filt": -1.0
    }

    for crop, box in zip(crops, boxes):
        x = DET_TFM(crop)
        p_fake0, p_real0, pm0 = predict_image_tensor(x, tta=tta)
        m_idx0 = int(np.argmax(pm0)); m_name0 = METHOD_NAMES[m_idx0]; m_conf0 = float(pm0[m_idx0])

        p_fake_filt, p_real_filt, pm_filt = -1.0, -1.0, None
        if enable_filters and m_conf0 >= method_gate:
            crop_f = apply_method_filters(crop, m_name0)
            x2 = DET_TFM(crop_f)
            p_fake_filt, p_real_filt, pm_filt = predict_image_tensor(x2, tta=tta)

        if p_fake_filt > p_fake0:
            p_fake_use, p_real_use, pm_use, m_idx_use = p_fake_filt, p_real_filt, (pm_filt if pm_filt is not None else pm0), int(np.argmax(pm_filt if pm_filt is not None else pm0))
        else:
            p_fake_use, p_real_use, pm_use, m_idx_use = p_fake0, p_real0, pm0, m_idx0

        if p_fake_use > best["p_fake"]:
            best.update({
                "p_fake": p_fake_use, "p_real": p_real_use, "box": box, "m_idx": m_idx_use, "pm": pm_use,
                "p_fake_orig": p_fake0, "p_fake_filt": p_fake_filt
            })

    fr_bar_html = _render_fake_real_bar(best["p_fake"], best["p_real"])
    method_rows = _make_method_table(best["pm"]) if best["pm"] is not None else []

    verdict = "Không phát hiện khuôn mặt."
    if best["box"] is not None:
        m_name = METHOD_NAMES[best["m_idx"]] if best["m_idx"] is not None else "Unknown"
        m_conf = float(best["pm"][best["m_idx"]]) if best["pm"] is not None and best["m_idx"] is not None else 0.0
        gate_ok = (m_conf >= float(method_gate))
        thr_eff = get_thr_for_method(thr_global, m_name, thr_map, gate_ok)
        label = "FAKE" if best["p_fake"] >= thr_eff else "REAL"
        color = (223, 64, 64) if label=="FAKE" else (64, 208, 120)
        draw_box_np(img_rgb, best["box"], color=color, thickness=int(box_thickness))
        verdict = f"{label} | top-method: {m_name} ({m_conf:.2f}) | thr_eff={thr_eff:.3f} | p_fake={best['p_fake']:.3f}"

    out_img = Image.fromarray(img_rgb)
    return out_img, verdict, fr_bar_html, method_rows

# ====================== Analyze VIDEO ======================
@torch.no_grad()
def analyze_video(video_path, use_face_crop=True, override_thr=None, tta=2, box_thickness=3,
                  per_method_thr_map=None, method_gate=0.55, enable_filters=True):
    thr_global = float(override_thr) if (override_thr is not None and not math.isnan(override_thr)) else float(DET_THR)
    thr_map = per_method_thr_map or {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Không mở được video.", "", []
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    tmpdir = tempfile.mkdtemp(prefix="df_web_")
    out_path = os.path.join(tmpdir, "out.mp4")
    if HAS_IMAGEIO:
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7)
    else:
        writer = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vout = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    sum_pfake = 0.0; sum_preal = 0.0; n_frames = 0
    pm_sum_all = np.zeros(len(METHOD_NAMES), dtype=np.float64)
    pm_sum_fake = np.zeros(len(METHOD_NAMES), dtype=np.float64)
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
            p_fake0, p_real0, pm0 = predict_image_tensor(x, tta=tta)
            m_idx0 = int(np.argmax(pm0)); m_name0 = METHOD_NAMES[m_idx0]; m_conf0 = float(pm0[m_idx0])

            p_fake_filt, p_real_filt, pm_filt = -1.0, -1.0, None
            if enable_filters and m_conf0 >= float(method_gate):
                crop_f = apply_method_filters(crop, m_name0)
                x2 = DET_TFM(crop_f)
                p_fake_filt, p_real_filt, pm_filt = predict_image_tensor(x2, tta=tta)

            if p_fake_filt > p_fake0:
                p_fake_use, p_real_use, pm_use, m_idx_use = p_fake_filt, p_real_filt, (pm_filt if pm_filt is not None else pm0), int(np.argmax(pm_filt if pm_filt is not None else pm0))
            else:
                p_fake_use, p_real_use, pm_use, m_idx_use = p_fake0, p_real0, pm0, m_idx0

            if p_fake_use > best["p_fake"]:
                best.update({"p_fake":p_fake_use, "p_real":p_real_use, "box":box, "m_idx":m_idx_use, "pm":pm_use})

        sum_pfake += best["p_fake"]; sum_preal += best["p_real"]; n_frames += 1
        if best["pm"] is not None:
            pm_sum_all += best["pm"]

        if best["m_idx"] is not None and best["pm"] is not None:
            m_name = METHOD_NAMES[best["m_idx"]]; m_conf = float(best["pm"][best["m_idx"]])
            thr_eff = get_thr_for_method(thr_global, m_name, thr_map, m_conf >= float(method_gate))
        else:
            thr_eff = thr_global
        is_fake = (best["p_fake"] >= thr_eff)
        if is_fake:
            n_fake_frames += 1
            if best["pm"] is not None: pm_sum_fake += best["pm"]

        color = (223,64,64) if is_fake else (64,208,120)
        if best["box"] is not None:
            draw_box_np(frame_rgb, best["box"], color=color, thickness=int(box_thickness))
        out_frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if HAS_IMAGEIO:
            writer.append_data(cv2.cvtColor(out_frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            vout.write(out_frame_bgr)

    cap.release()
    if HAS_IMAGEIO:
        writer.close()
    else:
        vout.release()

    pf = sum_pfake/max(1,n_frames); pr = sum_preal/max(1,n_frames)
    fr_bar_html = _render_fake_real_bar(pf, pr)
    if n_fake_frames > 0:
        pm = pm_sum_fake / max(1, pm_sum_fake.sum())
    else:
        pm = pm_sum_all / max(1, pm_sum_all.sum()) if pm_sum_all.sum() > 0 else np.zeros_like(pm_sum_all)
    method_rows = _make_method_table(pm)

    verdict = f"Frames: {n_frames} | Fake-frames: {n_fake_frames} ({(100.0*n_fake_frames/max(1,n_frames)):.1f}%)"
    return out_path, verdict, fr_bar_html, method_rows

# ====================== Gradio UI ======================
def _img_wrap(img: Image.Image, fc: bool, thr: float, tta: int, thick: int,
              pm_thr_str: str, method_gate: float, enable_filters: bool):
    if img is None:
        return None, "Chưa chọn ảnh.", "", []
    pm_map = parse_thr_map(pm_thr_str, METHOD_NAMES)
    out_img, verdict, fr_bar_html, method_rows = analyze_image(
        img, use_face_crop=bool(fc), override_thr=float(thr), tta=int(tta), box_thickness=int(thick),
        per_method_thr_map=pm_map, method_gate=float(method_gate), enable_filters=bool(enable_filters)
    )
    return out_img, verdict, fr_bar_html, method_rows

def _vid_wrap(vid_path: str, fc: bool, thr: float, tta: int, thick: int,
              pm_thr_str: str, method_gate: float, enable_filters: bool):
    if not vid_path:
        return None, "Chưa chọn video.", "", []
    pm_map = parse_thr_map(pm_thr_str, METHOD_NAMES)
    out_path, verdict, fr_bar_html, method_rows = analyze_video(
        vid_path, use_face_crop=bool(fc), override_thr=float(thr), tta=int(tta), box_thickness=int(thick),
        per_method_thr_map=pm_map, method_gate=float(method_gate), enable_filters=bool(enable_filters)
    )
    return out_path, verdict, fr_bar_html, method_rows

# ====================== App ======================
thr_default = float(CLI_THR) if (CLI_THR is not None and not math.isnan(CLI_THR)) else float(DET_THR)
pm_thr_default = CLI_PM_THR_STR or "Deepfakes=0.72,Face2Face=0.78,FaceShifter=0.76,FaceSwap=0.78,NeuralTextures=0.62"
gate_default = float(CLI_METHOD_GATE)
filt_default = bool(CLI_ENABLE_FILTERS)

with gr.Blocks(title="Deepfake Detect (UI)") as demo:
    gr.Markdown("## Deepfake Detect (UI) — threshold theo từng method + filter sau phân loại")
    with gr.Row():
        with gr.Column():
            face_crop = gr.Checkbox(label="Face crop", value=True)
            thr = gr.Slider(0.0, 0.99, value=thr_default, step=0.005, label="Global threshold (fake ≥ thr ⇒ FAKE)")
            tta = gr.Slider(1, 4, value=2, step=1, label="TTA (1/2/3/4)")
            thick = gr.Slider(1, 8, value=3, step=1, label="Độ dày khung")
        with gr.Column():
            pm_thr_str = gr.Textbox(label="Per-method threshold (vd: Deepfakes=0.72,Face2Face=0.78,FaceShifter=0.76,FaceSwap=0.78,NeuralTextures=0.62)", value=pm_thr_default)
            method_gate = gr.Slider(0.0, 1.0, value=gate_default, step=0.01, label="Method gate (>= gate mới dùng thr riêng & filter)")
            enable_filters = gr.Checkbox(label="Bật filter theo method (unsharp/deblock/denoise)", value=filt_default)
        with gr.Column():
            gr.Markdown(f"**Model**: {type(DETECTOR.backbone).__name__}  \n**img={IMG_SIZE}**  \n**Classes**: {CLASSES}  \n**Methods**: {', '.join(METHOD_NAMES)}")

    gr.Markdown("### Detect ảnh")
    with gr.Row():
        in_img = gr.Image(type="pil", label="Ảnh nguồn")
        out_img = gr.Image(type="pil", label="Ảnh đã phân tích")
    img_text = gr.Textbox(label="Kết luận", interactive=False)
    fr_bar_img = gr.HTML(label="Tỉ lệ Fake/Real")
    method_df_img = gr.Dataframe(headers=["Method", "%"], datatype=["str","number"], interactive=False, label="Tỉ lệ theo phương pháp (descending)")
    btn_img = gr.Button("Phân tích Ảnh")
    btn_img.click(_img_wrap, [in_img, face_crop, thr, tta, thick, pm_thr_str, method_gate, enable_filters], [out_img, img_text, fr_bar_img, method_df_img])

    gr.Markdown("### Detect video")
    with gr.Row():
        in_vid = gr.Video(label="Video nguồn")
        out_vid = gr.Video(label="Video đã phân tích (xem trực tiếp)")
    vid_text = gr.Textbox(label="Nhật ký", interactive=False)
    fr_bar_vid = gr.HTML(label="Tỉ lệ Fake/Real (video)")
    method_df_vid = gr.Dataframe(headers=["Method", "%"], datatype=["str","number"], interactive=False, label="Tỉ lệ theo phương pháp (descending)")
    btn_vid = gr.Button("Phân tích Video")
    btn_vid.click(_vid_wrap, [in_vid, face_crop, thr, tta, thick, pm_thr_str, method_gate, enable_filters], [out_vid, vid_text, fr_bar_vid, method_df_vid])

demo.launch(server_name="127.0.0.1", server_port=7860, share=False)