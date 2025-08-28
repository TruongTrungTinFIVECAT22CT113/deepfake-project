# app.py
import os, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import torch
from torchvision import transforms
import timm

# OpenCV optional
try:
    import cv2
except Exception:
    cv2 = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DET_CKPT = "deepfake_detector/checkpoints/detector_best.pt"  # đổi nếu cần
DEFAULT_LORA_DIR = "deepfake_generator/outputs/generic_lora"  # thư mục chứa pytorch_lora_weights.safetensors

# -------- Detector load --------
def load_detector():
    ckpt = torch.load(DET_CKPT, map_location="cpu", weights_only=False)  # để nguyên, hoặc thêm weights_only=True khi PyTorch flip mặc định
    meta = ckpt.get("meta", {})
    classes = meta.get("classes", ["fake","real"])
    mean = meta.get("norm_mean", [0.5,0.5,0.5])
    std  = meta.get("norm_std",  [0.5,0.5,0.5])
    thr  = float(meta.get("threshold", 0.7))
    model_name = meta.get("model_name", "vit_large_patch16_224")

    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print("[WARN] state_dict mismatch:", "missing:", missing, "unexpected:", unexpected)

    model.to(DEVICE).eval()
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    print("Classes:", classes, "| Threshold:", thr, "| Model:", model_name)
    return model, tfm, classes, thr

DETECTOR, DET_TFM, CLASSES, DEFAULT_THR = load_detector()

@torch.no_grad()
def _predict_tensor(x_tensor):
    logits = DETECTOR(x_tensor)
    # TTA flip ngang
    logits_flip = DETECTOR(torch.flip(x_tensor, dims=[3]))
    logits = (logits + logits_flip) / 2
    probs = torch.softmax(logits, dim=1)
    return probs

def predict_image(img: Image.Image, threshold: float):
    x = DET_TFM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = _predict_tensor(x).cpu().numpy()[0]
    p_fake = float(probs[0]); p_real = float(probs[1])
    verdict = ("⚠️ deepfake" if p_fake >= threshold else
               ("✅ real" if p_fake <= 1-threshold else "❓ không chắc"))
    return {"fake": p_fake, "real": p_real}, verdict

def predict_video(video_path, sample_every=10, max_frames=300, threshold=0.7, smooth_win=5):
    if cv2 is None:
        return {"fake":0.0, "real":1.0}, "OpenCV chưa cài: chỉ phân tích ảnh."
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fake":0.0, "real":1.0}, "Không mở được video."
    idx, fake_scores = 0, []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if idx % sample_every == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = DET_TFM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
            p = _predict_tensor(x).cpu().numpy()[0]
            fake_scores.append(float(p[0]))
            if len(fake_scores) >= max_frames: break
        idx += 1
    cap.release()
    if not fake_scores:
        return {"fake":0.0, "real":1.0}, "Không có khung hình hợp lệ."

    # smoothing EMA
    smoothed = []
    alpha = 2/(smooth_win+1)
    m = 0.0
    for v in fake_scores:
        m = alpha*v + (1-alpha)*m
        smoothed.append(m)
    mean_fake = float(np.mean(smoothed))
    mean_real = 1.0 - mean_fake
    verdict = ("⚠️ deepfake" if mean_fake >= threshold else
               ("✅ real" if mean_fake <= 1-threshold else "❓ không chắc"))
    return {"fake": mean_fake, "real": mean_real}, verdict

# -------- Generator (img2img) --------
from diffusers import StableDiffusionImg2ImgPipeline

def add_watermark(pil_img: Image.Image):
    draw = ImageDraw.Draw(pil_img)
    W,H = pil_img.size
    text = "S Y N T H E T I C"
    size = int(min(W,H)*0.08)
    try:
        font = ImageFont.truetype("arial.ttf", size=size)
    except:
        font = ImageFont.load_default()
    tw = draw.textlength(text, font=font)
    draw.text(((W-tw)/2, H-size-10), text, fill=(255,0,0,180), font=font, stroke_width=2)
    return pil_img

def load_sd_img2img(lora_dir=None):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE.type=='cuda' else torch.float32
    ).to(DEVICE)
    pipe.safety_checker=None
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()

    # nạp LoRA nếu có đúng file tiêu chuẩn
    weight_path = None
    if lora_dir:
        cand = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")
        if os.path.exists(cand):
            weight_path = cand

    if weight_path:
        try:
            pipe.load_lora_weights(lora_dir, weight_name="pytorch_lora_weights.safetensors")
            print(f"Loaded LoRA (load_lora_weights): {weight_path}")
        except Exception:
            pipe.unet.load_attn_procs(lora_dir)
            print(f"Loaded LoRA via load_attn_procs: {lora_dir}")
    else:
        print("⚠️ Không tìm thấy LoRA, dùng base model.")

    return pipe

SD_PIPE = None
def ensure_pipe(lora_dir):
    global SD_PIPE
    if SD_PIPE is None:
        SD_PIPE = load_sd_img2img(lora_dir)
    return SD_PIPE

def generate_img(target_img: Image.Image, strength=0.45, guidance=7.5, lora_dir=DEFAULT_LORA_DIR):
    pipe = ensure_pipe(lora_dir)
    prompt = "portrait of a generic face"
    out = pipe(prompt=prompt,
               image=target_img.convert("RGB").resize((512,512)),
               strength=float(strength),
               guidance_scale=float(guidance),
               num_inference_steps=30).images[0]
    return add_watermark(out)

def generate_video(target_video_path: str, strength=0.45, guidance=7.5, lora_dir=DEFAULT_LORA_DIR, sample_every=3):
    if cv2 is None:
        return None, "OpenCV chưa cài: chỉ hỗ trợ ảnh."
    pipe = ensure_pipe(lora_dir)
    cap = cv2.VideoCapture(target_video_path)
    if not cap.isOpened(): return None, "Không mở được video."
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out_path = "synthetic_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W,H))
    fidx=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if fidx % sample_every == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            gen = generate_img(img, strength, guidance, lora_dir).resize((W,H))
            frame = cv2.cvtColor(np.array(gen), cv2.COLOR_RGB2BGR)
        writer.write(frame); fidx+=1
    cap.release(); writer.release()
    return out_path, "OK"

# --------- Gradio UI ---------
with gr.Blocks(title="Deepfake Study Toolkit") as demo:
    gr.Markdown("## 🔍 Deepfake Detector (ViT) & 🧪 Synthetic Face Transfer (SD Img2Img — Study-only)\n"
                "**Lưu ý đạo đức:** Output luôn gắn watermark `SYNTHETIC`. Đừng dùng để mạo danh/bịa đặt.")
    with gr.Tab("Detect"):
        thr = gr.Slider(0.1, 0.9, DEFAULT_THR, step=0.01, label="Ngưỡng quyết định (fake ≥ ngưỡng)")
        smooth = gr.Slider(1, 21, 5, step=2, label="Smoothing window (EMA, video)")
        sample_every = gr.Slider(1, 30, 10, step=1, label="Lấy 1 khung mỗi N frames")
        max_frames = gr.Slider(50, 1200, 300, step=50, label="Tối đa khung phân tích")
        with gr.Row():
            img_in = gr.Image(type="pil", label="Ảnh")
            vid_in = gr.Video(label="Hoặc video")
        btn_img = gr.Button("Phân tích Ảnh")
        btn_vid = gr.Button("Phân tích Video")
        out_img = gr.Label(label="Kết quả Ảnh (softmax)")
        out_vid = gr.Label(label="Kết quả Video (trung bình)")
        verdict = gr.Textbox(label="Nhận định", interactive=False)

        def _img_wrap(img, thr_val): return predict_image(img, thr_val)
        def _vid_wrap(vid, every, mx, thr_val, sm): return predict_video(vid, int(every), int(mx), float(thr_val), int(sm))

        btn_img.click(fn=_img_wrap, inputs=[img_in, thr], outputs=[out_img, verdict])
        btn_vid.click(fn=_vid_wrap, inputs=[vid_in, sample_every, max_frames, thr, smooth], outputs=[out_vid, verdict])

    with gr.Tab("Generate (Study-only)"):
        gr.Markdown("**Output có watermark `SYNTHETIC`.** LoRA đã train từ `train/real` (generic face).")
        tgt_img = gr.Image(type="pil", label="Ảnh đích (img2img)")
        tgt_vid = gr.Video(label="Hoặc Video đích")
        lora_dir = gr.Textbox(value=DEFAULT_LORA_DIR, label="Thư mục LoRA (chứa lora.safetensors)")
        strength = gr.Slider(0.2, 0.8, 0.45, step=0.05, label="Img2Img strength")
        guidance = gr.Slider(4.0, 12.0, 7.5, step=0.5, label="Guidance scale")
        sample_ev = gr.Slider(1, 10, 3, step=1, label="Video: lấy 1 khung mỗi N frames")
        gen_btn_img = gr.Button("Tạo Ảnh Synthetic")
        gen_btn_vid = gr.Button("Tạo Video Synthetic")
        out_gen_img = gr.Image(label="Ảnh kết quả (watermarked)")
        out_gen_vid = gr.Video(label="Video kết quả (watermarked)")
        gen_log = gr.Textbox(label="Log", interactive=False)

        def _gen_img(img, s, g, ld):
            if img is None: return None
            return generate_img(img, float(s), float(g), ld)

        def _gen_vid(vid, s, g, ld, ev):
            if vid is None: return None, "Chưa chọn video."
            path, msg = generate_video(vid, float(s), float(g), ld, int(ev))
            return path, msg

        gen_btn_img.click(fn=_gen_img, inputs=[tgt_img, strength, guidance, lora_dir], outputs=out_gen_img)
        gen_btn_vid.click(fn=_gen_vid, inputs=[tgt_vid, strength, guidance, lora_dir, sample_ev], outputs=[out_gen_vid, gen_log])

if __name__ == "__main__":
    demo.launch()