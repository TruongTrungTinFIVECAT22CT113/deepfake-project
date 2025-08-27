# app.py (đặt ở root dự án)
import os, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import torch
from torchvision import transforms
import timm
from diffusers import StableDiffusionImg2ImgPipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DET_CKPT = "deepfake_detector/checkpoints/detector_best.pt"
CLASSES = ["fake","real"]  # theo train

# -------- Detector load --------
def load_detector():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    sd = torch.load(DET_CKPT, map_location="cpu")["model"]
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])
    return model, tfm

DETECTOR, DET_TFM = load_detector()

def predict_image(img: Image.Image):
    x = DET_TFM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = DETECTOR(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return { "fake": float(probs[0]), "real": float(probs[1]) }

def predict_video(video_path, sample_every=10, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    idx, fake_scores, real_scores = 0, [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if idx % sample_every == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            p = predict_image(img)
            fake_scores.append(p["fake"]); real_scores.append(p["real"])
        idx += 1
        if len(fake_scores) >= max_frames: break
    cap.release()
    if not fake_scores:
        return {"fake":0.0, "real":0.0}, "No frames"
    mean_fake = float(np.mean(fake_scores)); mean_real = float(np.mean(real_scores))
    verdict = "⚠️ Có dấu hiệu deepfake" if mean_fake > mean_real else "✅ Có vẻ là real"
    return {"fake":mean_fake, "real":mean_real}, verdict

# -------- Generator (study-only; watermark) --------
def load_sd_img2img():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if DEVICE.type=='cuda' else torch.float32
    )
    pipe.to(DEVICE); pipe.safety_checker=None
    return pipe
SD_PIPE = load_sd_img2img()

def add_watermark(pil_img: Image.Image):
    draw = ImageDraw.Draw(pil_img)
    W,H = pil_img.size
    text = "S Y N T H E T I C"
    size = int(min(W,H)*0.08)
    try:
        font = ImageFont.truetype("arial.ttf", size=size)
    except:
        font = ImageFont.load_default()
    tw,th = draw.textlength(text, font=font), size
    draw.text(((W-tw)/2, H-th-10), text, fill=(255,0,0,180), font=font, stroke_width=2)
    return pil_img

def generate_img(source_face: Image.Image, target_img: Image.Image, token="personx", strength=0.45, guidance=7.5, lora_path="deepfake_generator/outputs/lora_personx.pt"):
    # OPTIONAL: apply LoRA weights if trained
    if os.path.exists(lora_path):
        state = torch.load(lora_path, map_location="cpu")
        # here we don't actually merge; this is a placeholder to show where you would apply LoRA
        # For a real merge, load into attention processors accordingly (kept simple for educational use)
    prompt = f"photo of {token}"
    out = SD_PIPE(prompt=prompt,
                  image=target_img.convert("RGB").resize((512,512)),
                  strength=float(strength),
                  guidance_scale=float(guidance),
                  num_inference_steps=30).images[0]
    return add_watermark(out)

def generate_video(source_face: Image.Image, target_video_path: str, token="personx",
                   strength=0.45, guidance=7.5, lora_path="deepfake_generator/outputs/lora_personx.pt",
                   sample_every=3):
    cap = cv2.VideoCapture(target_video_path)
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
            gen = generate_img(source_face, img, token, strength, guidance, lora_path).resize((W,H))
            frame = cv2.cvtColor(np.array(gen), cv2.COLOR_RGB2BGR)
        writer.write(frame); fidx+=1
    cap.release(); writer.release()
    return out_path

# --------- Gradio UI ---------
with gr.Blocks(title="Deepfake Study Toolkit") as demo:
    gr.Markdown("## 🔍 Deepfake Detector (ViT) & 🧪 Synthetic Face Transfer (SD Img2Img — Study-only)")
    with gr.Tab("Detect"):
        with gr.Row():
            img_in = gr.Image(type="pil", label="Ảnh")
            vid_in = gr.Video(label="Hoặc video")
        btn_img = gr.Button("Phân tích Ảnh")
        btn_vid = gr.Button("Phân tích Video")
        out_img = gr.Label(label="Kết quả Ảnh (softmax)")
        out_vid = gr.Label(label="Kết quả Video (trung bình)")
        verdict = gr.Textbox(label="Nhận định", interactive=False)
        btn_img.click(fn=predict_image, inputs=img_in, outputs=out_img)
        btn_vid.click(fn=predict_video, inputs=vid_in, outputs=[out_vid, verdict])

    with gr.Tab("Generate (Study-only)"):
        gr.Markdown("**Cảnh báo đạo đức:** Output sẽ gắn watermark `SYNTHETIC`. Hãy tôn trọng quyền riêng tư & xin phép đối tượng liên quan.")
        src = gr.Image(type="pil", label="Ảnh khuôn mặt X (đã fine-tune LoRA trên ảnh cùng người)")
        tgt_img = gr.Image(type="pil", label="Ảnh đích (img2img)")
        tgt_vid = gr.Video(label="Hoặc Video đích")
        token = gr.Textbox(value="personx", label="Token đã dùng khi train LoRA")
        strength = gr.Slider(0.2, 0.8, 0.45, step=0.05, label="Img2Img strength (0.35–0.55 khuyên dùng)")
        guidance = gr.Slider(4.0, 12.0, 7.5, step=0.5, label="Guidance scale")
        lora = gr.Textbox(value="deepfake_generator/outputs/lora_personx.pt", label="Đường dẫn LoRA (nếu có)")
        gen_btn_img = gr.Button("Tạo Ảnh Synthetic")
        gen_btn_vid = gr.Button("Tạo Video Synthetic")
        out_gen_img = gr.Image(label="Ảnh kết quả (watermarked)")
        out_gen_vid = gr.Video(label="Video kết quả (watermarked)")
        gen_btn_img.click(fn=generate_img, inputs=[src, tgt_img, token, strength, guidance, lora], outputs=out_gen_img)
        gen_btn_vid.click(fn=generate_video, inputs=[src, tgt_vid, token, strength, guidance, lora], outputs=out_gen_vid)

if __name__ == "__main__":
    demo.launch()