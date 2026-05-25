# backend/components/xai_cam.py
from __future__ import annotations
import os
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ═══════════════════════════════════════════════════════════════
# COMMON UTILS
# ═══════════════════════════════════════════════════════════════

def _normalize_cam(cam: torch.Tensor) -> np.ndarray:
    cam = cam.detach().float()   # ép FP32 trước khi ra numpy — cv2 không hỗ trợ FP16
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam.cpu().numpy()

def overlay_heatmap_on_bgr(frame_bgr: np.ndarray, cam_2d: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    cam_resized = cv2.resize(cam_2d, (frame_bgr.shape[1], frame_bgr.shape[0]))
    # Guard NaN/Inf: xảy ra khi cam.max()==0 (frame không có gradient) → trả nguyên frame
    if not np.isfinite(cam_resized).all():
        return frame_bgr.copy()
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    # Re-normalize sau clip để tránh all-zero map (nền xanh tuyền)
    _cmax = cam_resized.max()
    if _cmax < 1e-6:
        return frame_bgr.copy()
    cam_resized = cam_resized / _cmax
    cam_resized = np.power(cam_resized, 0.6)
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(heat, alpha, frame_bgr, 1 - alpha, 0)
    return out

def save_cam_image(frame_bgr: np.ndarray, cam_2d: np.ndarray, out_dir: Optional[str] = None) -> str:
    merged = overlay_heatmap_on_bgr(frame_bgr, cam_2d)
    if out_dir is None:
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    else:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "xai_cam.png")
    cv2.imwrite(out_path, merged)
    return out_path


# ═══════════════════════════════════════════════════════════════
# CNN — GradCAM++ với cached hook (tối ưu tốc độ)
# ═══════════════════════════════════════════════════════════════

class CnnHookContext:
    """
    Register forward+backward hook lên target_layer MỘT LẦN duy nhất cho cả video.
    Dùng như context manager hoặc gọi .remove() thủ công khi xong.

    Tại sao cache hook thay vì register/remove mỗi frame:
    - register_forward_hook / register_backward_hook đều acquire lock nội bộ
      của nn.Module để gắn handle vào _forward_hooks / _backward_hooks.
    - Mỗi frame gọi 2×register + 2×remove = 4 lần acquire/release lock → overhead
      thuần Python cộng dồn đáng kể ở 25–30 fps.
    - Cache 1 lần, giữ suốt video → 0 overhead per-frame.
    """
    def __init__(self, target_layer: nn.Module):
        self.features: dict = {}
        self.gradients: dict = {}

        def fwd(module, inp, out):
            self.features["value"] = out.detach()

        def bwd(module, gin, gout):
            self.gradients["value"] = gout[0].detach()

        self._h1 = target_layer.register_forward_hook(fwd)
        self._h2 = target_layer.register_backward_hook(bwd)

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()


def _gradcam_pp_with_ctx(
    model: nn.Module,
    input_tensor: torch.Tensor,
    ctx: CnnHookContext,
    target_index: int,
    device: str,
) -> np.ndarray:
    """
    GradCAM++ dùng hook đã được register sẵn từ CnnHookContext.
    Không register/remove hook → zero per-frame overhead.
    Chạy native dtype của model (FP16/FP32).
    """
    first_param = next(model.parameters())
    x = input_tensor.to(device=device, dtype=first_param.dtype)

    with torch.enable_grad():
        # Không .clone() thừa — detach rồi requires_grad trực tiếp
        inp = x.detach().requires_grad_(True)
        out = model(inp)
        logits_bin = out[0] if isinstance(out, (list, tuple)) else out
        score = logits_bin[0, target_index]
        model.zero_grad()
        score.backward()

    with torch.no_grad():
        feat = ctx.features["value"].float()    # [1, C, h, w]
        grad = ctx.gradients["value"].float()   # [1, C, h, w]

        grad_2 = grad ** 2
        grad_3 = grad ** 3
        # mean (không phải sum) theo paper GradCAM++ — tránh scale blow-up
        sum_act = feat.mean(dim=(2, 3), keepdim=True)
        denom = 2.0 * grad_2 + sum_act * grad_3 + 1e-7
        alpha = grad_2 / denom
        weights = (alpha * torch.relu(grad)).sum(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feat).sum(dim=1).squeeze(0))

    return _normalize_cam(cam)


# Giữ lại hàm cũ để không break code nào đang gọi trực tiếp
def _gradcam_original_single(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_index: int,
    device: str,
) -> np.ndarray:
    """GradCAM thường — fallback khi không có CnnHookContext."""
    features: dict = {}
    gradients: dict = {}

    def fwd_hook(module, inp, out):
        features["value"] = out.detach()

    def bwd_hook(module, gin, gout):
        gradients["value"] = gout[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)

    first_param = next(model.parameters())
    x = input_tensor.to(device=device, dtype=first_param.dtype)

    with torch.enable_grad():
        inp = x.detach().requires_grad_(True)
        out = model(inp)
        logits_bin = out[0] if isinstance(out, (list, tuple)) else out
        score = logits_bin[0, target_index]
        model.zero_grad()
        score.backward()

    h1.remove()
    h2.remove()

    with torch.no_grad():
        feat = features["value"].float()
        grad = gradients["value"].float()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feat).sum(dim=1).squeeze(0))

    return _normalize_cam(cam)


def _gradcam_plus_plus_single(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_index: int,
    device: str,
) -> np.ndarray:
    """GradCAM++ fallback (register hook mỗi lần) — dùng khi không có CnnHookContext."""
    features: dict = {}
    gradients: dict = {}

    def fwd_hook(module, inp, out):
        features["value"] = out.detach()

    def bwd_hook(module, gin, gout):
        gradients["value"] = gout[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)

    first_param = next(model.parameters())
    x = input_tensor.to(device=device, dtype=first_param.dtype)

    with torch.enable_grad():
        inp = x.detach().requires_grad_(True)
        out = model(inp)
        logits_bin = out[0] if isinstance(out, (list, tuple)) else out
        score = logits_bin[0, target_index]
        model.zero_grad()
        score.backward()

    h1.remove()
    h2.remove()

    with torch.no_grad():
        feat = features["value"].float()
        grad = gradients["value"].float()
        grad_2 = grad ** 2
        grad_3 = grad ** 3
        sum_act = feat.mean(dim=(2, 3), keepdim=True)
        denom = 2.0 * grad_2 + sum_act * grad_3 + 1e-7
        alpha = grad_2 / denom
        weights = (alpha * torch.relu(grad)).sum(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feat).sum(dim=1).squeeze(0))

    return _normalize_cam(cam)


def _apply_cnn_postprocess(cam_avg: np.ndarray, extra_smooth: bool) -> np.ndarray:
    """
    Post-process chung cho CNN CAM: normalize → smooth → oval mask.
    Tách ra hàm riêng để tái dùng, tránh lặp code.
    """
    cam_avg = cam_avg - cam_avg.min()
    cam_avg = cam_avg / (cam_avg.max() + 1e-6)

    if extra_smooth:
        H_c, W_c = cam_avg.shape
        _sigma = max(2.0, W_c * 0.12)
        cam_avg = cv2.GaussianBlur(cam_avg, (0, 0), sigmaX=_sigma, sigmaY=_sigma)
        cam_avg = cam_avg - cam_avg.min()
        cam_avg = cam_avg / (cam_avg.max() + 1e-6)
        cam_avg = np.power(cam_avg, 1.8)
        cam_avg = cam_avg / (cam_avg.max() + 1e-6)
    else:
        cam_avg = np.power(cam_avg, 2.2)
        cam_avg = cam_avg / (cam_avg.max() + 1e-6)

    # Oval mask: giới hạn heatmap trong vùng đầu/mặt
    H_c, W_c = cam_avg.shape
    cy = int(H_c * 0.48)
    cx = int(W_c * 0.50)
    ry = int(H_c * 0.50)
    rx = int(W_c * 0.46)
    Y, X = np.ogrid[:H_c, :W_c]
    dist = ((X - cx) / max(rx, 1)) ** 2 + ((Y - cy) / max(ry, 1)) ** 2
    oval_mask = np.clip(1.5 - dist, 0.0, 1.0).astype(np.float32)
    oval_mask = cv2.GaussianBlur(oval_mask, (0, 0), sigmaX=20, sigmaY=20)
    cam_avg = cam_avg * oval_mask
    cam_avg = cam_avg - cam_avg.min()
    cam_avg = cam_avg / (cam_avg.max() + 1e-6)
    return cam_avg


def generate_cam_cnn_with_ctx(
    model: nn.Module,
    input_tensor: torch.Tensor,
    ctx: CnnHookContext,
    target_index: int = 0,
    device: str = "cuda",
    extra_smooth: bool = False,
) -> np.ndarray:
    """
    GradCAM++ nhanh nhất — dùng hook đã cache từ CnnHookContext.
    smooth_samples=1: trên CUDA 1 pass đã stable, không cần SmoothGrad.
    Gọi hàm này từ inference.py thay cho generate_cam_cnn khi có ctx.
    """
    model.eval()
    first_param = next(model.parameters())
    x0 = input_tensor.to(device=device, dtype=first_param.dtype)

    cam = _gradcam_pp_with_ctx(model, x0, ctx, target_index, device)
    return _apply_cnn_postprocess(cam, extra_smooth)


def generate_cam_cnn(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_index: int = 1,
    target_layer: Optional[nn.Module] = None,
    device: str = "cuda",
    smooth_samples: int = 1,
    smooth_noise: float = 0.015,
    extra_smooth: bool = False,
) -> np.ndarray:
    """
    GradCAM++ cho CNN — fallback khi không dùng CnnHookContext.
    smooth_samples mặc định = 1 (CUDA không cần SmoothGrad để stable).
    Nếu muốn SmoothGrad tăng lên 3, nhưng 1 là đủ trên GPU.

    Giữ nguyên serial loop — KHÔNG batch nhiều samples vào 1 forward:
    EfficientNet/ResNet dùng BatchNorm → batch N samples làm BN tính
    statistics cross-sample → gradient bị pha trộn chéo → heatmap nhiễu.
    """
    model.eval()
    if target_layer is None:
        raise ValueError("Bạn phải truyền target_layer vào generate_cam_cnn")

    first_param = next(model.parameters())
    model_dtype = first_param.dtype
    x0 = input_tensor.to(device=device, dtype=model_dtype)

    if smooth_samples <= 1:
        # Fast path: 1 pass, không tạo noise, không stack
        cam = _gradcam_plus_plus_single(model, x0, target_layer, target_index, device)
    else:
        inp_std = float(x0.float().std().item()) * smooth_noise
        cams = []
        for i in range(smooth_samples):
            noisy = x0 if (i == 0 or inp_std <= 0) else x0 + torch.randn_like(x0) * inp_std
            cams.append(_gradcam_plus_plus_single(model, noisy, target_layer, target_index, device))
        cams_stack = np.stack(cams, axis=0)
        cam = np.median(cams_stack, axis=0) if extra_smooth else np.mean(cams_stack, axis=0)

    return _apply_cnn_postprocess(cam, extra_smooth)


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# ViT / BEiT — GradCAM bản gốc (hook block cuối)
# ═══════════════════════════════════════════════════════════════

def generate_cam_vit(
    vit_model: nn.Module,
    input_tensor: torch.Tensor,    # [1,3,H,W] normalized
    target_index: int = 1,
    device: str = "cuda",
    patch_size: int = 16,
    smooth_output: bool = False,
) -> np.ndarray:
    """
    Grad-CAM cho ViT/BEiT timm:
      - hook vào block cuối (transformer encoder)
      - dùng patch token (bỏ cls token)
      - nếu map Grad-CAM gần như phẳng -> fallback sang feature energy
    """
    vit_model.eval()
    backbone = getattr(vit_model, "backbone", vit_model)

    if not hasattr(backbone, "blocks"):
        raise ValueError("ViT backbone không có thuộc tính .blocks (mong đợi timm VisionTransformer).")

    # Giữ nguyên dtype model — không ép FP32 từ bên ngoài
    first_param = next(vit_model.parameters())
    x0 = input_tensor.to(device=device, dtype=first_param.dtype)

    _, _, H, W = x0.shape
    h_p = H // patch_size
    w_p = W // patch_size

    feats: dict = {}
    grads: dict = {}
    last_block = backbone.blocks[-1]

    def fwd_hook(module, inp, out):
        feats["value"] = out.detach()

    def bwd_hook(module, gin, gout):
        grads["value"] = gout[0].detach()

    h1 = last_block.register_forward_hook(fwd_hook)
    h2 = last_block.register_backward_hook(bwd_hook)

    with torch.enable_grad():
        x = x0.clone().detach().requires_grad_(True)
        out = vit_model(x)
        if isinstance(out, (list, tuple)):
            logits_bin = out[0]
        else:
            logits_bin = out
        score = logits_bin[0, target_index]
        vit_model.zero_grad()
        score.backward(retain_graph=False)

    h1.remove()
    h2.remove()

    tokens = feats["value"].float()        # [1, N, C]
    grad_tokens = grads["value"].float()   # [1, N, C]

    # bỏ cls token
    tokens = tokens[:, 1:, :]
    grad_tokens = grad_tokens[:, 1:, :]

    weights = grad_tokens.mean(dim=1, keepdim=True)
    cam_patch = torch.bmm(tokens, weights.transpose(1, 2))
    cam_patch = cam_patch.view(1, h_p, w_p)
    cam_patch = F.relu(cam_patch)

    cam_up = F.interpolate(
        cam_patch.unsqueeze(1),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    # Fallback nếu map phẳng
    contrast = float((cam_up.max() - cam_up.min()).item())
    if contrast < 1e-6:
        with torch.no_grad():
            feats2: dict = {}
            def fwd_hook2(module, inp, out):
                feats2["value"] = out
            h3 = last_block.register_forward_hook(fwd_hook2)
            _ = vit_model(x0.to(device))
            h3.remove()
            tokens2 = feats2["value"].detach().float()[:, 1:, :]
            # float() trước pow(2) để tránh FP16 overflow → inf → NaN trong normalize
            cam_patch2 = tokens2.pow(2).sum(dim=-1).view(1, h_p, w_p)
            cam_up = F.interpolate(
                cam_patch2.unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

    cam_norm = _normalize_cam(cam_up)

    # Áp oval mask để cắt bớt vùng tràn ra ngoài mặt
    # BEiT: ellipse nhỏ hơn + blur mạnh hơn vì gradient loạn hơn
    # ViT: ellipse rộng hơn, chỉ cắt phần tràn ra tóc/nền
    if smooth_output:
        # BEiT — ellipse nhỏ, suppress mạnh
        _ry_frac, _rx_frac = 0.42, 0.38
        _sigma = 15
    else:
        # ViT — ellipse rộng hơn, chỉ cắt phần tràn
        _ry_frac, _rx_frac = 0.42, 0.38
        _sigma = 12

    if True:  # áp cho cả ViT lẫn BEiT
        H_c, W_c = cam_norm.shape
        cy = int(H_c * 0.48)
        cx = int(W_c * 0.50)
        ry = int(H_c * _ry_frac)
        rx = int(W_c * _rx_frac)
        Y, X = np.ogrid[:H_c, :W_c]
        dist = ((X - cx) / max(rx, 1)) ** 2 + ((Y - cy) / max(ry, 1)) ** 2
        oval_mask = np.clip(1.5 - dist, 0.0, 1.0).astype(np.float32)
        oval_mask = cv2.GaussianBlur(oval_mask, (0, 0), sigmaX=_sigma, sigmaY=_sigma)
        cam_norm = cam_norm * oval_mask
        cam_norm = cam_norm - cam_norm.min()
        cam_norm = cam_norm / (cam_norm.max() + 1e-6)
        # Tăng contrast: power > 1 làm vùng thấp xuống nhanh hơn
        # chỉ giữ lại vùng thực sự "nóng", tránh cả mặt đỏ đều
        _power = 3.0 if not smooth_output else 1.5
        cam_norm = np.power(cam_norm, _power)

    return cam_norm


# ═══════════════════════════════════════════════════════════════
# Swin GRAD-CAM (giữ nguyên, Swin thường ổn hơn ViT/CNN)
# ═══════════════════════════════════════════════════════════════

def generate_cam_swin(
    swin_model: nn.Module,
    input_tensor: torch.Tensor,
    target_index: int = 1,
    device: str = "cuda",
) -> np.ndarray:
    """
    Grad-CAM token-based cho Swin / SwinV2 timm.
    Hook vào block cuối của stage cuối.
    """
    swin_model.eval()
    backbone = getattr(swin_model, "backbone", swin_model)

    if not hasattr(backbone, "layers"):
        raise ValueError("Swin backbone không có thuộc tính .layers")

    first_param = next(swin_model.parameters())
    x0 = input_tensor.to(device=device, dtype=first_param.dtype)

    layers = backbone.layers
    last_stage = layers[-1]
    blocks = getattr(last_stage, "blocks", None)
    if not blocks or len(blocks) == 0:
        raise ValueError("Swin stage cuối không có blocks.")

    last_block = blocks[-1]
    feats: dict = {}
    grads: dict = {}

    def fwd_hook(module, inp, out): feats["value"] = out
    def bwd_hook(module, gin, gout): grads["value"] = gout[0]

    h1 = last_block.register_forward_hook(fwd_hook)
    h2 = last_block.register_backward_hook(bwd_hook)

    with torch.enable_grad():
        x = x0.clone().detach().requires_grad_(True)
        out = swin_model(x)
        logits_bin = out[0] if isinstance(out, (list, tuple)) else out
        score = logits_bin[0, target_index]
        swin_model.zero_grad()
        score.backward(retain_graph=False)

    h1.remove()
    h2.remove()

    tokens = feats["value"].float()
    grad_tokens = grads["value"].float()

    if tokens.dim() == 4:
        B, H_p, W_p, C = tokens.shape
        tokens = tokens.view(B, H_p * W_p, C)
        grad_tokens = grad_tokens.view(B, H_p * W_p, C)
        h_p, w_p = H_p, W_p
    elif tokens.dim() == 3:
        B, N, C = tokens.shape
        h_p = int(N ** 0.5)
        w_p = h_p
        if h_p * w_p > N:
            h_p = w_p = N
        tokens = tokens[:, :h_p * w_p, :]
        grad_tokens = grad_tokens[:, :h_p * w_p, :]
    else:
        raise ValueError(f"Shape output Swin không hỗ trợ: {tokens.shape}")

    weights = grad_tokens.mean(dim=1, keepdim=True)
    cam_patch = torch.bmm(tokens, weights.transpose(1, 2))
    cam_patch = cam_patch.view(1, 1, h_p, w_p)
    cam_patch = F.relu(cam_patch)

    _, _, H, W = x0.shape
    cam_up = F.interpolate(
        cam_patch, size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze(0).squeeze(0)

    return _normalize_cam(cam_up)