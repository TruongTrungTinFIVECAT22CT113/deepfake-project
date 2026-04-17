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
    cam = cam.detach()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam.cpu().numpy()

def overlay_heatmap_on_bgr(frame_bgr: np.ndarray, cam_2d: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    cam_resized = cv2.resize(cam_2d, (frame_bgr.shape[1], frame_bgr.shape[0]))
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
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
# CNN — GradCAM++ + SmoothGrad
# ═══════════════════════════════════════════════════════════════

def _gradcam_original_single(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_index: int,
    device: str,
) -> np.ndarray:
    """GradCAM thường — hoạt động tốt hơn GradCAM++ với EfficientNet."""
    features = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        features["value"] = out.detach()

    def bwd_hook(module, gin, gout):
        gradients["value"] = gout[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)

    with torch.enable_grad():
        inp = input_tensor.clone().requires_grad_(True)
        out = model(inp)
        logits_bin = out[0] if isinstance(out, (list, tuple)) else out
        score = logits_bin[0, target_index]
        model.zero_grad()
        score.backward()

    h1.remove()
    h2.remove()

    feat = features["value"]   # [1, C, h, w]
    grad = gradients["value"]  # [1, C, h, w]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feat).sum(dim=1).squeeze(0)
    cam = F.relu(cam)
    return _normalize_cam(cam)


def _gradcam_plus_plus_single(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_index: int,
    device: str,
) -> np.ndarray:
    """
    GradCAM++ cho 1 lần forward.
    Cải tiến so với GradCAM thường:
    - Dùng bậc 2 và bậc 3 của gradient để tính alpha weights
    - Cho phép tập trung vào nhiều vùng nhỏ cùng lúc thay vì chỉ 1 vùng lớn
    - Fix vấn đề ConvNeXt/EfficientNet chỉ highlight 1 vùng không ổn định
    """
    features = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        features["value"] = out.detach()

    def bwd_hook(module, gin, gout):
        gradients["value"] = gout[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)

    with torch.enable_grad():
        inp = input_tensor.clone().requires_grad_(True)
        out = model(inp)
        logits_bin = out[0] if isinstance(out, (list, tuple)) else out
        score = logits_bin[0, target_index]
        model.zero_grad()
        score.backward()

    h1.remove()
    h2.remove()

    feat = features["value"]    # [1, C, h, w]
    grad = gradients["value"]   # [1, C, h, w]

    # ── GradCAM++ alpha weights ──
    grad_2 = grad ** 2
    grad_3 = grad ** 3
    # global sum của feature map nhân grad bậc 3
    sum_act = feat.sum(dim=(2, 3), keepdim=True)          # [1, C, 1, 1]
    denom = 2.0 * grad_2 + sum_act * grad_3 + 1e-7
    alpha = grad_2 / denom                                 # [1, C, 1, 1]
    # chỉ giữ phần dương của gradient (ReLU)
    weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

    cam = (weights * feat).sum(dim=1).squeeze(0)           # [h, w]
    cam = F.relu(cam)
    return _normalize_cam(cam)


def generate_cam_cnn(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_index: int = 1,
    target_layer: Optional[nn.Module] = None,
    device: str = "cuda",
    smooth_samples: int = 3,
    smooth_noise: float = 0.015,
    extra_smooth: bool = False,
) -> np.ndarray:
    """
    GradCAM++ + SmoothGrad cho CNN (ConvNeXt, EfficientNet, ResNetRS...).

    SmoothGrad: chạy GradCAM++ nhiều lần với Gaussian noise nhỏ rồi average.
    - Fix vấn đề heatmap nhảy lung tung giữa các frame
    - Noise nhỏ (0.015) không làm thay đổi dự đoán nhưng ổn định gradient
    - smooth_samples=10 cho kết quả tốt, tăng lên 20 nếu muốn mịn hơn (chậm hơn)
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    if target_layer is None:
        raise ValueError("Bạn phải truyền target_layer vào generate_cam_cnn")

    inp_std = float(input_tensor.std().item()) * smooth_noise

    # EfficientNet dùng GradCAM thường (không GradCAM++) vì compound scaling
    # làm GradCAM++ cho ra heatmap flat. Các CNN khác dùng GradCAM++.
    single_fn = _gradcam_original_single if extra_smooth else _gradcam_plus_plus_single

    cams = []
    for _ in range(smooth_samples):
        noise = torch.randn_like(input_tensor) * inp_std
        noisy = (input_tensor + noise).to(device)
        cams.append(single_fn(model, noisy, target_layer, target_index, device))

    cam_avg = np.mean(np.stack(cams, axis=0), axis=0)
    cam_avg = cam_avg - cam_avg.min()
    cam_avg = cam_avg / (cam_avg.max() + 1e-6)

    # EfficientNet: oval mask tập trung vào vùng mặt
    # cy=0.40 vì crop EfficientNet ít cắt trán hơn ViT
    if extra_smooth:
        H_c, W_c = cam_avg.shape
        cy = int(H_c * 0.40)
        cx = int(W_c * 0.50)
        ry = int(H_c * 0.42)
        rx = int(W_c * 0.40)
        Y, X = np.ogrid[:H_c, :W_c]
        dist = ((X - cx) / max(rx, 1)) ** 2 + ((Y - cy) / max(ry, 1)) ** 2
        oval_mask = np.clip(1.5 - dist, 0.0, 1.0).astype(np.float32)
        oval_mask = cv2.GaussianBlur(oval_mask, (0, 0), sigmaX=20, sigmaY=20)
        cam_avg = cam_avg * oval_mask
        cam_avg = cam_avg - cam_avg.min()
        cam_avg = cam_avg / (cam_avg.max() + 1e-6)

    return cam_avg


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
    input_tensor = input_tensor.to(device)
    backbone = getattr(vit_model, "backbone", vit_model)

    if not hasattr(backbone, "blocks"):
        raise ValueError("ViT backbone không có thuộc tính .blocks (mong đợi timm VisionTransformer).")

    _, _, H, W = input_tensor.shape
    h_p = H // patch_size
    w_p = W // patch_size

    feats: dict = {}
    grads: dict = {}
    last_block = backbone.blocks[-1]

    def fwd_hook(module, inp, out):
        feats["value"] = out

    def bwd_hook(module, gin, gout):
        grads["value"] = gout[0]

    h1 = last_block.register_forward_hook(fwd_hook)
    h2 = last_block.register_backward_hook(bwd_hook)

    with torch.enable_grad():
        x = input_tensor.clone().detach().to(device)
        x.requires_grad_(True)
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

    tokens = feats["value"]        # [1, N, C]
    grad_tokens = grads["value"]   # [1, N, C]

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
            _ = vit_model(input_tensor.to(device))
            h3.remove()
            tokens2 = feats2["value"][:, 1:, :]
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
        _ry_frac, _rx_frac = 0.50, 0.46
        _sigma = 20

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
        _power = 2.2 if not smooth_output else 1.5
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
    input_tensor = input_tensor.to(device)
    backbone = getattr(swin_model, "backbone", swin_model)

    if not hasattr(backbone, "layers"):
        raise ValueError("Swin backbone không có thuộc tính .layers")

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
        x = input_tensor.clone().detach().to(device)
        x.requires_grad_(True)
        out = swin_model(x)
        logits_bin = out[0] if isinstance(out, (list, tuple)) else out
        score = logits_bin[0, target_index]
        swin_model.zero_grad()
        score.backward(retain_graph=False)

    h1.remove()
    h2.remove()

    tokens = feats["value"]
    grad_tokens = grads["value"]

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

    _, _, H, W = input_tensor.shape
    cam_up = F.interpolate(
        cam_patch, size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze(0).squeeze(0)

    return _normalize_cam(cam_up)