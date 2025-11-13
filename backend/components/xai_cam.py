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


# ----------------- COMMON UTILS -----------------

def _normalize_cam(cam: torch.Tensor) -> np.ndarray:
    """
    cam: [H, W] tensor
    -> numpy 0..1
    """
    cam = cam.detach()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam.cpu().numpy()

def overlay_heatmap_on_bgr(frame_bgr: np.ndarray, cam_2d: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    cam_resized = cv2.resize(cam_2d, (frame_bgr.shape[1], frame_bgr.shape[0]))

    # Giữ trong [0,1] và dùng gamma < 1 để làm nổi vùng có giá trị cao
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    cam_resized = np.power(cam_resized, 0.6)  # vùng nóng -> đỏ/vàng rõ hơn

    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(heat, alpha, frame_bgr, 1 - alpha, 0)
    return out

def save_cam_image(frame_bgr: np.ndarray, cam_2d: np.ndarray, out_dir: Optional[str] = None) -> str:
    """
    Lưu ảnh overlay, trả về path
    """
    merged = overlay_heatmap_on_bgr(frame_bgr, cam_2d)
    if out_dir is None:
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    else:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "xai_cam.png")
    cv2.imwrite(out_path, merged)
    return out_path


# ----------------- CNN GRAD-CAM -----------------

def generate_cam_cnn(
    model: nn.Module,
    input_tensor: torch.Tensor,   # [1,3,H,W] normalized
    target_index: int = 1,        # lớp "fake"
    target_layer: Optional[nn.Module] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Grad-CAM cho CNN:
      - target_layer: layer conv cuối cùng (vd model.layer4[-1] của ResNet)
    Trả: cam [H,W] (tensor 0..1)
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    if target_layer is None:
        raise ValueError("Bạn phải truyền target_layer (conv cuối của CNN) vào generate_cam_cnn")

    features = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        features["value"] = out

    def bwd_hook(module, gin, gout):
        gradients["value"] = gout[0]

    # gắn hook
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)

    with torch.enable_grad():
        out = model(input_tensor)
        # Multi-head: (logits_bin, logits_method, extra, ...)
        if isinstance(out, (list, tuple)):
            logits_bin = out[0]
        else:
            logits_bin = out
        score = logits_bin[0, target_index]
        model.zero_grad()
        score.backward()

    # bỏ hook
    h1.remove()
    h2.remove()

    # lấy feats + grads
    feat = features["value"]     # [1,C,h,w]
    grad = gradients["value"]    # [1,C,h,w]

    # global average pooling qua H,W để lấy weight cho từng channel
    weights = grad.mean(dim=(2, 3), keepdim=True)   # [1,C,1,1]
    cam = (weights * feat).sum(dim=1).squeeze(0)    # [h,w]

    cam = F.relu(cam)
    cam = _normalize_cam(cam)
    return cam  # [h,w] float 0..1


# ----------------- ViT GRAD-CAM -----------------

def generate_cam_vit(
    vit_model: nn.Module,
    input_tensor: torch.Tensor,    # [1,3,H,W] normalized
    target_index: int = 1,
    device: str = "cuda",
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Grad-CAM dạng đơn giản cho ViT timm:
      - hook vào block cuối (transformer encoder)
      - dùng patch token (bỏ cls token)
      - nếu map Grad-CAM gần như phẳng -> fallback sang 'feature energy'
    """
    vit_model.eval()
    input_tensor = input_tensor.to(device)
    backbone = getattr(vit_model, "backbone", vit_model)

    if not hasattr(backbone, "blocks"):
        raise ValueError("ViT backbone không có thuộc tính .blocks (mong đợi timm VisionTransformer).")

    # Kích thước input & lưới patch
    _, _, H, W = input_tensor.shape
    h_p = H // patch_size
    w_p = W // patch_size

    feats: dict = {}
    grads: dict = {}
    last_block = backbone.blocks[-1]

    def fwd_hook(module, inp, out):
        # out: [B, N, C] (N = 1 + h_p*w_p)
        feats["value"] = out

    def bwd_hook(module, gin, gout):
        grads["value"] = gout[0]

    h1 = last_block.register_forward_hook(fwd_hook)
    h2 = last_block.register_backward_hook(bwd_hook)

    # ------- Bước 1: Grad-CAM chuẩn -------
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
    tokens = tokens[:, 1:, :]          # [1, h*w, C]
    grad_tokens = grad_tokens[:, 1:, :]  # [1, h*w, C]

    # weights: global avg over tokens
    weights = grad_tokens.mean(dim=1, keepdim=True)        # [1,1,C]
    cam_patch = torch.bmm(tokens, weights.transpose(1, 2)) # [1, h*w, 1]
    cam_patch = cam_patch.view(1, h_p, w_p)                # [1,h_p,w_p]
    cam_patch = F.relu(cam_patch)

    cam_up = F.interpolate(
        cam_patch.unsqueeze(1),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)  # [H,W]

    # Nếu map gần như phẳng (khác biệt rất nhỏ) -> fallback
    contrast = float((cam_up.max() - cam_up.min()).item())

    if contrast < 1e-6:
        # ------- Bước 2: Fallback = feature-energy map -------
        with torch.no_grad():
            feats2: dict = {}
            def fwd_hook2(module, inp, out):
                feats2["value"] = out

            h3 = last_block.register_forward_hook(fwd_hook2)
            _ = vit_model(input_tensor.to(device))
            h3.remove()

            tokens2 = feats2["value"][:, 1:, :]  # [1,h*w,C]
            cam_patch2 = tokens2.pow(2).sum(dim=-1).view(1, h_p, w_p)  # L2 energy

            cam_up = F.interpolate(
                cam_patch2.unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

    # Chuẩn hoá 0..1 và trả về numpy
    cam_norm = _normalize_cam(cam_up)
    return cam_norm  # [H,W] float 0..1