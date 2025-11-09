# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import cv2

def draw_box_with_label_np(img_rgb: np.ndarray, box: List[int], label: str, color=(223,64,64), thickness=3):
    x1,y1,x2,y2 = [int(v) for v in box]
    cv2.rectangle(img_rgb, (x1,y1), (x2,y2), color, thickness)
    if label:
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(0, y1 - h - 6)
        cv2.rectangle(img_rgb, (x1, y), (x1 + w + 8, y + h + baseline + 6), color, -1)
        cv2.putText(img_rgb, label, (x1+4, y + h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def render_verdict_text(frames_total: int, fake_frames: int) -> str:
    ratio = (100.0 * fake_frames / max(1, frames_total))
    return f"Frames: {frames_total} | Fake-frames: {fake_frames} ({ratio:.1f}%)"

def average_threshold(infos: list) -> float:
    if not infos: return 0.5
    vals = [float(i.get("best_thr", 0.5)) for i in infos]
    return float(sum(vals) / max(1, len(vals)))
