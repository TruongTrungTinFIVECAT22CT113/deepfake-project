# -*- coding: utf-8 -*-
from __future__ import annotations
import os, tempfile
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import torch

from components.model import load_multiple_detectors, discover_checkpoints
from components.inference import analyze_video
from components.face_detection import retinaface_available
from components.utils import average_threshold

app = FastAPI(title="Deepfake Detect API (Strict Pipeline)", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

_DETECTORS_INFO: List[dict] = []
_MODELS_META: List[Dict[str, Any]] = []
_METHOD_NAMES: List[str] | None = None

def _clip_first_seconds(src_path: str, seconds: float) -> Optional[str]:
    try:
        if seconds is None or seconds <= 0:
            return None
        cap = cv2.VideoCapture(src_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0:
            cap.release()
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            cap.release()
            return None

        max_frames = int(seconds * fps + 0.5)
        if max_frames <= 0:
            cap.release()
            return None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        count = 0
        while count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            count += 1

        writer.release()
        cap.release()
        if count == 0:
            try:
                os.remove(out_path)
            except Exception:
                pass
            return None
        return out_path
    except Exception:
        return None

@app.on_event("startup")
def _load_models():
    global _DETECTORS_INFO, _MODELS_META, _METHOD_NAMES
    ckpt_paths = discover_checkpoints()
    if not ckpt_paths:
        raise RuntimeError("Không tìm thấy checkpoint (*.pt). Đặt ở backend/models/** hoặc deepfake_detector/models/**")

    _DETECTORS_INFO = load_multiple_detectors(ckpt_paths)
    _METHOD_NAMES = list(_DETECTORS_INFO[0]["method_names"] or [])

    _MODELS_META = []
    for i, info in enumerate(_DETECTORS_INFO):
        p = info["ckpt_path"]
        parent = os.path.basename(os.path.dirname(p))
        grand  = os.path.basename(os.path.dirname(os.path.dirname(p)))
        name = grand if parent.lower()=="checkpoints" else parent or os.path.basename(p)
        mnames = list(info["method_names"] or [])
        _MODELS_META.append({
            "id": f"m{i+1}",
            "path": p,
            "name": name,
            "enabled": True,
            "schema": {"method_names": mnames, "img_size": int(info["img_size"])},
            "best_thr": float(info["best_thr"]),
        })

@app.get("/api/health")
def health():
    if not _DETECTORS_INFO:
        return JSONResponse({"status":"loading"}, status_code=503)
    thr_mode = "single" if len([m for m in _MODELS_META if m["enabled"]]) == 1 else "average"
    thr_val = None
    enabled_idxs = [i for i,m in enumerate(_MODELS_META) if m["enabled"]]
    if enabled_idxs:
        infos = [_DETECTORS_INFO[i] for i in enabled_idxs]
        thr_val = float(infos[0]["best_thr"]) if len(infos)==1 else average_threshold(infos)
    return {
        "status":"ok",
        "methods": _METHOD_NAMES,
        "retinaface_available": retinaface_available(),
        "models":[{"id":m["id"],"name":m["name"],"enabled":m["enabled"],"best_thr":m["best_thr"]} for m in _MODELS_META],
        "threshold_mode": thr_mode,
        "threshold_default": thr_val,
    }

@app.get("/api/models")
def list_models():
    return [{"id": m["id"], "name": m["name"], "enabled": m["enabled"], "schema": m["schema"], "best_thr": m["best_thr"]} for m in _MODELS_META]

@app.post("/api/models/set-enabled")
def set_models_enabled(payload: Dict[str, List[str]] = Body(...)):
    ids = payload.get("enabled_ids", [])
    if not ids:
        return JSONResponse({"error":"Cần bật ít nhất 1 model."}, status_code=400)
    for m in _MODELS_META:
        m["enabled"] = (m["id"] in ids)
    if sum(1 for m in _MODELS_META if m["enabled"]) < 1:
        return JSONResponse({"error":"Phải bật ≥ 1 model."}, status_code=400)
    return [{"id": m["id"], "name": m["name"], "enabled": m["enabled"], "schema": m["schema"], "best_thr": m["best_thr"]} for m in _MODELS_META]

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    # Advanced only (FE): detector + bbox_scale + optional thr override
    detector_backend: str = Form("retinaface"),   # "retinaface" | "mediapipe"
    bbox_scale: float = Form(1.10),
    thr: float | None = Form(None),               # FE override; ignored if >=2 models
    # Other
    thickness: int = Form(3),
    duration_sec: float | None = Form(None),
):
    if not _DETECTORS_INFO:
        return JSONResponse({"error":"Model chưa sẵn sàng"}, status_code=503)

    # choose enabled models
    enabled_idxs = [i for i,m in enumerate(_MODELS_META) if m["enabled"]]
    if not enabled_idxs:
        return JSONResponse({"error":"Phải bật ≥ 1 model."}, status_code=400)
    chosen = [_DETECTORS_INFO[i] for i in enabled_idxs]

    # select threshold per rule
    fe_thr = float(thr) if (thr is not None and thr != "") else None
    if len(chosen) == 1:
        thr_used = float(fe_thr) if (fe_thr is not None) else float(chosen[0]["best_thr"])
        thr_override_ignored = False
    else:
        thr_used = float(sum(ci["best_thr"] for ci in chosen) / len(chosen))
        thr_override_ignored = True

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await file.read())
        src_path = f.name

    clip_path = None
    try:
        if duration_sec is not None:
            clip_path = _clip_first_seconds(src_path, float(duration_sec))
        use_path = clip_path or src_path

        out_path, verdict, stats, method_rows = analyze_video(
            use_path,
            chosen,
            _METHOD_NAMES or [],
            fe_thr_override=None if thr_override_ignored else fe_thr,
            detector_backend=detector_backend,
            bbox_scale=float(bbox_scale),
            det_thr=0.5,
            box_thickness=int(thickness),
        )

        if not out_path:
            return JSONResponse({"error": verdict or "Phân tích thất bại."}, status_code=400)

        token = os.path.basename(os.path.dirname(out_path))
        fname = "result.mp4"
        final_path = os.path.join(os.path.dirname(out_path), fname)
        if out_path != final_path:
            try:
                os.replace(out_path, final_path)
            except Exception:
                pass

        return {
            "verdict": verdict,
            "video_url": f"/api/download/{token}/{fname}",
            "frames_total": stats.get("frames_total", 0),
            "fake_frames": stats.get("fake_frames", 0),
            "fake_ratio": stats.get("fake_ratio", 0.0),
            "fps": stats.get("fps", 0.0),
            "duration_sec": stats.get("duration_sec", 0.0),
            "threshold_used": float(thr_used),
            "thr_override_ignored": thr_override_ignored,
            "detector_backend_used": stats.get("detector_backend_used"),
            "method_rows": method_rows,
            "method_distribution": stats.get("method_distribution", {}),
        }
    finally:
        try:
            os.remove(src_path)
        except Exception:
            pass
        if clip_path:
            try:
                os.remove(clip_path)
            except Exception:
                pass

@app.get("/api/download/{token}/{filename}")
def download(token: str, filename: str):
    tmp_root = tempfile.gettempdir()
    target_dir = None
    for d in os.listdir(tmp_root):
        if d.startswith("df_web_") and token in d:
            target_dir = os.path.join(tmp_root, d)
            break
    if not target_dir:
        return JSONResponse({"error":"Không tìm thấy file."}, status_code=404)
    path = os.path.join(target_dir, filename)
    if not os.path.isfile(path):
        return JSONResponse({"error":"File không tồn tại."}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=filename)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8081, reload=True)
