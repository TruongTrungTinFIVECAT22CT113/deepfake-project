# -*- coding: utf-8 -*-
from __future__ import annotations
import os, tempfile
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import torch

from components.model import load_multiple_detectors, discover_checkpoints
from components.inference import analyze_video, ENSEMBLE_THR_DEFAULT
from components.face_detection import retinaface_available
from components.artifact_profiles import ARTIFACT_PROFILES

app = FastAPI(title="Deepfake Detect API (Strict Pipeline)", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

_DETECTORS_INFO: List[dict] = []
_MODELS_META: List[Dict[str, Any]] = []
_METHOD_NAMES: List[str] | None = None

def _clip_range(src_path: str, start_sec: float | None, end_sec: float | None) -> Optional[str]:
    """
    Trả về đường dẫn video tạm đã cắt theo [start_sec, end_sec].
    - start_sec None => từ đầu
    - end_sec None   => tới hết
    - Nếu không cắt được, trả None để dùng full video.
    """
    try:
        if (start_sec is None or start_sec < 0) and (end_sec is None or end_sec < 0):
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # chuyển sang frame index
        start_f = 0 if not start_sec or start_sec < 0 else int(start_sec * fps + 0.5)
        end_f = total_frames if (end_sec is None or end_sec < 0) else int(end_sec * fps + 0.5)
        start_f = max(0, min(start_f, total_frames))
        end_f = max(0, min(end_f, total_frames))
        if end_f <= start_f:
            cap.release()
            return None

        # seek tới start_f
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        fidx = start_f
        while fidx < end_f:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            fidx += 1

        writer.release()
        cap.release()

        if fidx <= start_f:
            try: os.remove(out_path)
            except Exception: pass
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
        return JSONResponse({"status": "loading"}, status_code=503)

    enabled_idxs = [i for i, m in enumerate(_MODELS_META) if m["enabled"]]
    thr_mode = "single" if len(enabled_idxs) == 1 else "ensemble"
    thr_val = None

    if enabled_idxs:
        infos = [_DETECTORS_INFO[i] for i in enabled_idxs]
        if len(infos) == 1:
            thr_val = float(infos[0]["best_thr"])
        else:
            thr_val = ENSEMBLE_THR_DEFAULT

    return {
        "status": "ok",
        "methods": _METHOD_NAMES,
        "retinaface_available": retinaface_available(),
        "models": [
            {
                "id": m["id"],
                "name": m["name"],
                "enabled": m["enabled"],
                "best_thr": m["best_thr"],
            }
            for m in _MODELS_META
        ],
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

def _build_method_rows(counts: Dict[str, int], total_frames: int) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    if total_frames <= 0: return rows
    for k, v in counts.items():
        rows.append((k, 100.0 * float(v) / float(total_frames)))
    rows.sort(key=lambda x: -x[1])
    return rows

def _build_method_rows_fake(counts: Dict[str, int]) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    s = sum(counts.values())
    if s <= 0: return rows
    for k, v in counts.items():
        rows.append((k, 100.0 * float(v) / float(s)))
    rows.sort(key=lambda x: -x[1])
    return rows

def _build_basic_explanation(
    method_rows_total: List[Tuple[str, float]],
    method_rows_fake: List[Tuple[str, float]],
    fake_ratio: float,
) -> Optional[Dict[str, Any]]:
    # Nếu real chiếm áp đảo, không cần giải thích chi tiết
    # 0.5 = ít nhất 50% frame bị phát hiện giả mới giải thích
    if fake_ratio < 0.75:
        return None

    rows = method_rows_total or method_rows_fake
    if not rows:
        return None

    top_method, top_pct = rows[0]
    profile = ARTIFACT_PROFILES.get(top_method)
    if profile is None:
        return None

    return {
        "method": top_method,
        "method_share": float(top_pct),
        "fake_ratio": float(fake_ratio),
        "summary": profile["summary"],
        "artifacts": profile["artifacts"],
    }

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    detector_backend: str = Form("retinaface"),
    bbox_scale: float = Form(1.10),
    thr: float | None = Form(None),
    thickness: int = Form(3),
    start_sec: float | None = Form(None),   # NEW
    end_sec: float | None = Form(None),     # NEW
    xai_mode: str = Form("none"),
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
        thr_used = ENSEMBLE_THR_DEFAULT
        thr_override_ignored = True
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await file.read())
        src_path = f.name

    clip_path = None
    try:
        clip_path = _clip_range(src_path, start_sec, end_sec)
        start_used = float(start_sec) if start_sec is not None else None
        end_used = float(end_sec) if end_sec is not None else None

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
            xai_mode=xai_mode,  # NEW
        )

        if not out_path:
            return JSONResponse({"error": verdict or "Phân tích thất bại."}, status_code=400)
        if (not out_path) or (not os.path.isfile(out_path)):
            return JSONResponse({"error": verdict or "Không tạo được video kết quả."}, status_code=500)

        # ---- Build both tables from counts to avoid ambiguity ----
        frames_total = int(stats.get("frames_total", 0))
        fake_frames  = int(stats.get("fake_frames", 0))
        counts: Dict[str, int] = {k: int(v) for k, v in (stats.get("method_distribution") or {}).items()}

        method_rows_total = _build_method_rows(counts, frames_total)
        method_rows_fake  = _build_method_rows_fake(counts)
        fake_ratio = float(stats.get("fake_ratio", 0.0))
        explanation_basic = _build_basic_explanation(
            method_rows_total,
            method_rows_fake,
            fake_ratio,
        )

        token = os.path.basename(os.path.dirname(out_path))
        fname = "result.mp4"
        final_path = os.path.join(os.path.dirname(out_path), fname)
        if out_path != final_path:
            try:
                os.replace(out_path, final_path)
            except Exception:
                pass

        # legacy 'method_rows' -> default to TOTAL (clearer)
        return {
            "verdict": verdict,
            "video_url": f"/api/download/{token}/{fname}",
            "frames_total": frames_total,
            "fake_frames": fake_frames,
            "fake_ratio": stats.get("fake_ratio", 0.0),
            "fps": stats.get("fps", 0.0),
            "duration_sec": stats.get("duration_sec", 0.0),
            "threshold_used": float(thr_used),
            "thr_override_ignored": thr_override_ignored,
            "detector_backend_used": stats.get("detector_backend_used"),
            "method_rows_total": method_rows_total,
            "method_rows_fake": method_rows_fake,
            "method_rows": method_rows_total,  # legacy
            "method_distribution": counts,
            "frame_tags": stats.get("frame_tags", []),
            "analyzed_start_sec": start_used if clip_path else None,
            "analyzed_end_sec": end_used if clip_path else None,
            "explanation_basic": explanation_basic,
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
