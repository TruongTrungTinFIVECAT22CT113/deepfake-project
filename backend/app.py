import os, tempfile
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, Body, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2  # NEW: dùng để cắt N giây đầu

from components.model import load_multiple_detectors, discover_checkpoints
from components.inference import analyze_video

app = FastAPI(title="Deepfake Detect API", version="1.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

_DETECTORS_INFO: List[Any] = []
_MODELS_META: List[Dict[str, Any]] = []
_METHOD_NAMES: List[str] | None = None
_DET_THR: float = 0.5
_IMG_SIZE: int = 512

def _recompute_enabled_indices(enabled_ids: List[str]) -> List[int]:
    idxs = []
    for i, meta in enumerate(_MODELS_META):
        if meta["id"] in enabled_ids:
            idxs.append(i)
    if not idxs:
        idxs = [i for i, m in enumerate(_MODELS_META) if m["enabled"]]
    return idxs

@app.on_event("startup")
def _load_models():
    global _DETECTORS_INFO, _MODELS_META, _METHOD_NAMES, _DET_THR, _IMG_SIZE
    ckpt_paths = discover_checkpoints()
    if not ckpt_paths:
        raise RuntimeError(
            "Không tìm thấy checkpoint (*.pt). Hãy đặt vào backend/models/** "
            "hoặc deepfake_detector/models/** (ưu tiên 'detector_best.pt')."
        )

    _DETECTORS_INFO = load_multiple_detectors(ckpt_paths)
    _METHOD_NAMES = _DETECTORS_INFO[0][4]
    _IMG_SIZE     = _DETECTORS_INFO[0][5]
    thr_model     = _DETECTORS_INFO[0][6]
    _DET_THR      = float(thr_model if thr_model is not None else 0.5)

    # Build metadata từ đường dẫn ckpt_paths
    _MODELS_META = []
    for i, p in enumerate(ckpt_paths):
        parent = os.path.basename(os.path.dirname(p))
        grand  = os.path.basename(os.path.dirname(os.path.dirname(p)))
        label = grand if parent.lower() == "checkpoints" else parent
        if not label:
            label = os.path.basename(p)
        mnames = list(_DETECTORS_INFO[i][4] or [])
        _MODELS_META.append({
            "id": f"m{i+1}",
            "path": p,
            "name": label,
            "enabled": True,
            "schema": {"method_names": mnames, "img_size": int(_DETECTORS_INFO[i][5])},
        })

@app.get("/api/health")
def health():
    if not _DETECTORS_INFO:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {
        "status": "ok",
        "methods": _METHOD_NAMES,
        "img_size": _IMG_SIZE,
        "thr": _DET_THR,
        "models": [{"id": m["id"], "name": m["name"], "enabled": m["enabled"]} for m in _MODELS_META]
    }

@app.get("/api/models")
def list_models():
    return [{"id": m["id"], "name": m["name"], "enabled": m["enabled"], "schema": m["schema"]} for m in _MODELS_META]

@app.post("/api/models/set-enabled")
def set_models_enabled(payload: Dict[str, List[str]] = Body(...)):
    ids = payload.get("enabled_ids", [])
    if not ids:
        return JSONResponse({"error": "Cần bật ít nhất 1 model."}, status_code=400)
    for m in _MODELS_META:
        m["enabled"] = (m["id"] in ids)
    if sum(1 for m in _MODELS_META if m["enabled"]) < 1:
        return JSONResponse({"error": "Phải bật ≥ 1 model."}, status_code=400)
    return [{"id": m["id"], "name": m["name"], "enabled": m["enabled"], "schema": m["schema"]} for m in _MODELS_META]

def _clip_first_seconds(src_path: str, seconds: float) -> str | None:
    """Tạo 1 file mp4 chỉ gồm N giây đầu. Nếu seconds <=0 hoặc lỗi → trả None (dùng full)."""
    try:
        if seconds is None or seconds <= 0:
            return None
        cap = cv2.VideoCapture(src_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0:
            cap.release()
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
                break  # video ngắn hơn seconds → ghi hết
            writer.write(frame)
            count += 1

        writer.release()
        cap.release()

        # Nếu không ghi được khung nào → bỏ cắt
        if count == 0:
            try: os.remove(out_path)
            except Exception: pass
            return None
        return out_path
    except Exception:
        return None

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),

    face_crop: bool = Form(True),
    auto_thr: bool = Form(True),
    thr: float = Form(0.5),
    tta: int = Form(2),
    thickness: int = Form(3),
    enable_filters: bool = Form(True),
    method_gate: float = Form(0.55),
    saliency_density: float = Form(0.02),

    enabled_ids_csv: str = Form("", description="Danh sách id model bật, phân tách bởi dấu phẩy (vd: m1,m3)"),
    duration_sec: float | None = Form(None, description="Giới hạn số giây đầu để phân tích (None = full)"),  # NEW
):
    if not _DETECTORS_INFO:
        return JSONResponse({"error": "Model chưa sẵn sàng"}, status_code=503)

    if enabled_ids_csv.strip():
        idxs = _recompute_enabled_indices([s.strip() for s in enabled_ids_csv.split(",") if s.strip()])
    else:
        idxs = [i for i, m in enumerate(_MODELS_META) if m["enabled"]]
    if len(idxs) < 1:
        return JSONResponse({"error": "Phải bật ≥ 1 model."}, status_code=400)

    ckpts_chosen = [_DETECTORS_INFO[i] for i in idxs]

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await file.read())
        src_path = f.name

    clip_path = None
    try:
        # Nếu người dùng yêu cầu chỉ phân tích N giây đầu → cắt trước khi phân tích
        if duration_sec is not None:
            clip_path = _clip_first_seconds(src_path, float(duration_sec))
        use_path = clip_path or src_path

        override_thr = None if auto_thr else float(thr)
        out_path, verdict, fr_bar_html, method_rows = analyze_video(
            use_path, ckpts_chosen, _METHOD_NAMES, _DET_THR,
            use_face_crop=bool(face_crop),
            override_thr=override_thr,
            tta=int(tta),
            box_thickness=int(thickness),
            method_gate=float(method_gate),
            enable_filters=bool(enable_filters),
            saliency_density=float(saliency_density),
            saliency_mode="method",  # giữ mặc định "method" (FE đã bỏ lựa chọn này)
        )

        if not out_path:
            return JSONResponse({"error": verdict or "Phân tích thất bại."}, status_code=400)

        token = os.path.basename(os.path.dirname(out_path))
        fname = "result.mp4"
        new_path = os.path.join(os.path.dirname(out_path), fname)
        if out_path != new_path:
            try: os.replace(out_path, new_path)
            except Exception: pass

        return {
            "verdict": verdict,
            "fake_real_bar_html": fr_bar_html,
            "method_rows": method_rows,
            "video_url": f"/api/download/{token}/{fname}"
        }
    finally:
        try: os.remove(src_path)
        except Exception: pass
        if clip_path:
            try: os.remove(clip_path)
            except Exception: pass

@app.get("/api/download/{token}/{filename}")
def download(token: str, filename: str):
    tmp_root = tempfile.gettempdir()
    target_dir = None
    for d in os.listdir(tmp_root):
        if d.startswith("df_web_") and token in d:
            target_dir = os.path.join(tmp_root, d); break
    if not target_dir:
        return JSONResponse({"error": "Không tìm thấy file."}, status_code=404)
    path = os.path.join(target_dir, filename)
    if not os.path.isfile(path):
        return JSONResponse({"error": "File không tồn tại."}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=filename)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
