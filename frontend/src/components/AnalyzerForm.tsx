import React, { useEffect, useRef, useState } from "react";
import { analyzeVideo, AnalyzeOptions } from "../api";
import { useToast } from "./Toast";

export default function AnalyzerForm({
  onResult,
  setLoading,
  enabledIds,
}: {
  onResult: (r: Awaited<ReturnType<typeof analyzeVideo>>) => void;
  setLoading: (b: boolean) => void;
  enabledIds?: string[];
}): JSX.Element {
  const fileRef = useRef<HTMLInputElement>(null);
  const { addToast } = useToast();

  // File
  const [picked, setPicked] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [fileInfo, setFileInfo] = useState<{ name: string; size: string; duration?: number } | null>(null);

  // Basic – thay Duration bằng Start/End (giây)
  const [startTime, setStartTime] = useState<string>(""); // giây, rỗng = từ đầu
  const [endTime, setEndTime] = useState<string>("");     // giây, rỗng = tới hết

  // Advanced
  const [detectorBackend, setDetectorBackend] = useState<"retinaface" | "mediapipe">("retinaface");
  const [bboxScale, setBboxScale] = useState(1.10);
  const [thickness, setThickness] = useState(3);
  const [thrOverride, setThrOverride] = useState<string>(""); // empty = not sending

  // Derived: multiple models?
  const numEnabled = enabledIds?.length ?? 0;
  const isMultiModel = numEnabled >= 2;

  function formatBytes(b: number) {
    const units = ["B","KB","MB","GB"]; let i = 0; let v = b;
    while (v >= 1024 && i < units.length-1) { v /= 1024; i++; }
    return `${v.toFixed(1)} ${units[i]}`;
  }

  function setFile(f: File) {
    setPicked(f);
    setFileInfo({ name: f.name, size: formatBytes(f.size) });
    try {
      const url = URL.createObjectURL(f);
      const v = document.createElement("video");
      v.preload = "metadata";
      v.src = url;
      v.onloadedmetadata = () => {
        setFileInfo({ name: f.name, size: formatBytes(f.size), duration: Number(v.duration || 0) });
        URL.revokeObjectURL(url);
      };
    } catch {}
  }

  useEffect(() => {
    const input = fileRef.current; if (!input) return;
    const onChange = () => { const f = input.files?.[0]; if (f) setFile(f); };
    input.addEventListener("change", onChange);
    return () => input.removeEventListener("change", onChange);
  }, []);

  function parseSec(s: string): number | undefined {
    const t = s.trim();
    if (t === "") return undefined;
    const v = Number(t);
    return Number.isFinite(v) && v >= 0 ? v : (NaN as any);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const f = picked || fileRef.current?.files?.[0];
    if (!f) { addToast("Please choose a video file", "error"); return; }

// parse & validate start/end
const startRaw = parseSec(startTime);
const endRaw = parseSec(endTime);

if (Number.isNaN(startRaw as number)) {
  addToast("Start Time must be ≥ 0 seconds or empty", "error");
  return;
}
if (Number.isNaN(endRaw as number)) {
  addToast("End Time must be ≥ 0 seconds or empty", "error");
  return;
}

// Chuẩn hoá theo duration (nếu có metadata)
let sVal: number | undefined = startRaw as number | undefined;
let eVal: number | undefined = endRaw as number | undefined;

const D = (fileInfo?.duration && Number.isFinite(fileInfo.duration)) ? fileInfo.duration! : undefined;

// Nếu biết D: xử lý các case vượt D
if (D !== undefined) {
  const startOut = (sVal !== undefined && sVal >= D - 1e-9);
  const endOut   = (eVal !== undefined && eVal >= D - 1e-9);

  // Nếu cả hai phía đều ngoài biên (hoặc một phía ngoài biên và phía kia rỗng) => xem như full video
  if ((startOut && (eVal === undefined || endOut)) || (endOut && (sVal === undefined || startOut))) {
    sVal = undefined;
    eVal = undefined;
  } else {
    // Chuẩn hoá riêng lẻ: phần nào vượt D thì bỏ (coi như không nhập)
    if (startOut) sVal = undefined;
    if (endOut)   eVal = undefined;
  }
}

// Case còn lại: nếu Start/End đều hợp lệ nội biên và End <= Start -> lỗi
if (sVal !== undefined && eVal !== undefined && eVal <= sVal) {
  addToast("End Time must be greater than Start Time", "error");
  return;
}

const opts: AnalyzeOptions = {
  detector_backend: detectorBackend,
  bbox_scale: bboxScale,
  thickness,
  start_sec: sVal,   // có thể undefined
  end_sec: eVal,     // có thể undefined
  enabled_ids_csv: enabledIds && enabledIds.length ? enabledIds.join(",") : undefined,
  thr: (!isMultiModel && thrOverride.trim() !== "") ? Number(thrOverride) : undefined,
};

    setLoading(true);
    try {
      const res = await analyzeVideo(f, opts);
      onResult(res);
      if (res?.thr_override_ignored) {
        addToast("Multiple models enabled → using average threshold; override ignored. Warning");
      } else {
        addToast("Analysis complete", "success");
      }
    } catch (err: any) {
      addToast(err?.message || "Analyze failed", "error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="stack">
      {/* Video input */}
      <input id="file" type="file" ref={fileRef} accept="video/*" style={{display:"none"}} />
      <div className="row">
        <div>Video file</div>
        <div
          className={`dropzone ${dragOver ? "drag" : ""}`}
          onDragOver={(e)=>{ e.preventDefault(); setDragOver(true); }}
          onDragLeave={()=> setDragOver(false)}
          onDrop={(e)=>{ e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files?.[0]; if (f && f.type.startsWith("video/")) setFile(f); }}
          onClick={()=> fileRef.current?.click()}
        >
          <div className="meta">
            {fileInfo ? (
              <>
                <span>📹 {fileInfo.name}</span>
                <span>• {fileInfo.size}</span>
                {fileInfo.duration ? <span>• {Math.round(fileInfo.duration)}s</span> : null}
              </>
            ) : (
              <span>Click to choose or drop a video file</span>
            )}
          </div>
        </div>
      </div>

      {/* Basic */}
      <div className="row">
        <div>Basic</div>
        <div className="stack">
          <div className="row" style={{alignItems:"center", gap: 12}}>
            <div style={{minWidth: 110}}>Start Time (s)</div>
            <input
              type="number"
              min={0}
              step={0.1}
              placeholder="Empty = from start"
              value={startTime}
              onChange={(e)=>setStartTime(e.target.value)}
              style={{ width: 160 }}
            />
          </div>
          <div className="row" style={{alignItems:"center", gap: 12}}>
            <div style={{minWidth: 110}}>End Time (s)</div>
            <input
              type="number"
              min={0}
              step={0.1}
              placeholder="Empty = to end"
              value={endTime}
              onChange={(e)=>setEndTime(e.target.value)}
              style={{ width: 160 }}
            />
          </div>
          {fileInfo?.duration ? (
            <div className="muted">Video duration: <b>{fileInfo.duration.toFixed(1)}s</b></div>
          ) : null}
        </div>
      </div>

      {/* Advanced */}
      <details className="card" open>
        <summary>Advanced settings</summary>
        <div className="stack">
          <div className="row">
            <div>Detector</div>
            <div className="segmented" role="group" aria-label="Detector">
              <button type="button" aria-pressed={detectorBackend === "retinaface"} onClick={()=>setDetectorBackend("retinaface")}>
                RetinaFace (for-GPU)
              </button>
              <button type="button" aria-pressed={detectorBackend === "mediapipe"} onClick={()=>setDetectorBackend("mediapipe")}>
                MediaPipe (CPU-only)
              </button>
            </div>
          </div>

          <div className="row">
            <div>BBox scale</div>
            <input type="number" step={0.01} min={1.0} max={1.6}
                   value={bboxScale} onChange={(e)=>setBboxScale(parseFloat(e.target.value || "1.10"))}/>
          </div>

          <div className="row">
            <div>Box thickness</div>
            <input type="number" min={1} max={8}
                   value={thickness} onChange={(e)=>setThickness(parseInt(e.target.value || "3"))}/>
          </div>

          <div className="row">
            <div>Threshold override</div>
            <input
              type="number"
              step={0.001}
              min={0}
              max={1}
              placeholder="0.00 - 1.00"
              value={thrOverride}
              onChange={(e)=>setThrOverride(e.target.value)}
              disabled={isMultiModel}
            />
          </div>

          {isMultiModel && (
            <div className="muted">
              Multiple models are enabled. The backend will use the <b>average threshold</b> and ignore override.
            </div>
          )}
        </div>
      </details>

      <div>
        <button type="submit" className="btn">Analyze Video</button>
      </div>
    </form>
  );
}
