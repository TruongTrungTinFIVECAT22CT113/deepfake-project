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

  // Basic
  const [duration, setDuration] = useState<string>("");

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

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const f = picked || fileRef.current?.files?.[0];
    if (!f) { addToast("Please choose a video file", "error"); return; }

    // duration parse
    const dur = duration.trim() === "" ? undefined : Number(duration);
    if (dur !== undefined && (isNaN(dur) || dur < 0)) {
      addToast("Duration must be >= 0 seconds or empty", "error");
      return;
    }

    // thr override parse (only allow when single model enabled)
    let thr: number | null = null;
    if (!isMultiModel && thrOverride.trim() !== "") {
      const t = Number(thrOverride.trim());
      if (!Number.isFinite(t) || t < 0 || t > 1) {
        addToast("Threshold override must be in [0,1]", "error");
        return;
      }
      thr = t;
    }

    const opts: AnalyzeOptions = {
      detector_backend: detectorBackend,
      bbox_scale: bboxScale,
      thickness: thickness,
      duration_sec: dur,
      enabled_ids_csv: enabledIds && enabledIds.length ? enabledIds.join(",") : undefined,
      thr: thr ?? undefined, // BE will ignore if multiple models
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
          <div className="row" style={{alignItems:"center"}}>
            <div>Duration (sec, optional)</div>
            <input
              type="number"
              min={0}
              step={0.1}
              placeholder="Empty = full video"
              value={duration}
              onChange={(e)=>setDuration(e.target.value)}
            />
          </div>
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
                RetinaFace (default)
              </button>
              <button type="button" aria-pressed={detectorBackend === "mediapipe"} onClick={()=>setDetectorBackend("mediapipe")}>
                MediaPipe
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
