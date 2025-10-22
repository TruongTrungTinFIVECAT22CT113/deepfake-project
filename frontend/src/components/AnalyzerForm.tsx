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
  const [picked, setPicked] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [fileInfo, setFileInfo] = useState<{ name: string; size: string; duration?: number } | null>(null);

  const [faceCrop, setFaceCrop] = useState(true);
  const [autoThr, setAutoThr] = useState(true);
  const [thr, setThr] = useState(0.5);
  const [tta, setTta] = useState(2);
  const [thick, setThick] = useState(3);
  const [enableFilters, setEnableFilters] = useState(true);
  const [methodGate, setMethodGate] = useState(0.55);
  const [salDensity, setSalDensity] = useState(0.02);
  const [duration, setDuration] = useState<string>("");

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
      const v = document.createElement('video');
      v.preload = 'metadata';
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
    input.addEventListener('change', onChange);
    return () => input.removeEventListener('change', onChange);
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const f = picked || fileRef.current?.files?.[0];
    if (!f) { addToast("Please choose a video file", "error"); return; }

    const dur = duration.trim() === "" ? undefined : Number(duration);
    if (dur !== undefined && (isNaN(dur) || dur < 0)) {
      addToast("Duration must be >= 0 seconds or empty", "error");
      return;
    }

    const opts: AnalyzeOptions = {
      face_crop: faceCrop,
      auto_thr: autoThr,
      thr,
      tta,
      thickness: thick,
      enable_filters: enableFilters,
      method_gate: methodGate,
      saliency_density: salDensity,
      duration_sec: dur,
      ...(enabledIds && enabledIds.length ? { enabled_ids_csv: enabledIds.join(",") } : {}),
    };

    setLoading(true);
    try {
      const res = await analyzeVideo(f, opts);
      onResult(res);
      addToast("Analysis complete", "success");
    } catch (err: any) {
      addToast(err?.message || "Analyze failed", "error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="stack">
      <input id="file" type="file" ref={fileRef} accept="video/*" style={{display:'none'}} />
      <div className="row">
        <div>Video file</div>
        <div
          className={`dropzone ${dragOver ? 'drag' : ''}`}
          onDragOver={(e)=>{ e.preventDefault(); setDragOver(true); }}
          onDragLeave={()=> setDragOver(false)}
          onDrop={(e)=>{ e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files?.[0]; if (f && f.type.startsWith('video/')) setFile(f); }}
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

      <div className="row">
        <div>Basic settings</div>
        <div className="stack">
          <label className="inline">
            <input type="checkbox" checked={faceCrop}
                   onChange={(e) => setFaceCrop(e.target.checked)} /> Face crop
          </label>

          <label className="inline">
            <input type="checkbox" checked={autoThr}
                   onChange={(e) => setAutoThr(e.target.checked)} /> Auto threshold
          </label>

          <div className="slider-row">
            <div className="muted" style={{marginBottom:6}}>Detection threshold</div>
            {(() => {
              const min = 0.10, max = 0.99; const pct = ((thr - min) / (max - min)) * 100; const style = { ['--x' as any]: `${pct}%` } as React.CSSProperties;
              return <div className="bubble" style={style}><b>{thr.toFixed(3)}</b></div>;
            })()}
            <input type="range" min="0.10" max="0.99" step="0.005"
                   value={thr} onChange={(e) => setThr(parseFloat(e.target.value))}
                   disabled={autoThr} />
            <div className="help">Disable auto threshold to adjust manually.</div>
          </div>

          <div className="row" style={{alignItems:'center'}}>
            <div>TTA</div>
            <input type="number" min={1} max={4}
                   value={tta} onChange={(e)=>setTta(parseInt(e.target.value || "1"))}/>
          </div>

          <div className="row" style={{alignItems:'center'}}>
            <div>Duration (sec, optional)</div>
            <input type="number" min={0} step={0.1}
                   placeholder="Empty = full video"
                   value={duration} onChange={(e)=>setDuration(e.target.value)} />
          </div>
        </div>
      </div>

      <details className="card">
        <summary>Advanced options</summary>
        <div className="stack">
          <div className="row">
            <div>Box thickness</div>
            <input type="number" min={1} max={8}
                   value={thick} onChange={(e)=>setThick(parseInt(e.target.value || "3"))}/>
          </div>

          <label className="inline">
            <input type="checkbox" checked={enableFilters}
                   onChange={(e)=>setEnableFilters(e.target.checked)} /> Enable method-based filters
          </label>

          <div className="row">
            <div>Method gate</div>
            <input type="number" step={0.01}
                   value={methodGate} onChange={(e)=>setMethodGate(parseFloat(e.target.value || "0.55"))}/>
          </div>

          <div className="row">
            <div>Saliency density</div>
            <input type="number" step={0.005}
                   value={salDensity} onChange={(e)=>setSalDensity(parseFloat(e.target.value || "0.02"))}/>
          </div>
        </div>
      </details>

      <div>
        <button type="submit" className="btn">Analyze Video</button>
      </div>
    </form>
  );
}
