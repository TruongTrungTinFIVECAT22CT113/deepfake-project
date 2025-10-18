import React, { useRef, useState } from "react";
import { analyzeVideo, AnalyzeOptions } from "../api";

export default function AnalyzerForm({
  onResult,
  setLoading,
}: {
  onResult: (r: Awaited<ReturnType<typeof analyzeVideo>>) => void;
  setLoading: (b: boolean) => void;
}): JSX.Element {
  const fileRef = useRef<HTMLInputElement>(null);
  const [faceCrop, setFaceCrop] = useState(true);
  const [autoThr, setAutoThr] = useState(true);
  const [thr, setThr] = useState(0.5);
  const [tta, setTta] = useState(2);
  const [thick, setThick] = useState(3);
  const [enableFilters, setEnableFilters] = useState(true);
  const [methodGate, setMethodGate] = useState(0.55);
  const [salDensity, setSalDensity] = useState(0.02);
  const [duration, setDuration] = useState<string>(""); // NEW

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const f = fileRef.current?.files?.[0];
    if (!f) { alert("Chọn video"); return; }

    const dur = duration.trim() === "" ? undefined : Number(duration);
    if (dur !== undefined && (isNaN(dur) || dur < 0)) {
      alert("Thời lượng phải là số giây >= 0 hoặc để trống.");
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
      duration_sec: dur, // NEW
    };

    setLoading(true);
    try {
      const res = await analyzeVideo(f, opts);
      onResult(res);
    } catch (err: any) {
      alert(err.message || "Lỗi phân tích");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{ display: "grid", gap: 12 }}>
      <input type="file" ref={fileRef} accept="video/*" />

      <label>
        <input type="checkbox" checked={faceCrop}
               onChange={(e) => setFaceCrop(e.target.checked)} /> Face crop
      </label>

      <label>
        <input type="checkbox" checked={autoThr}
               onChange={(e) => setAutoThr(e.target.checked)} /> Auto threshold
      </label>

      <label>
        Threshold: <b>{thr.toFixed(3)}</b>
        <input type="range" min="0.10" max="0.99" step="0.005"
               value={thr} onChange={(e) => setThr(parseFloat(e.target.value))}
               disabled={autoThr} />
      </label>

      <label>
        TTA: <input type="number" min={1} max={4}
                    value={tta} onChange={(e)=>setTta(parseInt(e.target.value || "1"))}/>
      </label>

      <label>
        Box thickness: <input type="number" min={1} max={8}
                              value={thick} onChange={(e)=>setThick(parseInt(e.target.value || "3"))}/>
      </label>

      <label>
        <input type="checkbox" checked={enableFilters}
               onChange={(e)=>setEnableFilters(e.target.checked)} /> Bật filter theo method
      </label>

      <label>
        Method gate: <input type="number" step="0.01"
                            value={methodGate} onChange={(e)=>setMethodGate(parseFloat(e.target.value || "0.55"))}/>
      </label>

      <label>
        Saliency density: <input type="number" step="0.005"
                                 value={salDensity} onChange={(e)=>setSalDensity(parseFloat(e.target.value || "0.02"))}/>
      </label>

      {/* NEW: thời lượng phân tích */}
      <label>
        Analyze duration (sec, optional):
        <input type="number" min={0} step="0.1"
               placeholder="(để trống = full video)"
               value={duration} onChange={(e)=>setDuration(e.target.value)} />
      </label>

      <button type="submit">Phân tích Video</button>
    </form>
  );
}
