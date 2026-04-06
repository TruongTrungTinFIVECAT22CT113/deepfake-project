import React, { useEffect, useRef, useState } from "react";
import { analyzeVideo, AnalyzeOptions, type ModelMeta } from "../api";
import { useToast } from "./Toast";
import type { ThemeId } from "../App";

const THEME_OPTIONS: { id: ThemeId; label: string }[] = [
  { id: "dark",       label: "Tối" },
  { id: "light",      label: "Sáng" },
  { id: "balanced",   label: "Cân bằng" },
  { id: "colorblind", label: "Mù màu" },
];

export default function AnalyzerForm({
  onResult, setLoading, enabledIds, models, theme, setTheme, onPreviewUrl, onPreviewDuration, onError,
}: {
  onResult: (r: Awaited<ReturnType<typeof analyzeVideo>>) => void;
  setLoading: (b: boolean) => void;
  enabledIds?: string[];
  models?: ModelMeta[];
  theme: ThemeId;
  setTheme: (t: ThemeId) => void;
  onPreviewUrl: (url: string | null) => void;
  onPreviewDuration: (d: number | null) => void;
  onError: (msg: string | null) => void;
}): JSX.Element {
  const fileRef = useRef<HTMLInputElement>(null);
  const { addToast } = useToast();

  const [picked, setPicked] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [fileInfo, setFileInfo] = useState<{ name: string; size: string; duration?: number } | null>(null);

  const [startTime, setStartTime] = useState<string>("");
  const [endTime, setEndTime] = useState<string>("");

  const [detectorBackend, setDetectorBackend] = useState<"retinaface" | "mediapipe">("retinaface");
  const [bboxScale, setBboxScale] = useState(1.1);
  const [thickness, setThickness] = useState(3);
  const [thrOverride, setThrOverride] = useState<string>("");

  const enabledList = enabledIds ?? [];
  const isMultiModel = enabledList.length >= 2;
  const enabledModels: ModelMeta[] = (models ?? []).filter((m) => enabledList.includes(m.id));

  const [xaiModelId, setXaiModelId] = useState<string>("auto");
  useEffect(() => {
    if (xaiModelId !== "auto" && !enabledList.includes(xaiModelId)) setXaiModelId("auto");
  }, [enabledIds, xaiModelId, enabledList]);

  const [xaiMode, setXaiMode] = useState<"none" | "full">("none");

  function formatBytes(b: number) {
    const units = ["B", "KB", "MB", "GB"];
    let i = 0; let v = b;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(1)} ${units[i]}`;
  }

  function setFile(f: File) {
    setPicked(f);
    setFileInfo({ name: f.name, size: formatBytes(f.size) });

    // Create preview URL for ResultPanel
    const previewBlobUrl = URL.createObjectURL(f);
    onPreviewUrl(previewBlobUrl);

    // Get duration
    try {
      const url = URL.createObjectURL(f);
      const v = document.createElement("video");
      v.preload = "metadata"; v.src = url;
      v.onloadedmetadata = () => {
        const dur = Number(v.duration || 0);
        setFileInfo({ name: f.name, size: formatBytes(f.size), duration: dur });
        onPreviewDuration(dur > 0 ? dur : null);
        URL.revokeObjectURL(url);
      };
    } catch {}
  }

  useEffect(() => {
    const input = fileRef.current;
    if (!input) return;
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
    if (!f) { onError("Vui lòng chọn một tệp video trước khi phân tích."); return; }

    onError(null); // clear previous error

    const startRaw = parseSec(startTime);
    const endRaw = parseSec(endTime);
    if (Number.isNaN(startRaw as number)) { addToast("Thời gian bắt đầu phải ≥ 0 giây hoặc để trống", "error"); return; }
    if (Number.isNaN(endRaw as number)) { addToast("Thời gian kết thúc phải ≥ 0 giây hoặc để trống", "error"); return; }

    let sVal = startRaw as number | undefined;
    let eVal = endRaw as number | undefined;
    const D = fileInfo?.duration && Number.isFinite(fileInfo.duration) ? fileInfo.duration! : undefined;
    if (D !== undefined) {
      const startOut = sVal !== undefined && sVal >= D - 1e-9;
      const endOut = eVal !== undefined && eVal >= D - 1e-9;
      if ((startOut && (eVal === undefined || endOut)) || (endOut && (sVal === undefined || startOut))) { sVal = undefined; eVal = undefined; }
      else { if (startOut) sVal = undefined; if (endOut) eVal = undefined; }
    }
    if (sVal !== undefined && eVal !== undefined && eVal <= sVal) { addToast("Thời gian kết thúc phải lớn hơn thời gian bắt đầu", "error"); return; }

    let selectedXaiModelId: string | undefined;
    if (xaiMode !== "none" && enabledList.length > 0) {
      selectedXaiModelId = (xaiModelId === "auto" || !enabledList.includes(xaiModelId)) ? enabledList[0] : xaiModelId;
    }

    const opts: AnalyzeOptions = {
      detector_backend: detectorBackend, bbox_scale: bboxScale, thickness,
      start_sec: sVal, end_sec: eVal,
      enabled_ids_csv: enabledIds && enabledIds.length ? enabledIds.join(",") : undefined,
      thr: !isMultiModel && thrOverride.trim() !== "" ? Number(thrOverride) : undefined,
      xai_mode: xaiMode, xai_model_id: selectedXaiModelId,
    };

    setLoading(true);
    try {
      const res = await analyzeVideo(f, opts);
      onResult(res);
      if (res?.thr_override_ignored) addToast("Đang bật nhiều mô hình → sử dụng ngưỡng trung bình; bỏ qua giá trị ghi đè.");
      else addToast("Phân tích hoàn tất", "success");
    } catch (err: any) {
      addToast(err?.message || "Phân tích thất bại", "error");
    } finally { setLoading(false); }
  }

  return (
    <form onSubmit={handleSubmit} className="stack">
      <input id="file" type="file" ref={fileRef} accept="video/*" style={{ display: "none" }} />

      {/* Tệp video */}
      <div className="row">
        <div style={{ fontWeight: 500 }}>Tệp video</div>
        <div
          className={`dropzone ${dragOver ? "drag" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files?.[0]; if (f && f.type.startsWith("video/")) setFile(f); }}
          onClick={() => fileRef.current?.click()}
        >
          <div className="meta">
            {fileInfo ? (
              <>
                <span style={{ fontWeight: 500, color: "var(--text)" }}>{fileInfo.name}</span>
                <span>{fileInfo.size}</span>
                {fileInfo.duration ? <span>{Math.round(fileInfo.duration)} giây</span> : null}
              </>
            ) : (
              <span>Nhấn để chọn hoặc kéo thả tệp video vào đây</span>
            )}
          </div>
        </div>
      </div>

      {/* Cài đặt nâng cao */}
      <details className="card">
        <summary>Cài đặt nâng cao</summary>
        <div className="stack">
          {/* Khoảng thời gian phân tích */}
          <div className="row">
            <div style={{ fontWeight: 500 }}>Khoảng thời gian</div>
            <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Từ</span>
                <input type="number" min={0} step={0.1} placeholder="Bắt đầu (s)" value={startTime}
                  onChange={(e) => setStartTime(e.target.value)} style={{ width: "7rem" }} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>đến</span>
                <input type="number" min={0} step={0.1} placeholder="Kết thúc (s)" value={endTime}
                  onChange={(e) => setEndTime(e.target.value)} style={{ width: "7rem" }} />
              </div>
              <span className="help">Để trống = toàn bộ video</span>
            </div>
          </div>

          <div className="row">
            <div style={{ fontWeight: 500 }}>Giao diện</div>
            <div className="segmented" role="group" aria-label="Chế độ giao diện">
              {THEME_OPTIONS.map((t) => (
                <button key={t.id} type="button" aria-pressed={theme === t.id} onClick={() => setTheme(t.id)}>{t.label}</button>
              ))}
            </div>
          </div>

          <div className="row">
            <div style={{ fontWeight: 500 }}>Bộ phát hiện khuôn mặt</div>
            <div className="segmented" role="group" aria-label="Bộ phát hiện">
              <button type="button" aria-pressed={detectorBackend === "retinaface"} onClick={() => setDetectorBackend("retinaface")}>RetinaFace (GPU)</button>
              <button type="button" aria-pressed={detectorBackend === "mediapipe"} onClick={() => setDetectorBackend("mediapipe")}>MediaPipe (CPU)</button>
            </div>
          </div>

          <div className="row">
            <div style={{ fontWeight: 500 }}>Tỷ lệ khung viền</div>
            <input type="number" step={0.01} min={1.0} max={1.6} value={bboxScale}
              onChange={(e) => setBboxScale(parseFloat(e.target.value || "1.10"))} />
          </div>

          <div className="row">
            <div style={{ fontWeight: 500 }}>Độ dày khung viền</div>
            <input type="number" min={1} max={8} value={thickness}
              onChange={(e) => setThickness(parseInt(e.target.value || "3"))} />
          </div>

          <div className="row">
            <div style={{ fontWeight: 500 }}>Ghi đè ngưỡng</div>
            <div>
              <input type="number" step={0.001} min={0} max={1} placeholder="0.00 – 1.00"
                value={thrOverride} onChange={(e) => setThrOverride(e.target.value)} disabled={isMultiModel} />
              {isMultiModel && (
                <div className="help" style={{ marginTop: "0.4rem" }}>
                  Không khả dụng khi bật nhiều mô hình (sử dụng ngưỡng tổng hợp).
                </div>
              )}
            </div>
          </div>

          <div className="row">
            <div style={{ fontWeight: 500 }}>Grad-CAM</div>
            <div className="segmented" role="group" aria-label="Bật/tắt Grad-CAM">
              <button type="button" aria-pressed={xaiMode === "none"} onClick={() => setXaiMode("none")}>Tắt</button>
              <button type="button" aria-pressed={xaiMode === "full"} onClick={() => setXaiMode("full")}>Bật</button>
            </div>
          </div>

          {xaiMode === "full" && enabledModels.length >= 2 && (
            <div className="row">
              <div style={{ fontWeight: 500 }}>Mô hình XAI</div>
              <div className="stack">
                <select value={xaiModelId} onChange={(e) => setXaiModelId(e.target.value)}>
                  <option value="auto">Tự động (mô hình bật đầu tiên)</option>
                  {enabledModels.map((m) => (<option key={m.id} value={m.id}>{m.name || m.id}</option>))}
                </select>
                <div className="help">Chỉ các mô hình đang bật mới có thể dùng cho trực quan hoá Grad-CAM.</div>
              </div>
            </div>
          )}
        </div>
      </details>

      <div>
        <button type="submit" className="btn">Phân tích video</button>
      </div>
    </form>
  );
}