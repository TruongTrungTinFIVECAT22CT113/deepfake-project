import React, { useRef, useState, useEffect, useCallback } from "react";
import { type AnalyzeResult } from "../api";

type NewFields = {
  frames_total?: number;
  fake_frames?: number;
  fake_ratio?: number;
  verdict_level?: "clean" | "suspect" | "fake";
  threshold_used?: number;
  thr_override_ignored?: boolean;
  detector_backend_used?: string;
  method_rows_total?: [string, number][];
  method_rows_fake?: [string, number][];
  method_rows?: [string, number][];
  frame_tags?: string[];
  fps?: number;
  duration_sec?: number;
  analyzed_start_sec?: number | null;
  analyzed_end_sec?: number | null;
  explanation_basic?: ExplanationBasic;
  processing_time_sec?: number | null;
};

type ExplanationBasic = {
  method: string;
  method_share: number;
  fake_ratio?: number;
  summary: string;
  artifacts: [string, string][];
};

const METHOD_PALETTE_DEFAULT = ["#ef6b6b","#f5c542","#f0734f","#a87232","#8b5cf6","#3b82f6","#6b7a96"];
const REAL_COLOR_DEFAULT = "#3ecf8e";
const METHOD_PALETTE_CB = ["#e8874a","#e8c84a","#8b5cf6","#0ea5e9","#d97706","#6366f1","#78716c"];
const REAL_COLOR_CB = "#4a9ee8";

function getIsColorblind() { return document.documentElement.getAttribute("data-theme") === "colorblind"; }
function getRealColor() { return getIsColorblind() ? REAL_COLOR_CB : REAL_COLOR_DEFAULT; }
function getFakeColor() { return getIsColorblind() ? "#e8874a" : "#ef6b6b"; }

function buildColorMap(tags: string[]): Record<string, string> {
  const isCB = getIsColorblind();
  const palette = isCB ? METHOD_PALETTE_CB : METHOD_PALETTE_DEFAULT;
  const realColor = isCB ? REAL_COLOR_CB : REAL_COLOR_DEFAULT;
  const counts: Record<string, number> = {};
  for (const t of tags || []) { if (!t || t === "Real") continue; counts[t] = (counts[t] ?? 0) + 1; }
  const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k]) => k);
  const cmap: Record<string, string> = {};
  ranked.forEach((name, i) => { cmap[name] = palette[Math.min(i, palette.length - 1)]; });
  cmap["Real"] = realColor;
  return cmap;
}

function formatTimeLabel(sec: number, isEnd = false) {
  if (isEnd && !Number.isInteger(sec)) { const v = Math.max(0, sec); if (v < 60) return `${v.toFixed(1)}s`; const m = Math.floor(v / 60); const s = (v % 60).toFixed(1); return `${m}:${Number(s) < 10 ? "0" + s : s}`; }
  const s = Math.max(0, Math.floor(sec)); const m = Math.floor(s / 60); const ss = String(s % 60).padStart(2, "0");
  return m > 0 ? `${m}:${ss}` : `${s}s`;
}
function chooseTickStep(d: number) { if (d <= 15) return 1; if (d <= 60) return 5; if (d <= 180) return 10; return 30; }

function formatProcessingTime(sec: number): string {
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(0).padStart(2, "0");
  return `${m}m${s}s`;
}

function formatAnalyzedRange(dur: number | undefined, aStart?: number | null, aEnd?: number | null, procSec?: number | null) {
  if (!dur || dur <= 0) return null;
  const timeSuffix = procSec != null && procSec > 0 ? ` (hết ${formatProcessingTime(procSec)})` : "";
  if (aStart == null && aEnd == null) return `Đã phân tích: toàn bộ video${timeSuffix}`;
  const start = aStart == null ? 0 : Math.max(0, aStart);
  const end = aEnd == null ? dur : Math.max(0, Math.min(dur, aEnd));
  if (end <= start) return `Đã phân tích: toàn bộ video${timeSuffix}`;
  const fmt = (s: number) => s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}:${String((s % 60).toFixed(1)).padStart(4, "0")}`;
  return `Đã phân tích: ${fmt(start)} → ${fmt(end)}${timeSuffix}`;
}

function buildStatsReport(r: any): string {
  const lines: string[] = [];
  lines.push("═══════════════════════════════════════");
  lines.push("  BÁO CÁO PHÂN TÍCH");
  lines.push("  Ngày: " + new Date().toLocaleString("vi-VN"));
  lines.push("═══════════════════════════════════════");
  lines.push("");

  if (r.verdict) lines.push(`Kết luận: ${r.verdict}`);
  if (r.frames_total != null) {
    const real = r.frames_total - (r.fake_frames ?? 0);
    lines.push(`Tổng khung hình có khuôn mặt: ${r.frames_total}`);
    lines.push(`Khung hình có Deepfake:  ${r.fake_frames ?? 0} (${((r.fake_ratio ?? 0) * 100).toFixed(1)}%)`);
    lines.push(`Khung hình có khuôn mặt thật: ${real} (${((1 - (r.fake_ratio ?? 0)) * 100).toFixed(1)}%)`);
  }
  if (r.duration_sec) lines.push(`Thời lượng video: ${r.duration_sec.toFixed(1)}s`);
  if (r.fps) lines.push(`FPS: ${r.fps}`);

  const aRange = formatAnalyzedRange(r.duration_sec, r.analyzed_start_sec, r.analyzed_end_sec, r.processing_time_sec);
  if (aRange) lines.push(aRange);

  const rows = r.method_rows_total?.length ? r.method_rows_total : r.method_rows_fake?.length ? r.method_rows_fake : r.method_rows || [];
  if (rows.length) {
    lines.push("");
    lines.push("── Các loại kỹ thuật phát hiện được ──");
    for (const [m, p] of rows) lines.push(`  ${m}: ${typeof p === "number" ? p.toFixed(1) : p}%`);
  }

  if (r.explanation_basic) {
    lines.push("");
    lines.push(`── Giải thích về: ${r.explanation_basic.method} ──`);
    lines.push(r.explanation_basic.summary);
    if (r.explanation_basic.artifacts?.length) {
      for (const [name, desc] of r.explanation_basic.artifacts) lines.push(`  • ${name}: ${desc}`);
    }
  }

  lines.push("");
  lines.push("═══════════════════════════════════════");
  return lines.join("\n");
}

export default function ResultPanel(props: {
  result?: AnalyzeResult | null;
  loading?: boolean;
  previewUrl?: string | null;
  previewDuration?: number | null;
  errorMsg?: string | null;
}): JSX.Element | null {
  const r = props.result;
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoClass, setVideoClass] = useState<string>("");
  const [aspectRatio, setAspectRatio] = useState<string>("16/9");

  function handleVideoMeta(e: React.SyntheticEvent<HTMLVideoElement>) {
    const v = e.currentTarget;
    if (v.videoWidth && v.videoHeight) {
      const ratio = v.videoWidth / v.videoHeight;
      if (ratio < 0.8) { setVideoClass("portrait"); setAspectRatio("9/16"); }
      else if (ratio < 1.4) { setVideoClass("classic"); setAspectRatio("4/3"); }
      else { setVideoClass(""); setAspectRatio("16/9"); }
    }
  }
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const [rate, setRate] = useState(1.0);
  const [showCompare, setShowCompare] = useState(false);

  // 1. Đồng bộ tốc độ phát cho cả hai video
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate;
    if (previewVideoRef.current) previewVideoRef.current.playbackRate = rate;
  }, [rate, showCompare]);

  // 2. Logic đồng bộ hóa hai chiều (Sync Logic)
  const syncVideos = useCallback((source: HTMLVideoElement, target: HTMLVideoElement | null) => {
    if (!showCompare || !target) return;
    
    // Đồng bộ thời gian nếu lệch > 0.1s
    if (Math.abs(target.currentTime - source.currentTime) > 0.1) {
      target.currentTime = source.currentTime;
    }

    // Đồng bộ trạng thái Play/Pause
    if (source.paused && !target.paused) target.pause();
    else if (!source.paused && target.paused) target.play();
  }, [showCompare]);

  const handleMainAction = () => syncVideos(videoRef.current!, previewVideoRef.current);
  const handlePreviewAction = () => syncVideos(previewVideoRef.current!, videoRef.current);

  function downloadReport() {
    if (!r) return;
    const text = buildStatsReport(r);
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bao-cao-deepfake-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ── Fake progress khi loading ────────────────────────────────────────────
  // Backend không có SSE/WebSocket → không có progress thực tế.
  // Dùng easing asymptotic: progress = 99 * (1 - e^(-t/τ))
  //   - Không bao giờ đứng cứng, luôn tăng (ngày càng chậm hơn)
  //   - τ (tau) = ước tính thời gian xử lý = videoDuration / 2 (baseline 2× realtime)
  //   - Ở t=τ: ~63%, t=2τ: ~86%, t=3τ: ~95%, không bao giờ đạt 100% cho đến khi xong thật
  const [progress, setProgress] = useState(0);
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const progressStartRef = useRef<number>(0);
  const progressTauRef = useRef<number>(30000);  // ms

  useEffect(() => {
    if (props.loading) {
      setProgress(0);
      progressStartRef.current = Date.now();
      // τ = videoDuration/2 giây, tối thiểu 8s, tối đa 300s
      const videoDur = props.previewDuration ?? 60;
      progressTauRef.current = Math.min(300_000, Math.max(8_000, (videoDur / 2) * 1000));

      progressRef.current = setInterval(() => {
        const elapsed = Date.now() - progressStartRef.current;
        const tau = progressTauRef.current;
        // Asymptotic: 99 * (1 - e^(-t/τ)), tối đa 99%
        const pct = 99 * (1 - Math.exp(-elapsed / tau));
        setProgress(Math.min(99, pct));
      }, 250);
    } else {
      if (progressRef.current) {
        clearInterval(progressRef.current);
        progressRef.current = null;
      }
      if (progress > 0) {
        setProgress(100);
        const t = setTimeout(() => setProgress(0), 700);
        return () => clearTimeout(t);
      }
    }
    return () => {
      if (progressRef.current) { clearInterval(progressRef.current); progressRef.current = null; }
    };
  }, [props.loading]);

  if (props.loading) {
    const pct = Math.round(progress);
    return (
      <div className="stack" style={{ padding: "0.5rem 0" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.4rem" }}>
          <span style={{ fontSize: "0.9rem", fontWeight: 500, color: "var(--text)" }}>Đang phân tích video…</span>
          <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontVariantNumeric: "tabular-nums" }}>
            {pct}%
          </span>
        </div>

        {/* Track */}
        <div style={{
          height: "0.55rem", borderRadius: "1rem", background: "var(--surface-3)",
          border: "1px solid var(--border-subtle)", overflow: "hidden",
        }}>
          {/* Fill với transition mượt */}
          <div style={{
            height: "100%", borderRadius: "1rem",
            width: `${pct}%`,
            background: pct < 95
              ? "linear-gradient(90deg, var(--accent, #6366f1), #a78bfa)"
              : "linear-gradient(90deg, #22c55e, #4ade80)",
            transition: "width 0.18s ease-out, background 0.4s",
          }} />
        </div>

        <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
          {pct < 25 && "Đang nhận diện khuôn mặt…"}
          {pct >= 25 && pct < 55 && "Đang phân tích từng khung hình…"}
          {pct >= 55 && pct < 80 && "Đang tổng hợp kết quả…"}
          {pct >= 80 && pct < 99 && "Đang hoàn thiện, vui lòng chờ…"}
          {pct >= 99 && "Chuẩn bị trả về kết quả…"}
        </div>

        {/* Preview video mờ trong khi chờ */}
        {props.previewUrl && (
          <div style={{ opacity: 0.45, pointerEvents: "none", marginTop: "0.5rem" }}>
            <div className={`video-wrap ${videoClass}`} style={{ aspectRatio }}>
              <video src={props.previewUrl} className="media" muted onLoadedMetadata={handleVideoMeta} />
            </div>
          </div>
        )}
      </div>
    );
  }

  if (!r) {
    const errorBlock = props.errorMsg ? (
      <div className="warn" style={{ textAlign: "center" }}>{props.errorMsg}</div>
    ) : null;

    if (props.previewUrl) {
      return (
        <div className="stack">
          {errorBlock}
          <div style={{ display: "flex", alignItems: "baseline", gap: "0.75rem" }}>
            <div className="section-title">Xem trước</div>
            {props.previewDuration ? (
              <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                Thời lượng: {props.previewDuration.toFixed(1)} giây
              </span>
            ) : null}
          </div>
          <div className={`video-wrap ${videoClass}`} style={{ aspectRatio }}>
            <video src={props.previewUrl} className="media" controls onLoadedMetadata={handleVideoMeta} />
          </div>
          <div style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
            Nhấn <b style={{ color: "var(--text)" }}>Phân tích video</b> để bắt đầu.
          </div>
        </div>
      );
    }
    return (
      <div style={{ textAlign: "center", padding: "1.5rem 1rem", color: "var(--text-secondary)" }}>
        {errorBlock || (
          <>
            <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem", opacity: 0.4 }}>◉</div>
            <div style={{ fontSize: "1rem" }}>
              Tải lên video và nhấn <b>Phân tích</b> để bắt đầu.
            </div>
          </>
        )}
      </div>
    );
  }

  const fakeColor = getFakeColor();
  const realColor = getRealColor();

  const totalLine =
    r.frames_total != null
      ? <div style={{ fontSize: "0.9rem", fontWeight: 500 }}>Tổng: <b>{r.frames_total}</b> khung hình có khuôn mặt</div>
      : null;

  const framesLine =
    r.frames_total != null && r.fake_frames != null && r.fake_ratio != null
      ? (() => {
          const realFrames = r.frames_total - r.fake_frames;
          const realRatio = 1 - r.fake_ratio;
          return (
            <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", fontSize: "0.9rem", fontWeight: 500, alignItems: "center" }}>
              <span style={{ color: "var(--text-secondary)", fontWeight: 400 }}>Kết quả sau khi phân tích:</span>
              <span style={{ color: fakeColor }}>Có Deepfake: <b>{r.fake_frames}</b> ({(r.fake_ratio * 100).toFixed(1)}%)</span>
              <span style={{ color: realColor }}>Thật: <b>{realFrames}</b> ({(realRatio * 100).toFixed(1)}%)</span>
              {r.verdict_level === "clean" && (
                <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.3rem 0.75rem", borderRadius: "0.4rem", fontSize: "0.82rem", fontWeight: 600, background: "rgba(34,197,94,0.15)", color: "#22c55e", border: "1px solid rgba(34,197,94,0.3)" }}>
                  ✅ Không phát hiện dấu hiệu Deepfake (dưới 30%)
                </span>
              )}
              {r.verdict_level === "suspect" && (
                <span className="warn-badge">
                  ⚠️ Có dấu hiệu của Deepfake (từ 30-75%)
                </span>
              )}
              {r.verdict_level === "fake" && (
                <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.3rem 0.75rem", borderRadius: "0.4rem", fontSize: "0.82rem", fontWeight: 600, background: "rgba(239,68,68,0.15)", color: "#ef4444", border: "1px solid rgba(239,68,68,0.3)" }}>
                  🚨 Xác nhận: là có Deepfake bên trong video (trên 75%)
                </span>
              )}
            </div>
          );
        })()
      : null;

  const rows = r.method_rows_total?.length ? r.method_rows_total : r.method_rows_fake?.length ? r.method_rows_fake : r.method_rows || [];
  const title = r.method_rows_total?.length ? "Các loại kỹ thuật Deepfake phát hiện được" : r.method_rows_fake?.length ? "Phân bố phương pháp (khung hình giả)" : "Phân bố phương pháp";
  const tags: string[] = r.frame_tags || [];
  const totalFrames = typeof r.frames_total === "number" ? r.frames_total : tags?.length || 0;
  const cmap = buildColorMap(tags);
  const duration = typeof r.duration_sec === "number" && r.duration_sec > 0 ? r.duration_sec : totalFrames > 0 ? totalFrames / (typeof r.fps === "number" && r.fps > 0 ? r.fps : 25) : 0;
  const analyzedLabel = formatAnalyzedRange(duration, r.analyzed_start_sec ?? null, r.analyzed_end_sec ?? null, r.processing_time_sec ?? null);

  return (
    <div className="stack">
      <div className="section-title">Kết quả</div>

      {showCompare && props.previewUrl ? (
        <div className="compare-grid">
          <div className="compare-col">
            <div className="compare-label">Video đã tải lên</div>
            <div className={`video-wrap ${videoClass}`} style={{ aspectRatio }}>
              <video
                ref={previewVideoRef}
                src={props.previewUrl}
                className="media"
                controls
                onPlay={handlePreviewAction}
                onPause={handlePreviewAction}
                onLoadedMetadata={handleVideoMeta}
                onSeeked={handlePreviewAction}
              />
            </div>
          </div>
          <div className="compare-col">
            <div className="compare-label">Video đã phân tích</div>
            <div className={`video-wrap ${videoClass}`} style={{ aspectRatio }}>
              <video
                ref={videoRef}
                src={r.video_url}
                className="media"
                controls
                onPlay={handleMainAction}
                onPause={handleMainAction}
                onSeeked={handleMainAction}
              />
            </div>
          </div>
        </div>
      ) : (
        <div className={`video-wrap ${videoClass}`} style={{ aspectRatio }}>
          <video ref={videoRef} src={r.video_url} className="media" controls onLoadedMetadata={handleVideoMeta} />
        </div>
      )}

      <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
        <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
          Tốc độ phát: <b style={{ color: "var(--text)" }}>{rate.toFixed(2)}x</b>
        </div>
        <input type="range" min="0.25" max="2" step="0.05" value={rate}
          onChange={(e) => setRate(parseFloat(e.target.value))} style={{ width: "12rem" }} />
      </div>

      {totalLine}

      {analyzedLabel && <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>{analyzedLabel}</div>}

      {framesLine}

      {totalFrames > 0 && tags?.length === totalFrames ? (
        <details className="stack">
          <summary className="section-title" style={{ cursor: "pointer", listStyle: "none", outline: "none" }}>Dòng thời gian phân bổ khung hình của video</summary>
          <div style={{ display: "flex", alignItems: "stretch", height: "1rem", borderRadius: "0.5rem", overflow: "hidden", background: "var(--surface-3)", border: "1px solid var(--border-subtle)" }}
            title="Dòng thời gian theo từng khung hình">
            {tags.map((t, i) => (
              <span key={i} title={`Khung ${i + 1}: ${t}`} style={{ width: `${100 / totalFrames}%`, background: cmap[t] || "#374151" }} />
            ))}
          </div>

          {(() => {
            if (!duration || duration <= 0) return null;
            const step = chooseTickStep(duration);
            const ticks: number[] = [0];
            for (let t = step; t < duration - 1e-9; t += step) ticks.push(t);
            ticks.push(duration);
            return (
              <div style={{ position: "relative", height: "1.6rem", marginTop: "0.25rem" }}>
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 1, background: "var(--border-subtle)" }} />
                {ticks.map((t, i) => {
                  const leftPct = (t / duration) * 100;
                  const isFirst = i === 0; const isLast = i === ticks.length - 1;
                  const align = isFirst ? "left" : isLast ? "right" : "center";
                  const transform = align === "center" ? "translateX(-50%)" : align === "left" ? "translateX(0)" : "translateX(-100%)";
                  return (
                    <div key={i} style={{ position: "absolute", left: `${leftPct}%`, transform }}>
                      <div style={{ width: 1, height: "0.5rem", background: "var(--text-secondary)", opacity: 0.5 }} />
                      <div style={{ fontSize: "0.7rem", marginTop: "0.15rem", whiteSpace: "nowrap", color: "var(--text-secondary)" }}>{formatTimeLabel(t, isLast)}</div>
                    </div>
                  );
                })}
              </div>
            );
          })()}


        </details>
      ) : null}

      <div className={(rows?.length && r.explanation_basic) ? "explanation-layout" : "stack"}>
        {rows?.length ? (
          <details className="stack">
            <summary className="section-title" style={{ cursor: "pointer", listStyle: "none", outline: "none" }}>
              {title}
            </summary>
            <table className="pretty">
              <thead>
                <tr>
                  <th>Kỹ Thuật</th>
                  <th className="col-ratio">Tỷ lệ</th>
                  <th className="col-bar"></th> 
                </tr>
              </thead>
              <tbody>
                {rows.map(([m, p], i) => {
                  const methodColor = cmap[m] ?? null;
                  return (
                    <tr key={i}>
                      <td style={{ fontWeight: 500, whiteSpace: "nowrap" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                          {methodColor && (
                            <span style={{ width: "0.7rem", height: "0.7rem", background: methodColor, borderRadius: 3, display: "inline-block", flexShrink: 0 }} />
                          )}
                          {m}
                        </div>
                      </td>
                      <td className="col-ratio">
                        {typeof p === "number" ? p.toFixed(1) : p}%
                      </td>
                      <td className="col-bar">
                        <div className="bar">
                          <span style={{ width: `${Math.max(0, Math.min(100, Number(p)))}%`, background: methodColor || "var(--border-subtle)" }} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </details>
        ) : null}

        {r.explanation_basic && (
          <details className="stack">
            <summary className="section-title" style={{ cursor: "pointer", listStyle: "none", outline: "none" }}>
              Giải thích Về {r.explanation_basic.method}
            </summary>
            <div style={{ fontSize: "0.9rem", lineHeight: 1.7, color: "var(--text-secondary)", marginTop: "0.5rem" }}>
              <p>
                Kỹ thuật <b style={{ color: "var(--text)" }}>{r.explanation_basic.method}</b> chiếm{" "}
                <b style={{ color: "var(--text)" }}>{r.explanation_basic.method_share.toFixed(1)}%</b> số khung hình.
              </p>
              <p>{r.explanation_basic.summary}</p>
              <ul style={{ paddingLeft: "1.2rem" }}>
                {r.explanation_basic.artifacts.map(([name, desc]) => (
                  <li key={name} style={{ marginBottom: "0.25rem" }}>
                    <b style={{ color: "var(--text)" }}>{name}:</b> {desc}
                  </li>
                ))}
              </ul>
            </div>
          </details>
        )}
      </div>

      <div className="actions">
        <a className="btn small" href={r.video_url} download>Tải video đã phân tích</a>
        <button className="btn small btn-ghost" onClick={downloadReport}>Tải báo cáo thống kê</button>
        {/* {props.previewUrl && (
          <button className="btn small btn-ghost" onClick={() => setShowCompare(!showCompare)}>
            {showCompare ? "Ẩn so sánh" : "So sánh Video đã tải lên / đã phân tích"}
          </button>
        )} */}
      </div>
    </div>
  );
}