import React, { useRef, useState, useEffect, useCallback } from "react";

type NewFields = {
  frames_total?: number;
  fake_frames?: number;
  fake_ratio?: number;
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

function formatAnalyzedRange(dur: number | undefined, aStart?: number | null, aEnd?: number | null) {
  if (!dur || dur <= 0) return null;
  if (aStart == null && aEnd == null) return "Đã phân tích: toàn bộ video";
  const start = aStart == null ? 0 : Math.max(0, aStart);
  const end = aEnd == null ? dur : Math.max(0, Math.min(dur, aEnd));
  if (end <= start) return "Đã phân tích: toàn bộ video";
  const fmt = (s: number) => s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}:${String((s % 60).toFixed(1)).padStart(4, "0")}`;
  return `Đã phân tích: ${fmt(start)} → ${fmt(end)}`;
}

// Generate stats report as text
function buildStatsReport(r: any): string {
  const lines: string[] = [];
  lines.push("═══════════════════════════════════════");
  lines.push("  BÁO CÁO PHÂN TÍCH DEEPFAKE");
  lines.push("  Ngày: " + new Date().toLocaleString("vi-VN"));
  lines.push("═══════════════════════════════════════");
  lines.push("");

  if (r.verdict) lines.push(`Kết luận: ${r.verdict}`);
  if (r.frames_total != null) {
    const real = r.frames_total - (r.fake_frames ?? 0);
    lines.push(`Tổng khung hình: ${r.frames_total}`);
    lines.push(`Khung hình giả:  ${r.fake_frames ?? 0} (${((r.fake_ratio ?? 0) * 100).toFixed(1)}%)`);
    lines.push(`Khung hình thật: ${real} (${((1 - (r.fake_ratio ?? 0)) * 100).toFixed(1)}%)`);
  }
  if (r.threshold_used != null) lines.push(`Ngưỡng sử dụng: ${r.threshold_used.toFixed(3)}`);
  if (r.detector_backend_used) lines.push(`Bộ phát hiện: ${r.detector_backend_used}`);
  if (r.duration_sec) lines.push(`Thời lượng video: ${r.duration_sec.toFixed(1)}s`);
  if (r.fps) lines.push(`FPS: ${r.fps}`);

  const aRange = formatAnalyzedRange(r.duration_sec, r.analyzed_start_sec, r.analyzed_end_sec);
  if (aRange) lines.push(aRange);

  const rows = r.method_rows_total?.length ? r.method_rows_total : r.method_rows_fake?.length ? r.method_rows_fake : r.method_rows || [];
  if (rows.length) {
    lines.push("");
    lines.push("── Phân bố phương pháp ──");
    for (const [m, p] of rows) lines.push(`  ${m}: ${typeof p === "number" ? p.toFixed(1) : p}%`);
  }

  if (r.explanation_basic) {
    lines.push("");
    lines.push(`── Giải thích: ${r.explanation_basic.method} ──`);
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
  result?: { verdict: string; video_url: string } & NewFields;
  loading?: boolean;
  previewUrl?: string | null;
  previewDuration?: number | null;
  errorMsg?: string | null;
}): JSX.Element | null {
  const r = props.result;
  const videoRef = useRef<HTMLVideoElement>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const [rate, setRate] = useState(1.0);
  const [showCompare, setShowCompare] = useState(false);

  useEffect(() => { if (videoRef.current) videoRef.current.playbackRate = rate; }, [rate]);

  // Sync playback in compare mode
  const syncPlay = useCallback(() => {
    if (!showCompare) return;
    const a = previewVideoRef.current;
    const b = videoRef.current;
    if (a && b) { b.currentTime = a.currentTime; b.play(); }
  }, [showCompare]);

  // Download stats report
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

  if (props.loading) {
    return (
      <div className="stack">
        <div className="skeleton box" />
        <div className="skeleton line" style={{ width: "60%" }} />
        <div className="skeleton line" style={{ width: "40%" }} />
      </div>
    );
  }

  if (!r) {
    const errorBlock = props.errorMsg ? (
      <div className="warn" style={{ textAlign: "center" }}>{props.errorMsg}</div>
    ) : null;

    // Show preview video if available, otherwise placeholder
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
          <video src={props.previewUrl} className="media" controls />
          <div style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
            Nhấn <b style={{ color: "var(--text)" }}>Phân tích video</b> để bắt đầu phát hiện deepfake.
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

  const framesLine =
    r.frames_total != null && r.fake_frames != null && r.fake_ratio != null
      ? (() => {
          const realFrames = r.frames_total - r.fake_frames;
          const realRatio = 1 - r.fake_ratio;
          return (
            <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", fontSize: "0.9rem", fontWeight: 500 }}>
              <span>Tổng: <b>{r.frames_total}</b> khung hình</span>
              <span style={{ color: fakeColor }}>Giả: <b>{r.fake_frames}</b> ({(r.fake_ratio * 100).toFixed(1)}%)</span>
              <span style={{ color: realColor }}>Thật: <b>{realFrames}</b> ({(realRatio * 100).toFixed(1)}%)</span>
            </div>
          );
        })()
      : null;

  const rows = r.method_rows_total?.length ? r.method_rows_total : r.method_rows_fake?.length ? r.method_rows_fake : r.method_rows || [];
  const title = r.method_rows_total?.length ? "Các loại kỹ thuật Deepfake được sử dụng" : r.method_rows_fake?.length ? "Phân bố phương pháp (khung hình giả)" : "Phân bố phương pháp";
  const tags: string[] = r.frame_tags || [];
  const totalFrames = typeof r.frames_total === "number" ? r.frames_total : tags?.length || 0;
  const cmap = buildColorMap(tags);
  const duration = typeof r.duration_sec === "number" && r.duration_sec > 0 ? r.duration_sec : totalFrames > 0 ? totalFrames / (typeof r.fps === "number" && r.fps > 0 ? r.fps : 25) : 0;
  const analyzedLabel = formatAnalyzedRange(duration, r.analyzed_start_sec ?? null, r.analyzed_end_sec ?? null);

  return (
    <div className="stack">
      <div className="section-title">Kết quả</div>

      {/* Compare mode or single video */}
      {showCompare && props.previewUrl ? (
        <div>
          <div className="compare-grid">
            <div className="compare-col">
              <div className="compare-label">Video gốc</div>
              <video ref={previewVideoRef} src={props.previewUrl} controls onPlay={syncPlay} />
            </div>
            <div className="compare-col">
              <div className="compare-label">Đã phân tích</div>
              <video ref={videoRef} src={r.video_url} controls />
            </div>
          </div>
          <button className="btn small btn-ghost" style={{ marginTop: "0.5rem" }} onClick={() => setShowCompare(false)}>
            Ẩn so sánh
          </button>
        </div>
      ) : (
        <video ref={videoRef} src={r.video_url} className="media" controls />
      )}

      <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
        <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
          Tốc độ phát: <b style={{ color: "var(--text)" }}>{rate.toFixed(2)}x</b>
        </div>
        <input type="range" min="0.25" max="2" step="0.05" value={rate}
          onChange={(e) => setRate(parseFloat(e.target.value))} style={{ width: "12rem" }} />
      </div>

      {framesLine}

      {analyzedLabel && <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>{analyzedLabel}</div>}

      {typeof r.threshold_used === "number" && (
        <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Ngưỡng đã sử dụng: <b>{r.threshold_used.toFixed(3)}</b></div>
      )}
      {r.detector_backend_used && (
        <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Bộ phát hiện khuôn mặt: <b>{r.detector_backend_used}</b></div>
      )}
      {r.thr_override_ignored && (
        <div className="warn">Đang có nhiều hơn 1 mô hình phân tích được bật.</div>
      )}

      {/* Timeline */}
      {totalFrames > 0 && tags?.length === totalFrames ? (
        <div className="stack">
          <div className="section-title">Dòng thời gian phân bổ khung hình của video</div>
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

          <div style={{ display: "flex", gap: "0.9rem", flexWrap: "wrap", marginTop: "0.25rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.82rem" }}>
              <span style={{ width: "0.65rem", height: "0.65rem", background: realColor, borderRadius: 3, display: "inline-block" }} />
              <span style={{ color: "var(--text-secondary)" }}>Thật</span>
            </div>
            {Object.entries(cmap).filter(([k]) => k !== "Real").map(([k, color]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.82rem" }}>
                <span style={{ width: "0.65rem", height: "0.65rem", background: color, borderRadius: 3, display: "inline-block" }} />
                <span style={{ color: "var(--text-secondary)" }}>{k}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* Method table */}
      {rows?.length ? (
        <div className="stack">
          <div className="section-title">{title}</div>
          <table className="pretty">
            <thead><tr><th>Kỹ Thuật</th><th style={{ width: "6rem" }}>Tỷ lệ</th><th></th></tr></thead>
            <tbody>
              {rows.map(([m, p], i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 500 }}>{m}</td>
                  <td>{typeof p === "number" ? p.toFixed(1) : p}%</td>
                  <td><div className="bar"><span style={{ width: `${Math.max(0, Math.min(100, Number(p)))}%` }} /></div></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {/* Explanation */}
      {r.explanation_basic && (
        <div className="stack">
          <div className="section-title">Giải thích Về {r.explanation_basic.method}</div>
          <div style={{ fontSize: "0.9rem", lineHeight: 1.7, color: "var(--text-secondary)" }}>
            <p>
              Kỹ thuật <b style={{ color: "var(--text)" }}>{r.explanation_basic.method}</b> chiếm{" "}
              <b style={{ color: "var(--text)" }}>{r.explanation_basic.method_share.toFixed(1)}%</b> số khung
              hình (tính trên toàn bộ video), tỷ lệ tổng số khung hình bị phát hiện là giả chiếm{" "}
              <b style={{ color: "var(--text)" }}>{((r.explanation_basic.fake_ratio ?? r.fake_ratio ?? 0) * 100).toFixed(1)}%</b>.
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
        </div>
      )}

      {/* Action buttons */}
      <div className="actions">
        <a className="btn small" href={r.video_url} download>Tải video đã phân tích</a>
        <button className="btn small btn-ghost" onClick={downloadReport}>Tải báo cáo thống kê</button>
        {props.previewUrl && (
          <button className="btn small btn-ghost" onClick={() => setShowCompare(!showCompare)}>
            {showCompare ? "Ẩn so sánh" : "So sánh gốc / đã phân tích"}
          </button>
        )}
        <button className="btn small btn-ghost"
          onClick={() => navigator.clipboard?.writeText(window.location.origin + r.video_url)}>
          Sao chép liên kết
        </button>
      </div>
    </div>
  );
}