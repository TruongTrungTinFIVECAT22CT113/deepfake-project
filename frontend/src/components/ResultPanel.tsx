import React, { useRef, useState, useEffect } from "react";

type NewFields = {
  frames_total?: number;
  fake_frames?: number;
  fake_ratio?: number;    // 0..1
  threshold_used?: number;
  thr_override_ignored?: boolean;
  detector_backend_used?: string;
  method_rows_total?: [string, number][];
  method_rows_fake?: [string, number][];
  method_rows?: [string, number][];
  frame_tags?: string[];  // per-frame labels from BE
  fps?: number;
  duration_sec?: number;
};

// --- Colors: Real + ranked methods ---
const METHOD_PALETTE = ["#df4040", "#eab308", "#f97316", "#92400e", "#7c3aed", "#2563eb", "#6b7280"]; // đỏ, vàng, cam, nâu, tím, xanh dương, xám
const REAL_COLOR = "#40d078";

function buildColorMap(tags: string[]): Record<string, string> {
  // Rank methods by frequency (exclude "Real")
  const counts: Record<string, number> = {};
  for (const t of tags || []) {
    if (!t || t === "Real") continue;
    counts[t] = (counts[t] ?? 0) + 1;
  }
  const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k]) => k);
  const cmap: Record<string, string> = {};
  ranked.forEach((name, i) => { cmap[name] = METHOD_PALETTE[Math.min(i, METHOD_PALETTE.length - 1)]; });
  cmap["Real"] = REAL_COLOR;
  return cmap;
}

// --- Time ruler helpers ---
function formatTimeLabel(sec: number, isEnd = false) {
  // Tick cuối cùng: nếu không phải số nguyên giây thì hiện 1 chữ số thập phân để tránh trùng nhãn
  if (isEnd && !Number.isInteger(sec)) {
    const v = Math.max(0, sec);
    if (v < 60) return `${v.toFixed(1)}s`;
    const m = Math.floor(v / 60);
    const s = (v % 60).toFixed(1);
    return `${m}:${Number(s) < 10 ? "0" + s : s}`;
  }
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const ss = String(s % 60).padStart(2, "0");
  return m > 0 ? `${m}:${ss}` : `${s}s`;
}
function chooseTickStep(duration: number) {
  if (duration <= 15) return 1;
  if (duration <= 60) return 5;
  if (duration <= 180) return 10;
  return 30;
}

export default function ResultPanel(props: {
  result?: {
    verdict: string;
    video_url: string;
  } & NewFields;
  loading?: boolean;
}): JSX.Element | null {
  const r = props.result;

  const videoRef = useRef<HTMLVideoElement>(null);
  const [rate, setRate] = useState(1.0);
  useEffect(() => { if (videoRef.current) videoRef.current.playbackRate = rate; }, [rate]);

  if (props.loading) {
    return (
      <div className="stack">
        <div className="skeleton box" />
        <div className="skeleton line" />
        <div className="skeleton line" />
      </div>
    );
  }
  if (!r) return <div className="muted">No result yet. Upload a video and analyze.</div>;

  const framesLine =
    r.frames_total !== undefined &&
    r.fake_frames !== undefined &&
    r.fake_ratio !== undefined
      ? (() => {
          const realFrames = (r.frames_total as number) - (r.fake_frames as number);
          const realRatio = 1 - (r.fake_ratio as number);
          return (
            <span>
              <b>Total-frames:</b> {r.frames_total}{" "}
              | <b style={{ color: "#df4040" }}>Fake-frames:</b> {r.fake_frames}{" "}
              (<span style={{ color: "#df4040" }}>
                {((r.fake_ratio as number) * 100).toFixed(1)}%
              </span>)
              {" | "}
              <b style={{ color: "#40d078" }}>Real-frames:</b> {realFrames}{" "}
              (<span style={{ color: "#40d078" }}>
                {(realRatio * 100).toFixed(1)}%
              </span>)
            </span>
          );
        })()
      : null;

  // Prefer % of all frames; fallback -> % of fake frames; final fallback -> whatever BE provided
  const rows = (r.method_rows_total && r.method_rows_total.length ? r.method_rows_total
              : (r.method_rows_fake && r.method_rows_fake.length ? r.method_rows_fake
              : (r.method_rows || [])));

  const title = r.method_rows_total && r.method_rows_total.length
    ? "Method distribution (of all frames)"
    : (r.method_rows_fake && r.method_rows_fake.length
        ? "Method distribution (fake frames)"
        : "Method distribution");

  // Timeline data
  const tags: string[] = (r.frame_tags || []);
  const totalFrames = typeof r.frames_total === "number" ? r.frames_total : (tags?.length || 0);
  const cmap = buildColorMap(tags);

  return (
    <div className="stack">
      <video ref={videoRef} src={r.video_url} className="media" controls />

      <div>
        <div className="muted" style={{ marginBottom: 6 }}>
          Playback speed: <b>{rate.toFixed(2)}x</b>
        </div>
        <input
          type="range" min="0.25" max="2" step="0.05"
          value={rate} onChange={(e) => setRate(parseFloat(e.target.value))}
          style={{ width: "220px" }}
        />
      </div>

      {framesLine && <div style={{ fontWeight: 600 }}>{framesLine}</div>}

      {typeof r.threshold_used === "number" && (
        <div className="muted">Threshold used: <b>{r.threshold_used.toFixed(3)}</b></div>
      )}
      {r.detector_backend_used && (
        <div className="muted">Detector used: <b>{r.detector_backend_used}</b></div>
      )}
      {r.thr_override_ignored && (
        <div className="warn">Multiple models enabled → average threshold used (override ignored).</div>
      )}

      {/* Timeline per-frame */}
      {totalFrames > 0 && tags?.length === totalFrames ? (
        <div className="stack">
          <div className="section-title">Frame timeline</div>
          <div
            style={{
              display: "flex",
              alignItems: "stretch",
              height: 14,
              borderRadius: 6,
              overflow: "hidden",
              background: "#111827",
            }}
            title="Per-frame method/real timeline"
          >
            {tags.map((t, i) => (
              <span
                key={i}
                title={`Frame ${i + 1}: ${t}`}
                style={{
                  width: `${100 / totalFrames}%`,
                  background: cmap[t] || "#374151",
                }}
              />
            ))}
          </div>

          {/* Time ruler under the timeline */}
          {(() => {
            const duration =
              typeof r.duration_sec === "number" && r.duration_sec > 0
                ? r.duration_sec
                : (totalFrames > 0
                    ? totalFrames / (typeof r.fps === "number" && r.fps > 0 ? r.fps : 25)
                    : 0);

            if (!duration || duration <= 0) return null;

            const step = chooseTickStep(duration);
            // ticks = [0, step, 2*step, ... < duration] + [duration]
            const ticks: number[] = [0];
            for (let t = step; t < duration - 1e-9; t += step) ticks.push(t);
            ticks.push(duration); // only once

            return (
              <div style={{ position: "relative", height: 26, marginTop: 6 }}>
                {/* baseline */}
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    height: 1,
                    background: "#374151",
                  }}
                />
                {/* ticks + labels */}
                {ticks.map((t, i) => {
                  const leftPct = (t / duration) * 100;
                  const isFirst = i === 0;
                  const isLast = i === ticks.length - 1;
                  const align = isFirst ? "left" : (isLast ? "right" : "center");
                  const transform =
                    align === "center" ? "translateX(-50%)" : (align === "left" ? "translateX(0)" : "translateX(-100%)");
                  return (
                    <div key={i} style={{ position: "absolute", left: `${leftPct}%`, transform }}>
                      <div style={{ width: 1, height: 8, background: "#9CA3AF" }} />
                      <div
                        className="muted"
                        style={{ fontSize: 11, marginTop: 2, whiteSpace: "nowrap" }}
                      >
                        {formatTimeLabel(t, isLast)}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })()}

          {/* Legend */}
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 12, height: 12, background: REAL_COLOR, borderRadius: 3, display: "inline-block" }} />
              <span className="muted">Real</span>
            </div>
            {Object.entries(cmap)
              .filter(([k]) => k !== "Real")
              .map(([k, color]) => (
                <div key={k} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ width: 12, height: 12, background: color, borderRadius: 3, display: "inline-block" }} />
                  <span className="muted">{k}</span>
                </div>
              ))}
          </div>
        </div>
      ) : null}

      {rows?.length ? (
        <div className="stack">
          <div className="section-title">{title}</div>
          <table className="pretty">
            <thead>
              <tr>
                <th>Method</th>
                <th style={{ width: 120 }}>Percent</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([m, p], i) => (
                <tr key={i}>
                  <td>{m}</td>
                  <td>{typeof p === "number" ? p.toFixed(1) : p}%</td>
                  <td>
                    <div className="bar">
                      <span style={{ width: `${Math.max(0, Math.min(100, Number(p))) }%` }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className="actions">
        <a className="btn small" href={r.video_url} download>Download result</a>
        <button
          className="btn small btn-ghost"
          onClick={() => navigator.clipboard?.writeText(window.location.origin + r.video_url)}
        >
          Copy link
        </button>
      </div>
    </div>
  );
}
