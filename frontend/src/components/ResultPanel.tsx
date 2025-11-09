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
};

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
        const realFrames = r.frames_total - r.fake_frames;
        const realRatio = 1 - r.fake_ratio;
        return (
          <span>
            <b>Total-frames:</b> {r.frames_total}{" "}
            | <b style={{ color: "#df4040" }}>Fake-frames:</b> {r.fake_frames}{" "}
            (<span style={{ color: "#df4040" }}>
              {(r.fake_ratio * 100).toFixed(1)}%
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

  return (
    <div className="stack">
      <video ref={videoRef} src={r.video_url} className="media" controls />

      <div>
        <div className="muted" style={{marginBottom:6}}>Playback speed: <b>{rate.toFixed(2)}x</b></div>
        <input type="range" min="0.25" max="2" step="0.05"
               value={rate} onChange={(e)=>setRate(parseFloat(e.target.value))}/>
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

      {rows?.length ? (
        <div className="stack">
          <div className="section-title">{title}</div>
          <table className="pretty">
            <thead><tr><th>Method</th><th style={{width:120}}>Percent</th><th></th></tr></thead>
            <tbody>
              {rows.map(([m, p], i) => (
                <tr key={i}>
                  <td>{m}</td>
                  <td>{typeof p === 'number' ? p.toFixed(1) : p}%</td>
                  <td><div className="bar"><span style={{width: `${Math.max(0, Math.min(100, Number(p))) }%`}} /></div></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className="actions">
        <a className="btn small" href={r.video_url} download>Download result</a>
        <button className="btn small btn-ghost" onClick={() => navigator.clipboard?.writeText(window.location.origin + r.video_url)}>
          Copy link
        </button>
      </div>
    </div>
  );
}
