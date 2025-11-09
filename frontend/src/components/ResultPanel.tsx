import React, { useRef, useState, useEffect } from "react";

type NewFields = {
  frames_total?: number;
  fake_frames?: number;
  fake_ratio?: number;    // 0..1
  threshold_used?: number;
  thr_override_ignored?: boolean;
  detector_backend_used?: string;
};

export default function ResultPanel(props: {
  result?: {
    verdict: string;
    video_url: string;
    method_rows?: [string, number][];
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
    r.frames_total !== undefined && r.fake_frames !== undefined && r.fake_ratio !== undefined
      ? `Frames: ${r.frames_total} | Fake-frames: ${r.fake_frames} (${(r.fake_ratio * 100).toFixed(1)}%)`
      : null;

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

      {r.method_rows?.length ? (
        <div className="stack">
          <div className="section-title">Method distribution (fake frames)</div>
          <table className="pretty">
            <thead><tr><th>Method</th><th style={{width:120}}>Percent</th><th></th></tr></thead>
            <tbody>
              {r.method_rows.map(([m, p], i) => (
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
