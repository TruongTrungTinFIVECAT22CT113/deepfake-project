import React, { useMemo, useRef, useState, useEffect } from "react";

function extractFramesLine(html: string): string | null {
  const text = html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  const m = text.match(/Frames:\s*\d+\s*\|\s*Fake-frames:\s*\d+\s*\(\d+(?:\.\d+)?%\)/i);
  return m ? m[0] : null;
}

export default function ResultPanel(props: {
  result?: {
    verdict: string;
    fake_real_bar_html: string;
    method_rows: [string, number][];
    video_url: string;
  };
  loading?: boolean;
}): JSX.Element | null {
  const r = props.result;
  const framesLine = useMemo(() => (r ? extractFramesLine(r.fake_real_bar_html) : null), [r]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [rate, setRate] = useState(1.0);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate;
  }, [rate]);

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

  return (
    <div className="stack">
      <video ref={videoRef} src={r.video_url} className="media" controls />

      <div>
        <div className="muted" style={{marginBottom:6}}>Playback speed: <b>{rate.toFixed(2)}x</b></div>
        <input type="range" min="0.25" max="2" step="0.05"
               value={rate} onChange={(e)=>setRate(parseFloat(e.target.value))}/>
      </div>

      {r.fake_real_bar_html && (
        <div dangerouslySetInnerHTML={{__html: r.fake_real_bar_html}} />
      )}

      {framesLine && <div style={{ fontWeight: 600 }}>{framesLine}</div>}
      <div><b>{r.verdict}</b></div>

      <div className="actions">
        <a className="btn small" href={r.video_url} download>
          Download result
        </a>
        <button className="btn small btn-ghost" onClick={() => navigator.clipboard?.writeText(window.location.origin + r.video_url)}>
          Copy link
        </button>
      </div>

      {r.method_rows?.length > 0 && (
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
      )}
    </div>
  );
}

