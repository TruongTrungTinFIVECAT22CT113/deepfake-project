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
  }
}): JSX.Element | null {
  const r = props.result;
  const framesLine = useMemo(() => (r ? extractFramesLine(r.fake_real_bar_html) : null), [r]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [rate, setRate] = useState(1.0); // NEW: playback rate

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate;
  }, [rate]);

  if (!r) return null;

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <video ref={videoRef} src={r.video_url} controls style={{ maxWidth: "100%", borderRadius: 12 }} />

      {/* NEW: slider tốc độ phát */}
      <label>
        Playback speed: <b>{rate.toFixed(2)}×</b>
        <input type="range" min="0.25" max="2" step="0.05"
               value={rate} onChange={(e)=>setRate(parseFloat(e.target.value))}/>
      </label>

      {/* Chỉ hiển thị thống kê Frames; ẩn thanh Fake/Real */}
      {framesLine && <div style={{ fontWeight: 600 }}>{framesLine}</div>}

      <div><b>{r.verdict}</b></div>

      {r.method_rows?.length > 0 && (
        <table>
          <thead><tr><th>Method</th><th>%</th></tr></thead>
          <tbody>
            {r.method_rows.map(([m, p], i) => (
              <tr key={i}><td>{m}</td><td>{p}</td></tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
