export type AnalyzeOptions = {
  face_crop: boolean;
  auto_thr: boolean;
  thr: number;
  tta: number;
  thickness: number;
  enable_filters: boolean;
  method_gate: number;
  saliency_density: number;
  duration_sec?: number; // NEW: tùy chọn
};

export async function analyzeVideo(file: File, opts: AnalyzeOptions) {
  const form = new FormData();
  form.append("file", file);
  Object.entries(opts).forEach(([k, v]) => {
    // chỉ gửi những field có giá trị
    if (v !== undefined && v !== null && !(typeof v === "number" && Number.isNaN(v))) {
      form.append(k, String(v));
    }
  });

  const res = await fetch("/api/analyze", { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).error || "Analyze failed");
  return res.json() as Promise<{
    verdict: string;
    fake_real_bar_html: string;
    method_rows: [string, number][];
    video_url: string;
  }>;
}
