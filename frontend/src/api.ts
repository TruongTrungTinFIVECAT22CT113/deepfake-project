// ---- Types ----
export type ModelMeta = { id: string; name: string; enabled: boolean };

export type AnalyzeOptions = {
  face_crop: boolean;
  auto_thr: boolean;
  thr: number;
  tta: number;
  thickness: number;
  enable_filters: boolean;
  method_gate: number;
  saliency_density: number;
  duration_sec?: number;
  enabled_ids_csv?: string; // FE gửi list model đang bật
};

// ---- Helpers ----
const API_BASE = ""; // dùng proxy Vite: '' + '/api/...'

// ---- Calls ----
export async function getHealth() {
  const r = await fetch(`/api/health`);
  if (!r.ok) throw new Error("health failed");
  return r.json();
}

export async function listModels(): Promise<ModelMeta[]> {
  const r = await fetch(`/api/models`);
  if (!r.ok) throw new Error("models failed");
  return r.json();
}

export async function setModelsEnabled(enabled_ids: string[]): Promise<ModelMeta[]> {
  const r = await fetch(`/api/models/set-enabled`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled_ids }),
  });
  if (!r.ok) throw new Error((await r.json()).error || "set-enabled failed");
  return r.json();
}

export async function analyzeVideo(file: File, opts: AnalyzeOptions) {
  const form = new FormData();
  form.append("file", file);
  Object.entries(opts).forEach(([k, v]) => {
    if (v !== undefined && v !== null && !(typeof v === "number" && Number.isNaN(v))) {
      form.append(k, String(v));
    }
  });

  const r = await fetch(`/api/analyze`, { method: "POST", body: form });
  if (!r.ok) throw new Error((await r.json()).error || "analyze failed");

  // Kết quả backend mới
  type NewResp = {
    verdict: string;
    video_url: string;
    frames_total?: number;
    fake_frames?: number;
    fake_ratio?: number;      // 0..1
    threshold_used?: number;
    method_rows: [string, number][];
    fake_real_bar_html?: string;
  };
  return r.json() as Promise<NewResp>;
}
