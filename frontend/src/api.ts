export type AnalyzeOptions = {
  face_crop: boolean;
  auto_thr: boolean;
  thr: number;
  tta: number;
  thickness: number;
  enable_filters: boolean;
  method_gate: number;
  saliency_density: number;
  duration_sec?: number; // optional
  enabled_ids_csv?: string; // optional, comma-separated model ids
};

export type ModelMeta = {
  id: string;
  name: string;
  enabled: boolean;
  schema?: { method_names?: string[]; img_size?: number };
};

export async function analyzeVideo(file: File, opts: AnalyzeOptions) {
  const form = new FormData();
  form.append("file", file);
  Object.entries(opts).forEach(([k, v]) => {
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

export async function getHealth() {
  const r = await fetch("/api/health");
  return r.json();
}

export async function listModels() {
  const r = await fetch("/api/models");
  if (!r.ok) throw new Error("Failed to load models");
  return r.json() as Promise<ModelMeta[]>;
}

export async function setModelsEnabled(enabled_ids: string[]) {
  const r = await fetch("/api/models/set-enabled", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled_ids })
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Failed to update models");
  return j as ModelMeta[];
}
