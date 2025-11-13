export type Health = {
  status: "ok" | "loading" | "error";
  methods?: string[];
  retinaface_available?: boolean;
  models?: ModelMeta[];
  threshold_mode?: "single" | "average";
  threshold_default?: number | null;
};

export type ModelMeta = {
  id: string;
  name: string;
  enabled: boolean;
  schema?: { method_names?: string[]; img_size?: number };
  best_thr?: number;
};

export type AnalyzeOptions = {
  // Advanced
  detector_backend?: "retinaface" | "mediapipe";
  bbox_scale?: number;   // 1.10 default
  thickness?: number;    // 3 default
  thr?: number | null;   // override, only effective when exactly 1 model is enabled
  // Basic
  start_sec?: number;        // NEW
  end_sec?: number;          // NEW
  // Models
  enabled_ids_csv?: string;
  xai_mode?: string;
};

const API_BASE = "";

export async function getHealth(): Promise<Health> {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    if (!r.ok) return { status: "error" };
    const j = await r.json();
    return j;
  } catch {
    return { status: "error" };
  }
}

export async function listModels(): Promise<ModelMeta[]> {
  const r = await fetch(`${API_BASE}/api/models`);
  if (!r.ok) throw new Error("Failed to list models");
  return await r.json();
}

export async function setModelsEnabled(enabled_ids: string[]): Promise<ModelMeta[]> {
  const r = await fetch(`${API_BASE}/api/models/set-enabled`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled_ids }),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function analyzeVideo(
  file: File,
  opts: AnalyzeOptions
): Promise<any> {
  const fd = new FormData();
  fd.append("file", file);

  // Advanced
  fd.append("detector_backend", String(opts.detector_backend ?? "retinaface"));
  fd.append("bbox_scale", String(opts.bbox_scale ?? 1.10));
  fd.append("thickness", String(opts.thickness ?? 3));

  // thr override: only send if number (frontend still sends; BE will ignore if >=2 models)
  if (typeof opts.thr === "number" && !Number.isNaN(opts.thr)) {
    fd.append("thr", String(opts.thr));
  }

  // Basic
  if (opts.start_sec != null) fd.append("start_sec", String(opts.start_sec));
  if (opts.end_sec != null)   fd.append("end_sec", String(opts.end_sec));

  // Models
  if (opts.enabled_ids_csv) fd.append("enabled_ids_csv", opts.enabled_ids_csv);
  if (opts.xai_mode) fd.append("xai_mode", opts.xai_mode);

  const r = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
