export type Health = {
  status: "ok" | "loading" | "error";
  methods?: string[];
  retinaface_available?: boolean;
  models?: ModelMeta[];
  threshold_mode?: "single" | "ensemble";   // khớp backend
  threshold_default?: number | null;
};

export type ModelMeta = {
  id: string;
  name: string;
  enabled: boolean;
  schema?: { method_names?: string[]; img_size?: number };
  best_thr?: number;
};

export type ArtifactEntry = [string, string];

export type ExplanationBasic = {
  method: string;
  method_share: number;
  fake_ratio: number;
  summary: string;
  artifacts: ArtifactEntry[];
};

export type AnalyzeResult = {
  verdict: string;
  verdict_level: "clean" | "suspect" | "fake";
  video_url: string;
  frames_total: number;
  fake_frames: number;
  fake_ratio: number;
  fps: number;
  duration_sec: number;
  threshold_used: number;
  thr_override_ignored: boolean;
  detector_backend_used: string;
  method_rows_total: [string, number][];
  method_rows_fake: [string, number][];
  method_rows: [string, number][];
  method_distribution: Record<string, number>;
  frame_tags: string[];
  analyzed_start_sec: number | null;
  analyzed_end_sec: number | null;
  explanation_basic: ExplanationBasic | null;
};

export type AnalyzeOptions = {
  // Advanced
  detector_backend?: "retinaface" | "mediapipe";
  bbox_scale?: number;   // 1.10 default
  thickness?: number;    // 3 default
  thr?: number | null;   // override, only effective when exactly 1 model is enabled

  // Basic
  start_sec?: number;
  end_sec?: number;

  // Models
  enabled_ids_csv?: string;
  xai_mode?: "none" | "full";
  xai_model_id?: string;
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
  return await r.json() as ModelMeta[];
}

/**
 * Bật/tắt model.
 * Backend route: POST /api/models/set-enabled
 * Body: { enabled_ids: string[] }
 */
export async function setModelsEnabled(enabled_ids: string[]): Promise<ModelMeta[]> {
  const r = await fetch(`${API_BASE}/api/models/set-enabled`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled_ids }),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json() as ModelMeta[];
}

export async function analyzeVideo(
  file: File,
  opts: AnalyzeOptions
): Promise<AnalyzeResult> {
  const fd = new FormData();
  fd.append("file", file);

  // Advanced
  fd.append("detector_backend", String(opts.detector_backend ?? "retinaface"));
  fd.append("bbox_scale", String(opts.bbox_scale ?? 1.10));
  fd.append("thickness", String(opts.thickness ?? 3));

  // thr override: chỉ gửi nếu là number; BE sẽ tự ignore nếu >= 2 model
  if (typeof opts.thr === "number" && !Number.isNaN(opts.thr)) {
    fd.append("thr", String(opts.thr));
  }

  // Basic
  if (opts.start_sec != null) fd.append("start_sec", String(opts.start_sec));
  if (opts.end_sec != null)   fd.append("end_sec", String(opts.end_sec));

  // Models
  if (opts.enabled_ids_csv) fd.append("enabled_ids_csv", opts.enabled_ids_csv);
  if (opts.xai_mode)        fd.append("xai_mode", opts.xai_mode);
  if (opts.xai_model_id)    fd.append("xai_model_id", opts.xai_model_id);

  const r = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json() as AnalyzeResult;
}