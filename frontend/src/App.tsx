import React, { useEffect, useState } from "react";
import Header from "./components/Header";
import Footer from "./components/Footer";
import AnalyzerForm from "./components/AnalyzerForm";
import ResultPanel from "./components/ResultPanel";
import ModelSelectorCard from "./components/ModelSelectorCard";
import { getHealth, listModels, setModelsEnabled, type ModelMeta } from "./api";
import { ToastProvider } from "./components/Toast";

export type ThemeId = "dark" | "light" | "balanced" | "colorblind";

export default function App(): JSX.Element {
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<any>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewDuration, setPreviewDuration] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Theme — default is "balanced"
  const [theme, setTheme] = useState<ThemeId>(() =>
    ((typeof localStorage !== "undefined" && localStorage.getItem("theme")) || "balanced") as ThemeId
  );
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("theme", theme); } catch {}
  }, [theme]);

  // Backend
  const [models, setModels] = useState<ModelMeta[]>([]);
  const enabledIds = models.filter((m) => m.enabled).map((m) => m.id);
  const [apiStatus, setApiStatus] = useState<string>("loading");

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const h = await getHealth();
        if (!mounted) return;
        setApiStatus(h.status || "ok");
        if (Array.isArray(h.models)) setModels(h.models);
        else setModels(await listModels());
      } catch { setApiStatus("error"); }
    })();
    return () => { mounted = false; };
  }, []);

  async function toggleModel(id: string, next: boolean) {
    const nextIds = next
      ? [...new Set([...enabledIds, id])]
      : enabledIds.filter((x) => x !== id);
    if (nextIds.length < 1) return;
    try { const updated = await setModelsEnabled(nextIds); setModels(updated); } catch {}
  }

  return (
    <ToastProvider>
      <Header apiStatus={apiStatus} />

      <main className="container stack-lg">
        <div className="layout-top">
          <section className="card">
            <ModelSelectorCard models={models} onToggle={toggleModel} />
          </section>

          <section className="card stack">
            <div className="section-title">Phân tích video</div>
            <AnalyzerForm
              onResult={(r) => { setErrorMsg(null); setRes(r); }}
              setLoading={setLoading}
              enabledIds={enabledIds}
              models={models}
              theme={theme}
              setTheme={setTheme}
              onPreviewUrl={setPreviewUrl}
              onPreviewDuration={setPreviewDuration}
              onError={setErrorMsg}
            />
            {loading && (
              <div className="loading-row">
                <div className="spinner" />
                <span>Đang phân tích video của bạn…</span>
              </div>
            )}
          </section>

          <section className="card full-span">
            <ResultPanel result={res} loading={loading} previewUrl={previewUrl} previewDuration={previewDuration} errorMsg={errorMsg} />
          </section>
        </div>
      </main>

      <Footer />
    </ToastProvider>
  );
}