import React, { useEffect, useState } from "react";
import AnalyzerForm from "./components/AnalyzerForm";
import ResultPanel from "./components/ResultPanel";
import ModelSelectorCard from "./components/ModelSelectorCard";
import { getHealth, listModels, setModelsEnabled, type ModelMeta } from "./api";
import { ToastProvider } from "./components/Toast";

export default function App(): JSX.Element {
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<any>(null);

  // Theme
  const [theme, setTheme] = useState<string>(() =>
    (typeof localStorage !== "undefined" && localStorage.getItem("theme")) || "dark"
  );
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("theme", theme); } catch {}
  }, [theme]);

  // Backend models / health
  const [models, setModels] = useState<ModelMeta[]>([]);
  const enabledIds = models.filter(m => m.enabled).map(m => m.id);
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
      } catch {
        setApiStatus("error");
      }
    })();
    return () => { mounted = false; };
  }, []);

  async function toggleModel(id: string, next: boolean) {
    const nextIds = next
      ? [...new Set([...enabledIds, id])]
      : enabledIds.filter(x => x !== id);
    if (nextIds.length < 1) return; // giữ ≥1 model bật
    try {
      const updated = await setModelsEnabled(nextIds);
      setModels(updated);
    } catch {}
  }

  return (
    <ToastProvider>
      <header className="app-header">
        <div className="app-header-inner">
          <div className="title">Deepfake Detect</div>
          <div className="toolbar">
            <span className="muted" style={{fontSize:12}}>Dark-first UI</span>
            <div className="segmented" role="group" aria-label="Theme">
              <button aria-pressed={theme === "dark"} onClick={() => setTheme("dark")}>Dark</button>
              <button aria-pressed={theme === "light"} onClick={() => setTheme("light")}>Light</button>
            </div>
          </div>
        </div>
      </header>

      <main className="container stack-lg">
        {/* Hàng trên: Models (trái) + Analyze (phải) */}
        <div className="layout-top">
          <section className="card">
            <ModelSelectorCard models={models} onToggle={toggleModel} />
          </section>

          <section className="card stack">
            <div className="section-title">ANALYZE VIDEO</div>
            <div className="muted" style={{fontSize:12}}>
              API:&nbsp;
              {apiStatus === "ok" ? <span className="pill success">OK</span> :
               apiStatus === "loading" ? <span className="pill">Loading</span> :
               <span className="pill danger">Error</span>}
            </div>

            <AnalyzerForm onResult={setRes} setLoading={setLoading} enabledIds={enabledIds} />

            {loading && (
              <div className="loading-row">
                <div className="spinner" />
                <span>Processing...</span>
              </div>
            )}
          </section>

          {/* Hàng dưới: Result full width */}
          <section className="card full-span">
            <ResultPanel result={res} loading={loading} />
          </section>
        </div>
      </main>
    </ToastProvider>
  );
}
