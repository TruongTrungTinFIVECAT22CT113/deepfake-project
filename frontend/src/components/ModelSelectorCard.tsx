import React from "react";
import type { ModelMeta } from "../api";

export default function ModelSelectorCard({
  models,
  onToggle,
}: {
  models: ModelMeta[];
  onToggle: (id: string, next: boolean) => void;
}) {
  const enabledCount = models.filter(m => m.enabled).length;

  return (
    <div className="stack">
      <div className="section-title">MODELS</div>
      <div className="model-list">
        {models.map(m => {
          const onlyOneLeft = enabledCount === 1 && m.enabled;
          return (
            <label key={m.id} className="model-row">
              <input
                type="checkbox"
                checked={m.enabled}
                disabled={onlyOneLeft}
                onChange={(e) => onToggle(m.id, e.target.checked)}
              />
              <span>{m.name}</span>
              {onlyOneLeft && <em className="muted" style={{marginLeft:6}}>(required)</em>}
            </label>
          );
        })}
      </div>
      <div className="help">Giữ bật ≥ 1 model.</div>
    </div>
  );
}
