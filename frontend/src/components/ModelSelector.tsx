import React from "react";
import type { ModelMeta } from "../api";

export default function ModelSelector({
  models, onToggle
}:{
  models: ModelMeta[];
  onToggle: (id: string, next: boolean) => void;
}) {
  const enabledCount = models.filter(m=>m.enabled).length;
  return (
    <div className="card stack">
      <div className="section-title">Models</div>
      <div className="chip-wrap">
        {models.map(m=>{
          const onlyOneLeft = enabledCount === 1 && m.enabled;
          return (
            <button
              key={m.id}
              type="button"
              className={`chip-toggle ${m.enabled ? 'on' : 'off'}`}
              aria-pressed={m.enabled}
              disabled={onlyOneLeft && m.enabled}
              onClick={()=> onToggle(m.id, !m.enabled)}
            >{m.name}</button>
          );
        })}
      </div>
      {enabledCount === 1 && (
        <div className="help">At least one model must remain enabled.</div>
      )}
    </div>
  );
}
