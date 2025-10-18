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
    <fieldset style={{border:"1px solid #ccc", padding:12, borderRadius:8}}>
      <legend>Models</legend>
      {models.map(m=>{
        const onlyOneLeft = enabledCount === 1 && m.enabled;
        return (
          <label key={m.id} style={{display:"block", marginBottom:6}}>
            <input
              type="checkbox"
              checked={m.enabled}
              disabled={onlyOneLeft}  // Không cho tắt khi chỉ còn 1 bật
              onChange={e=>onToggle(m.id, e.target.checked)}
            />{" "}
            {m.name} {onlyOneLeft && <em>(required)</em>}
          </label>
        );
      })}
    </fieldset>
  );
}
