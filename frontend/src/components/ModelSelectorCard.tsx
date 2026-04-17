import React from "react";
import type { ModelMeta } from "../api";

export default function ModelSelectorCard({
  models,
  onToggle,
}: {
  models: ModelMeta[];
  onToggle: (id: string, next: boolean) => void;
}) {
  const enabledCount = models.filter((m) => m.enabled).length;

  return (
    <div className="stack">
      <div className="section-title">Danh Sách Các Mô Hình</div>
      <div className="help" style={{ marginTop: -8 }}>
        Tùy chọn các mô hình bằng Checkbox.
      </div>
      <div className="model-list">
        {models.map((m) => {
          const onlyOneLeft = enabledCount === 1 && m.enabled;
          return (
            <label key={m.id} className="model-row">
              <input
                type="checkbox"
                checked={m.enabled}
                disabled={onlyOneLeft}
                onChange={(e) => onToggle(m.id, e.target.checked)}
              />
              <span style={{ flex: 1 }}>{m.name}</span>
              {onlyOneLeft && (
                <em className="muted" style={{ fontSize: 11 }}>
                  bắt buộc
                </em>
              )}
            </label>
          );
        })}
      </div>
      {models.length === 0 && (
        <div className="muted" style={{ fontSize: 12, textAlign: "center", padding: "12px 0" }}>
          Không có mô hình nào
        </div>
      )}
      <div className="help">
        Cần ít nhất 1 mô hình được bật.
      </div>
    </div>
  );
}