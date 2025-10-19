import React, { createContext, useContext, useMemo, useRef, useState } from "react";

type ToastType = "info" | "success" | "error";
export type ToastItem = { id: number; type: ToastType; text: string };

type ToastCtx = { addToast: (text: string, type?: ToastType, ms?: number) => void };
const Ctx = createContext<ToastCtx | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const idRef = useRef(1);

  const api = useMemo<ToastCtx>(() => ({
    addToast(text: string, type: ToastType = "info", ms = 3500) {
      const id = idRef.current++;
      setItems((lst) => [...lst, { id, type, text }]);
      window.setTimeout(() => setItems((lst) => lst.filter((t) => t.id !== id)), ms);
    },
  }), []);

  return (
    <Ctx.Provider value={api}>
      {children}
      <div className="toasts">
        {items.map((t) => (
          <div key={t.id} className={`toast ${t.type}`} role="status">{t.text}</div>
        ))}
      </div>
    </Ctx.Provider>
  );
}

export function useToast() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useToast must be used within ToastProvider");
  return c;
}

