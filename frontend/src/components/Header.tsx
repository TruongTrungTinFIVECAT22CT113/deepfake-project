import React from "react";

export default function Header({
  apiStatus,
}: {
  apiStatus: string;
}): JSX.Element {
  return (
    <header className="app-header">
      <div className="app-header-inner">
        <div className="title">Phát hiện Deepfake</div>
        <div className="toolbar">
          {apiStatus === "ok" ? (
            <span className="pill success">Đã kết nối</span>
          ) : apiStatus === "loading" ? (
            <span className="pill">Đang kết nối…</span>
          ) : (
            <span className="pill danger">Mất kết nối</span>
          )}
        </div>
      </div>
    </header>
  );
}