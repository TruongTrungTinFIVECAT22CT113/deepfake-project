import React from "react";

const APP_VERSION = "1.0.0";

export default function Footer(): JSX.Element {
  return (
    <footer className="app-footer">
      <div className="app-footer-inner">
        {/* Hàng 1: Phiên bản + Liên hệ */}
        <div className="footer-row">
          <div className="footer-section">
            <div className="footer-label">Phiên bản</div>
            <div>v{APP_VERSION}</div>
          </div>
          <div className="footer-section">
            <div className="footer-label">Liên hệ / Hỗ trợ</div>
            <div className="footer-links">
              <a href="mailto:support@deepfakedetect.local">Email hỗ trợ</a>
              <span className="footer-sep">·</span>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub</a>
              <span className="footer-sep">·</span>
              <a href="https://discord.gg" target="_blank" rel="noopener noreferrer">Discord</a>
            </div>
          </div>
        </div>

        {/* Hàng 2: Công nghệ */}
        <div className="footer-section">
          <div className="footer-label">Công nghệ</div>
          <div>Powered by PyTorch</div>
        </div>

        {/* Disclaimer */}
        <div className="footer-disclaimer">
          Lưu ý: Kết quả phân tích mang tính tham khảo. AI không chính xác 100% 
          — không thay thế giám định chuyên môn.
        </div>

        {/* Bản quyền */}
        <div className="footer-copyright">
          © {new Date().getFullYear()} Deepfake Detect. Mọi quyền được bảo lưu.
        </div>
      </div>
    </footer>
  );
}