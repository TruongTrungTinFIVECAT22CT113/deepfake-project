import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8081",  // sửa theo cổng backend của bạn
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
