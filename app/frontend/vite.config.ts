/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 127.0.0.1, not localhost: Node may resolve localhost to IPv6 while uvicorn listens on IPv4.
const API_TARGET = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: API_TARGET, ws: true } },   // ws: the live monitor uses WebSockets
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    coverage: { include: ["src/**/*.{ts,tsx}"], exclude: ["src/main.tsx", "src/test/**"] },
  },
});
