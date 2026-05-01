import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    hmr: {
      host: 'localhost',
      port: 5173,
      protocol: 'ws',
    },
  },
});
