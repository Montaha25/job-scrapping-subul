import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      // ── Page d'accueil → login JobScan (port 8000) ──────────────────────────
      { source: "/",              destination: "http://localhost:8000/" },

      // ── Career Assistant API (port 8001) ────────────────────────────────────
      { source: "/api/:path*",    destination: "http://localhost:8001/api/:path*" },

      // ── JobScan API (port 8000) ──────────────────────────────────────────────
      { source: "/scan",          destination: "http://localhost:8000/scan" },
      { source: "/jobs/:path*",   destination: "http://localhost:8000/jobs/:path*" },
      { source: "/profile/:path*",destination: "http://localhost:8000/profile/:path*" },
      { source: "/cv",            destination: "http://localhost:8000/cv" },
    ];
  },
};

export default nextConfig;