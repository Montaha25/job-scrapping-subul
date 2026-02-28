import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: "#6c63ff",
        "brand-light": "#a78bfa",
        dark: { 900: "#0f0f1a", 800: "#1e1e2e", 700: "#2d2d44" },
      },
    },
  },
  plugins: [],
};
export default config;
