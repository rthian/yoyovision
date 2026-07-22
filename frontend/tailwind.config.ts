import type { Config } from "tailwindcss";

/**
 * Duxton-derived design tokens (see workspace design.md rule): Grab-green
 * brand primary, semantic status colors, and the shared S/M/L/XL/full
 * radius scale. Only the light-mode values are wired in for the MVP; dark
 * mode tokens are left for a later pass.
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          primary: "#00b14f",
          softest: "#d9fcde",
          soft: "#b1eaba",
          bold: "#00804a",
          boldest: "#005339",
        },
        status: {
          positive: "#00b14f",
          informative: "#136fd8",
          notice: "#f09800",
          alert: "#d42e1c",
          neutral: "#707070",
        },
        surface: {
          default: "#ffffff",
          alt: "#f5f5f5",
        },
        content: {
          default: "#1a1a1a",
          subtle: "#3d3d3d",
          dim: "#707070",
          placeholder: "#a3a3a3",
        },
        outline: {
          softest: "#e8e8e8",
          soft: "#dbdbdb",
          default: "#bfbfbf",
        },
      },
      borderRadius: {
        s: "8px",
        m: "16px",
        l: "32px",
        xl: "64px",
        full: "10240px",
      },
      fontFamily: {
        sans: [
          "Inter",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
