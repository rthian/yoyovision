import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:3100",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    // `next start` refuses to run against an `output: "standalone"` build
    // (see next.config.mjs, needed for the Docker image); run the
    // standalone server it produces instead, same as the Docker CMD will.
    // The standalone bundle doesn't include static chunks by itself -- Next
    // expects the build step (here, or the Dockerfile) to copy them in.
    command:
      "npm run build && cp -r .next/static .next/standalone/.next/static && node .next/standalone/server.js",
    url: "http://localhost:3100",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: {
      NEXT_PUBLIC_API_BASE_URL: "http://localhost:8099",
      PORT: "3100",
    },
  },
});
