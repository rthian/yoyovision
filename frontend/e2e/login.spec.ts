import { expect, test } from "@playwright/test";

/**
 * Smoke test for the auth gate + login flow. Runs against the built app with
 * no API backend reachable (see `playwright.config.ts`'s
 * `NEXT_PUBLIC_API_BASE_URL`), so it only exercises client-side behavior that
 * doesn't depend on a live FastAPI instance: the unauthenticated redirect,
 * the login form rendering, and the generic error path when the login
 * request itself fails (e.g. the API being unreachable).
 */
test.describe("login", () => {
  test("redirects an unauthenticated visitor from / to /login", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveURL(/\/login$/);
    await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();
  });

  test("renders the dev-only login form with email and password fields", async ({ page }) => {
    await page.goto("/login");

    await expect(page.getByLabel("Email")).toHaveValue("dev@yoyovision.local");
    await expect(page.getByLabel("Password")).toBeVisible();
    await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
  });

  test("shows a login error and stays on the page when the API is unreachable", async ({
    page,
  }) => {
    await page.goto("/login");

    await page.getByLabel("Password").fill("does-not-matter");
    await page.getByRole("button", { name: "Sign in" }).click();

    await expect(page.getByText("Login failed. Please try again.")).toBeVisible();
    await expect(page).toHaveURL(/\/login$/);
  });

  test("shows the scoring disclaimer banner on every page", async ({ page }) => {
    await page.goto("/login");
    await expect(page.getByText(/never certified by IYYF/i)).toBeVisible();
  });
});
