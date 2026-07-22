/**
 * Minimal client-side access-token storage for the MVP dev-only JWT auth
 * flow (`yoyovision_api.routers.auth`). Deliberately NOT cookies/SSR-aware:
 * this mirrors the backend's "dev-only, not production-grade SSO" scope.
 */

const STORAGE_KEY = "yoyovision.access_token";

export function getStoredToken(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return window.localStorage.getItem(STORAGE_KEY);
}

export function setStoredToken(token: string): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(STORAGE_KEY, token);
}

export function clearStoredToken(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.removeItem(STORAGE_KEY);
}
