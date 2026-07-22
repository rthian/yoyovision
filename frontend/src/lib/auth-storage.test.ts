import { beforeEach, describe, expect, it } from "vitest";

import { clearStoredToken, getStoredToken, setStoredToken } from "@/lib/auth-storage";

describe("auth-storage", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("returns null when no token has been stored", () => {
    expect(getStoredToken()).toBeNull();
  });

  it("round-trips a stored token", () => {
    setStoredToken("dev-jwt-abc123");
    expect(getStoredToken()).toBe("dev-jwt-abc123");
  });

  it("clears the token so getStoredToken reverts to null", () => {
    setStoredToken("dev-jwt-abc123");
    clearStoredToken();
    expect(getStoredToken()).toBeNull();
  });
});
