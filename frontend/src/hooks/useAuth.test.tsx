import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AuthProvider, useAuth } from "@/hooks/useAuth";
import * as apiClient from "@/lib/api-client";
import { getStoredToken } from "@/lib/auth-storage";

describe("useAuth", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("resolves isAuthenticated=false once initialization completes with no stored token", async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });

    await waitFor(() => expect(result.current.isInitializing).toBe(false));
    expect(result.current.isAuthenticated).toBe(false);
  });

  it("becomes authenticated and persists the token after a successful login", async () => {
    vi.spyOn(apiClient, "login").mockResolvedValue({
      access_token: "dev-jwt-abc123",
      token_type: "bearer",
    });

    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await waitFor(() => expect(result.current.isInitializing).toBe(false));

    await act(async () => {
      await result.current.login("coach@example.com", "hunter2");
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(getStoredToken()).toBe("dev-jwt-abc123");
  });

  it("clears the token and flips isAuthenticated on logout", async () => {
    vi.spyOn(apiClient, "login").mockResolvedValue({
      access_token: "dev-jwt-abc123",
      token_type: "bearer",
    });

    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await waitFor(() => expect(result.current.isInitializing).toBe(false));

    await act(async () => {
      await result.current.login("coach@example.com", "hunter2");
    });
    act(() => {
      result.current.logout();
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(getStoredToken()).toBeNull();
  });

  it("throws when used outside an AuthProvider", () => {
    expect(() => renderHook(() => useAuth())).toThrow(
      "useAuth must be used within an AuthProvider."
    );
  });
});
