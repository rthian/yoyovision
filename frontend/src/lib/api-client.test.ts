import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  cancelAnalysis,
  exportReportJson,
  getVideo,
  listVideos,
  login,
  triggerVideoAnalysis,
} from "@/lib/api-client";
import { clearStoredToken, setStoredToken } from "@/lib/auth-storage";

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

describe("api-client", () => {
  beforeEach(() => {
    clearStoredToken();
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("omits the Authorization header when no token is stored", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(jsonResponse([]));

    await listVideos();

    const [, init] = fetchMock.mock.calls[0] ?? [];
    const headers = init?.headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
  });

  it("attaches a bearer token when one is stored", async () => {
    setStoredToken("dev-jwt-abc123");
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(jsonResponse([]));

    await listVideos();

    const [, init] = fetchMock.mock.calls[0] ?? [];
    const headers = init?.headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer dev-jwt-abc123");
  });

  it("sends login credentials as a JSON body and returns the token response", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ access_token: "dev-jwt-abc123", token_type: "bearer" })
    );

    const result = await login({ email: "coach@example.com", password: "hunter2" });

    expect(result).toEqual({ access_token: "dev-jwt-abc123", token_type: "bearer" });
    const [, init] = fetchMock.mock.calls[0] ?? [];
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      email: "coach@example.com",
      password: "hunter2",
    });
  });

  it("clears the stored token on a 401 response", async () => {
    setStoredToken("dev-jwt-abc123");
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "Not authenticated" }, 401));

    await expect(getVideo("video-1")).rejects.toBeInstanceOf(ApiError);
    // A subsequent call should go out with no Authorization header.
    fetchMock.mockResolvedValueOnce(jsonResponse([]));
    await listVideos();
    const [, init] = fetchMock.mock.calls[1] ?? [];
    const headers = init?.headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
  });

  it("surfaces a plain string detail as the ApiError message", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "Video not found." }, 404));

    await expect(getVideo("missing")).rejects.toMatchObject({
      status: 404,
      message: "Video not found.",
    });
  });

  it("surfaces a structured {code, message} detail on the ApiError", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ detail: { code: "invalid_mime_type", message: "Unsupported file type." } }, 400)
    );

    await expect(getVideo("bad")).rejects.toMatchObject({
      status: 400,
      code: "invalid_mime_type",
      message: "Unsupported file type.",
    });
  });

  it("falls back to a generic message when the error body has no JSON", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(new Response("internal error", { status: 500 }));

    await expect(getVideo("boom")).rejects.toMatchObject({
      status: 500,
      message: "Request failed with status 500.",
    });
  });

  it("parses the filename out of Content-Disposition for blob exports", async () => {
    const fetchMock = vi.mocked(fetch);
    const blob = new Blob(["{}"], { type: "application/json" });
    fetchMock.mockResolvedValueOnce(
      new Response(blob, {
        status: 200,
        headers: { "Content-Disposition": 'attachment; filename="report.json"' },
      })
    );

    const result = await exportReportJson("analysis-1");

    expect(result.filename).toBe("report.json");
    expect(result.blob.size).toBeGreaterThan(0);
  });

  it("posts to the cancel endpoint for cancelAnalysis", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ id: "analysis-1", cancel_requested: true, status: "running" })
    );

    const result = await cancelAnalysis("analysis-1");

    expect(result.cancel_requested).toBe(true);
    const [url, init] = fetchMock.mock.calls[0] ?? [];
    expect(String(url)).toContain("/analyses/analysis-1/cancel");
    expect(init?.method).toBe("POST");
  });

  it("passes shadow=true as a query param when triggering a shadow analysis", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: "analysis-2", is_shadow: true }));

    const result = await triggerVideoAnalysis("video-1", { shadow: true });

    expect(result.is_shadow).toBe(true);
    const [url] = fetchMock.mock.calls[0] ?? [];
    expect(String(url)).toContain("shadow=true");
  });
});
