import { describe, expect, it } from "vitest";

import {
  formatBytes,
  formatConfidence,
  formatDateTime,
  formatMsAsTimecode,
  titleCaseFromSnakeCase,
} from "@/lib/format";

describe("formatMsAsTimecode", () => {
  it("formats zero as 00:00.000", () => {
    expect(formatMsAsTimecode(0)).toBe("00:00.000");
  });

  it("pads minutes, seconds and millis", () => {
    expect(formatMsAsTimecode(65_001)).toBe("01:05.001");
  });

  it("falls back to 00:00.000 for negative or non-finite input", () => {
    expect(formatMsAsTimecode(-5)).toBe("00:00.000");
    expect(formatMsAsTimecode(Number.NaN)).toBe("00:00.000");
    expect(formatMsAsTimecode(Number.POSITIVE_INFINITY)).toBe("00:00.000");
  });
});

describe("formatBytes", () => {
  it("renders sub-1024 byte counts verbatim", () => {
    expect(formatBytes(512)).toBe("512 B");
  });

  it("scales up through KB/MB/GB", () => {
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(formatBytes(3 * 1024 * 1024 * 1024)).toBe("3.0 GB");
  });

  it("caps at GB instead of overflowing to a further unit", () => {
    expect(formatBytes(5 * 1024 * 1024 * 1024 * 1024)).toBe("5120.0 GB");
  });
});

describe("formatConfidence", () => {
  it("renders a 0-1 fraction as a rounded percentage", () => {
    expect(formatConfidence(0.874)).toBe("87%");
    expect(formatConfidence(1)).toBe("100%");
    expect(formatConfidence(0)).toBe("0%");
  });
});

describe("formatDateTime", () => {
  it("returns a locale string for a valid ISO timestamp", () => {
    const result = formatDateTime("2026-01-15T10:30:00.000Z");
    expect(result).not.toBe("2026-01-15T10:30:00.000Z");
    expect(result.length).toBeGreaterThan(0);
  });

  it("falls back to the raw input for invalid dates", () => {
    expect(formatDateTime("not-a-date")).toBe("not-a-date");
  });
});

describe("titleCaseFromSnakeCase", () => {
  it("title-cases each underscore-separated word", () => {
    expect(titleCaseFromSnakeCase("yoyo_stop")).toBe("Yoyo Stop");
    expect(titleCaseFromSnakeCase("unknown_technical_element")).toBe(
      "Unknown Technical Element"
    );
  });

  it("handles a single word", () => {
    expect(titleCaseFromSnakeCase("mount")).toBe("Mount");
  });

  it("handles an empty string", () => {
    expect(titleCaseFromSnakeCase("")).toBe("");
  });
});
