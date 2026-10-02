import { describe, expect, it } from "vitest";

import { eventInRoutine } from "@/lib/routine-window";

describe("eventInRoutine", () => {
  const routine = { startMs: 10_000, endMs: 70_000 };

  it("includes events fully contained by the routine boundaries", () => {
    expect(eventInRoutine({ start_ms: 10_000, end_ms: 11_000 }, routine)).toBe(true);
    expect(eventInRoutine({ start_ms: 69_000, end_ms: 70_000 }, routine)).toBe(true);
  });

  it("excludes events before, after, or crossing a routine boundary", () => {
    expect(eventInRoutine({ start_ms: 9_000, end_ms: 10_500 }, routine)).toBe(false);
    expect(eventInRoutine({ start_ms: 69_500, end_ms: 70_500 }, routine)).toBe(false);
    expect(eventInRoutine({ start_ms: 71_000, end_ms: 72_000 }, routine)).toBe(false);
  });
});
