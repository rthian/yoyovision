import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { EventTimeline } from "@/components/EventTimeline";
import type { AnalysisEvent } from "@/lib/types";

function makeEvent(overrides: Partial<AnalysisEvent> = {}): AnalysisEvent {
  return {
    id: "event-1",
    analysis_id: "analysis-1",
    label: "mount",
    family: "mount",
    start_ms: 1000,
    end_ms: 2000,
    confidence: 0.9,
    outcome: "success",
    difficulty_band: "basic",
    source: "model",
    review_status: "pending",
    model_name: "mock-detector",
    model_version: "0.1.0-mock",
    evidence_json: {},
    created_at: "2026-01-15T10:00:00.000Z",
    updated_at: "2026-01-15T10:00:00.000Z",
    ...overrides,
  };
}

describe("EventTimeline", () => {
  it("renders a slider with the given duration bounds", () => {
    render(<EventTimeline events={[]} durationMs={10_000} currentMs={0} onSeek={vi.fn()} />);

    const slider = screen.getByRole("slider", { name: "Event timeline" });
    expect(slider).toHaveAttribute("aria-valuemax", "10000");
  });

  it("renders one segment per event", () => {
    const events = [makeEvent({ id: "a" }), makeEvent({ id: "b", outcome: "miss" })];
    render(<EventTimeline events={events} durationMs={10_000} currentMs={0} onSeek={vi.fn()} />);

    expect(screen.getAllByTitle(/mount/)).toHaveLength(2);
  });

  it("calls onSeek with a timestamp proportional to click position", () => {
    const onSeek = vi.fn();
    render(<EventTimeline events={[]} durationMs={10_000} currentMs={0} onSeek={onSeek} />);

    const slider = screen.getByRole("slider", { name: "Event timeline" });
    vi.spyOn(slider, "getBoundingClientRect").mockReturnValue({
      left: 0,
      right: 200,
      width: 200,
      top: 0,
      bottom: 0,
      height: 0,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });

    fireEvent.click(slider, { clientX: 100, clientY: 0 });

    expect(onSeek).toHaveBeenCalledWith(5000);
  });
});
