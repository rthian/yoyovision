import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AnalysisJobList } from "@/components/AnalysisJobList";
import type { AnalysisJob } from "@/lib/types";

function makeJob(overrides: Partial<AnalysisJob> = {}): AnalysisJob {
  return {
    id: "job-1",
    video_id: "video-1",
    status: "pending",
    progress: 0,
    current_stage: "queued",
    error_code: null,
    error_message: null,
    pipeline_version: "0.1.0-dev",
    created_at: "2026-07-21T12:00:00.000Z",
    started_at: null,
    completed_at: null,
    model_versions: null,
    device: null,
    runtime_versions: null,
    stage_durations_ms: null,
    is_shadow: false,
    cancel_requested: false,
    retry_count: 0,
    routine_start_ms: null,
    routine_end_ms: null,
    review_state: "draft",
    submitted_at: null,
    ruleset_version: "1a-draft-0.1",
    ...overrides,
  };
}

describe("AnalysisJobList", () => {
  it("shows a Shadow badge for shadow-mode jobs", () => {
    render(<AnalysisJobList jobs={[makeJob({ is_shadow: true })]} />);

    expect(screen.getByText("Shadow")).toBeInTheDocument();
  });

  it("shows a Cancel button for pending/running jobs when onCancel is provided", () => {
    const onCancel = vi.fn();
    render(
      <AnalysisJobList
        jobs={[makeJob({ status: "running", current_stage: "pose_extraction" })]}
        onCancel={onCancel}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledWith("job-1");
  });

  it("hides Cancel once cancel_requested is set and shows Cancelling...", () => {
    render(
      <AnalysisJobList
        jobs={[makeJob({ status: "running", cancel_requested: true })]}
        onCancel={vi.fn()}
      />
    );

    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
    expect(screen.getByText("Cancelling...")).toBeInTheDocument();
  });

  it("shows Review for completed jobs and no Cancel button", () => {
    render(
      <AnalysisJobList
        jobs={[makeJob({ status: "completed", current_stage: "done" })]}
        onCancel={vi.fn()}
      />
    );

    expect(screen.getByRole("link", { name: "Review" })).toHaveAttribute("href", "/analyses/job-1");
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
  });

  it("shows Delete for completed jobs when onDelete is provided", () => {
    const onDelete = vi.fn();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(
      <AnalysisJobList
        jobs={[makeJob({ status: "completed", current_stage: "done" })]}
        onDelete={onDelete}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(onDelete).toHaveBeenCalledWith("job-1");
  });
});
