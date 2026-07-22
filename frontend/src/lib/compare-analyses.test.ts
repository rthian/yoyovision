import { describe, expect, it } from "vitest";

import {
  compareEventCounts,
  compareScoreBreakdowns,
  pickDefaultComparisonJobs,
} from "@/lib/compare-analyses";
import type { AnalysisEvent, AnalysisJob, ScoreBreakdown } from "@/lib/types";

const SCORE: ScoreBreakdown = {
  technical_raw: 40,
  technical_scaled: 40,
  freestyle_evaluation_raw: 0,
  freestyle_evaluation_scaled: 0,
  major_deductions: 2,
  final_score: 22,
  confidence: 0.8,
  ruleset_version: "1a-draft-0.1",
  warnings: [],
};

const EVENT: AnalysisEvent = {
  id: "e1",
  analysis_id: "a1",
  label: "Trapeze",
  family: "mount",
  start_ms: 1000,
  end_ms: 2000,
  confidence: 0.9,
  outcome: "success",
  difficulty_band: "basic",
  source: "model",
  review_status: "pending",
  model_name: "mock",
  model_version: "1",
  evidence_json: {},
  created_at: "2026-07-22T00:00:00.000Z",
  updated_at: "2026-07-22T00:00:00.000Z",
};

describe("compare-analyses", () => {
  it("formats score deltas", () => {
    const rows = compareScoreBreakdowns(SCORE, { ...SCORE, final_score: 25 });
    const finalRow = rows.find((row) => row.label === "Final score");
    expect(finalRow?.delta).toBe("+3.000");
  });

  it("compares event counts", () => {
    const row = compareEventCounts([EVENT], [EVENT, EVENT]);
    expect(row.delta).toBe("+1");
  });

  it("picks latest official and shadow completed jobs", () => {
    const jobs: AnalysisJob[] = [
      {
        id: "shadow",
        video_id: "v1",
        status: "completed",
        progress: 1,
        current_stage: "done",
        error_code: null,
        error_message: null,
        pipeline_version: "0.1.0-dev",
        created_at: "2026-07-22T12:00:00.000Z",
        started_at: null,
        completed_at: null,
        model_versions: null,
        device: null,
        runtime_versions: null,
        stage_durations_ms: null,
        is_shadow: true,
        cancel_requested: false,
        retry_count: 0,
        routine_start_ms: null,
        routine_end_ms: null,
        review_state: "draft",
        submitted_at: null,
        ruleset_version: "1a-draft-0.1",
      },
      {
        id: "official",
        video_id: "v1",
        status: "completed",
        progress: 1,
        current_stage: "done",
        error_code: null,
        error_message: null,
        pipeline_version: "0.1.0-dev",
        created_at: "2026-07-22T11:00:00.000Z",
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
      },
    ];

    const picked = pickDefaultComparisonJobs(jobs);
    expect(picked.baseline?.id).toBe("official");
    expect(picked.candidate?.id).toBe("shadow");
  });
});
