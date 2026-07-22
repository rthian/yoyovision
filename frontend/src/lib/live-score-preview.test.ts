import { describe, expect, it } from "vitest";

import { computeLiveScorePreview } from "@/lib/live-score-preview";
import type {
  AnalysisEvent,
  MajorDeduction,
  Ruleset,
  ScoreBreakdown,
  TechnicalLineItem,
} from "@/lib/types";

const RULESET: Ruleset = {
  version: "1a-draft-0.1",
  is_official: false,
  disclaimer: "unofficial",
  difficulty_band_points: { basic: 1, intermediate: 2, advanced: 3, unknown: 0 },
  repeated_element_decay: {},
  deduction_rules: [
    {
      type: "yoyo_stop",
      points_per_occurrence: 2,
      max_occurrences_penalized: null,
      requires_manual_confirmation: false,
    },
  ],
  freestyle_evaluation_weights: { execution: 1 },
  technical_scale_max: 100,
  freestyle_evaluation_scale_max: 100,
  technical_weight: 0.6,
  freestyle_evaluation_weight: 0.4,
};

const SCORE: ScoreBreakdown = {
  technical_raw: 2,
  technical_scaled: 2,
  freestyle_evaluation_raw: 80,
  freestyle_evaluation_scaled: 80,
  major_deductions: 2,
  final_score: 47.2,
  confidence: 0.9,
  ruleset_version: "1a-draft-0.1",
  warnings: [],
};

function event(id: string, startMs: number, endMs: number): AnalysisEvent {
  return {
    id,
    analysis_id: "analysis-1",
    label: id,
    family: "mount",
    start_ms: startMs,
    end_ms: endMs,
    confidence: 0.9,
    outcome: "success",
    difficulty_band: "basic",
    source: "model",
    review_status: "confirmed",
    model_name: "mock",
    model_version: "0",
    evidence_json: { evidence: [] },
  };
}

describe("computeLiveScorePreview", () => {
  it("credits only completed tricks at the playhead", () => {
    const events = [event("a", 0, 1000), event("b", 2000, 3000)];
    const lineItems = new Map<string, TechnicalLineItem>([
      [
        "a",
        {
          event_id: "a",
          start_ms: 0,
          label: "a",
          family: "mount",
          base_points: 1,
          multiplier: 1,
          points: 1,
          reason: "credited",
        },
      ],
      [
        "b",
        {
          event_id: "b",
          start_ms: 2000,
          label: "b",
          family: "mount",
          base_points: 1,
          multiplier: 1,
          points: 1,
          reason: "credited",
        },
      ],
    ]);

    const mid = computeLiveScorePreview(events, lineItems, [], SCORE, RULESET, 1500);
    const end = computeLiveScorePreview(events, lineItems, [], SCORE, RULESET, 3500);

    expect(mid.technical_raw).toBe(1);
    expect(mid.completed_event_count).toBe(1);
    expect(end.technical_raw).toBe(2);
    expect(end.completed_event_count).toBe(2);
  });

  it("includes deductions only after their timestamp", () => {
    const deductions: MajorDeduction[] = [
      {
        id: "d1",
        analysis_id: "analysis-1",
        type: "yoyo_stop",
        timestamp_ms: 5000,
        quantity: 1,
        points: 2,
        confidence: 0.9,
        source: "model",
        review_status: "confirmed",
      },
    ];

    const before = computeLiveScorePreview([], new Map(), deductions, SCORE, RULESET, 1000);
    const after = computeLiveScorePreview([], new Map(), deductions, SCORE, RULESET, 6000);

    expect(before.major_deductions).toBe(0);
    expect(after.major_deductions).toBe(2);
  });
});
