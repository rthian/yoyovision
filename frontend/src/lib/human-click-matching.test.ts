import { describe, expect, it } from "vitest";

import {
  buildHumanClickComparison,
  humanEvidenceDisagrees,
} from "@/lib/human-click-matching";
import type {
  AnalysisEvent,
  HumanJudgingEntryReference,
  JudgeResultRow,
} from "@/lib/types";

function event(id: string, startMs: number, endMs: number, outcome: AnalysisEvent["outcome"] = "success"): AnalysisEvent {
  return {
    id,
    analysis_id: "analysis",
    label: id,
    family: "mount",
    start_ms: startMs,
    end_ms: endMs,
    confidence: 0.8,
    outcome,
    difficulty_band: "basic",
    source: "model",
    review_status: "pending",
    model_name: "test",
    model_version: "1",
    evidence_json: {},
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

function judge(overrides: Partial<JudgeResultRow> = {}): JudgeResultRow {
  return {
    assignment_id: "judge-1",
    display_name: "Alex",
    include_in_results: true,
    is_shadow: false,
    is_submitted: true,
    included_in_aggregate: true,
    scores: {
      execution: null,
      control: null,
      trick_diversity: null,
      space_use_emphasis: null,
      music_choreography: null,
      music_construction: null,
      body_control: null,
      showmanship: null,
    },
    notes: "",
    positive_clicks: 0,
    negative_clicks: 0,
    net_technical_clicks: 0,
    technical_clicks: [],
    ...overrides,
  };
}

function entry(judges: JudgeResultRow[]): HumanJudgingEntryReference {
  return {
    entry_id: "entry",
    title: "Final",
    mode: "contest",
    status: "open",
    judges,
    panel_net_technical_clicks: null,
    technical_click_range: null,
    panel_freestyle: judge().scores,
  };
}

describe("human click matching", () => {
  it("matches clicks inside or close to the nearest trick and keeps unmatched clicks", () => {
    const events = [event("first", 1_000, 2_000), event("second", 5_000, 6_000)];
    const comparison = buildHumanClickComparison(
      events,
      [
        entry([
          judge({
            technical_clicks: [
              { id: "a", timestamp_ms: 1_500, kind: "positive", created_at: "1" },
              { id: "b", timestamp_ms: 3_400, kind: "negative", created_at: "2" },
              { id: "c", timestamp_ms: 9_000, kind: "positive", created_at: "3" },
            ],
          }),
        ]),
      ]
    );

    expect(comparison.byEventId.get("first")?.positive).toBe(1);
    expect(comparison.byEventId.get("first")?.negative).toBe(1);
    expect(comparison.unmatchedClicks.map((click) => click.id)).toEqual(["c"]);
  });

  it("excludes shadow and non-counting judges", () => {
    const comparison = buildHumanClickComparison(
      [event("first", 1_000, 2_000)],
      [
        entry([
          judge({
            assignment_id: "shadow",
            is_shadow: true,
            technical_clicks: [
              { id: "shadow-click", timestamp_ms: 1_500, kind: "positive", created_at: "1" },
            ],
          }),
          judge({
            assignment_id: "excluded",
            include_in_results: false,
            technical_clicks: [
              { id: "excluded-click", timestamp_ms: 1_500, kind: "positive", created_at: "1" },
            ],
          }),
        ]),
      ]
    );

    expect(comparison.totalClicks).toBe(0);
  });

  it("flags outcome and human-net conflicts", () => {
    const success = event("success", 0, 1_000, "success");
    const miss = event("miss", 2_000, 3_000, "miss");
    expect(
      humanEvidenceDisagrees(success, {
        clicks: [
          {
            id: "negative",
            timestamp_ms: 500,
            kind: "negative",
            created_at: "1",
            judgeName: "Alex",
            entryTitle: "Final",
          },
        ],
        positive: 0,
        negative: 1,
        net: -1,
        judgeCount: 1,
      })
    ).toBe(true);
    expect(
      humanEvidenceDisagrees(miss, {
        clicks: [
          {
            id: "positive",
            timestamp_ms: 2_500,
            kind: "positive",
            created_at: "1",
            judgeName: "Alex",
            entryTitle: "Final",
          },
        ],
        positive: 1,
        negative: 0,
        net: 1,
        judgeCount: 1,
      })
    ).toBe(true);
  });
});
