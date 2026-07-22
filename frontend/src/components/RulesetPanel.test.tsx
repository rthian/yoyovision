import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RulesetPanel } from "@/components/RulesetPanel";
import type { Ruleset } from "@/lib/types";

const RULESET: Ruleset = {
  version: "2026.1.0-mock",
  is_official: false,
  disclaimer:
    "This is an unofficial, AI-assisted estimate. It is not certified by IYYF or any competition body.",
  difficulty_band_points: { basic: 1, intermediate: 2, advanced: 3, unknown: 0 },
  repeated_element_decay: { factor: 0.5 },
  deduction_rules: [
    {
      type: "yoyo_stop",
      points_per_occurrence: 1,
      max_occurrences_penalized: null,
      requires_manual_confirmation: false,
    },
    {
      type: "yoyo_detach",
      points_per_occurrence: 2,
      max_occurrences_penalized: 3,
      requires_manual_confirmation: false,
    },
    {
      type: "dangerous_play_review",
      points_per_occurrence: 5,
      max_occurrences_penalized: null,
      requires_manual_confirmation: true,
    },
  ],
  freestyle_evaluation_weights: { execution: 1 },
  technical_scale_max: 60,
  freestyle_evaluation_scale_max: 40,
};

describe("RulesetPanel", () => {
  it("renders nothing when no ruleset is provided", () => {
    const { container } = render(<RulesetPanel ruleset={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the version, disclaimer and deduction rules", () => {
    render(<RulesetPanel ruleset={RULESET} />);

    expect(screen.getByText(/Ruleset 2026.1.0-mock/)).toBeInTheDocument();
    expect(screen.getByText(/unofficial draft/)).toBeInTheDocument();
    expect(screen.getByText(RULESET.disclaimer)).toBeInTheDocument();
    expect(screen.getByText(/yoyo_detach: 2 pts each \(capped at 3 occurrences\)/)).toBeInTheDocument();
    expect(
      screen.getByText(/dangerous_play_review: 5 pts each -- requires manual confirmation/)
    ).toBeInTheDocument();
  });

  it("omits the unofficial-draft label for an official ruleset", () => {
    render(<RulesetPanel ruleset={{ ...RULESET, is_official: true }} />);
    expect(screen.queryByText(/unofficial draft/)).not.toBeInTheDocument();
  });
});
