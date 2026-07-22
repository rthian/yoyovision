"use client";

import type { Ruleset } from "@/lib/types";

interface RulesetPanelProps {
  ruleset: Ruleset | null;
}

/** Full transparency panel for the ruleset that produced the current score.
 * Core Product Principle #8 ("Keep the rule set versioned and configurable")
 * plus the disclaimer requirements: nothing about scoring should be opaque,
 * and the ruleset's own disclaimer text is shown verbatim. */
export function RulesetPanel({ ruleset }: RulesetPanelProps): JSX.Element | null {
  if (!ruleset) {
    return null;
  }

  return (
    <details className="rounded-m border border-outline-soft bg-surface-default p-4">
      <summary className="cursor-pointer text-base font-semibold text-content-default">
        Ruleset {ruleset.version} {ruleset.is_official ? "" : "(unofficial draft)"}
      </summary>
      <div className="mt-3 flex flex-col gap-3 text-sm">
        <p className="text-content-subtle">{ruleset.disclaimer}</p>
        <div>
          <h4 className="font-semibold text-content-default">Difficulty band points</h4>
          <ul className="text-content-dim">
            {Object.entries(ruleset.difficulty_band_points).map(([band, points]) => (
              <li key={band}>
                {band}: {points}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="font-semibold text-content-default">Deduction rules</h4>
          <ul className="text-content-dim">
            {ruleset.deduction_rules.map((rule) => (
              <li key={rule.type}>
                {rule.type}: {rule.points_per_occurrence} pts each
                {rule.max_occurrences_penalized != null
                  ? ` (capped at ${rule.max_occurrences_penalized} occurrences)`
                  : ""}
                {rule.requires_manual_confirmation ? " -- requires manual confirmation" : ""}
              </li>
            ))}
          </ul>
        </div>
        <div className="text-content-dim">
          Technical scale max: {ruleset.technical_scale_max} - Freestyle evaluation scale max:{" "}
          {ruleset.freestyle_evaluation_scale_max}
        </div>
      </div>
    </details>
  );
}
