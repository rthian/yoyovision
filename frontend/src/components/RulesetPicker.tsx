"use client";

import type { Ruleset } from "@/lib/types";

interface RulesetPickerProps {
  rulesets: Ruleset[];
  selectedVersion: string;
  disabled?: boolean;
  onChange: (version: string) => void;
}

/** Lets judges pick which versioned scoring config applies to this analysis. */
export function RulesetPicker({
  rulesets,
  selectedVersion,
  disabled = false,
  onChange,
}: RulesetPickerProps): JSX.Element {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="font-semibold text-content-default">Scoring ruleset</span>
      <select
        className="h-10 rounded-s border border-outline-soft bg-surface-default px-3 text-content-default disabled:cursor-not-allowed disabled:opacity-60"
        value={selectedVersion}
        disabled={disabled || rulesets.length === 0}
        onChange={(event) => onChange(event.target.value)}
      >
        {rulesets.map((ruleset) => (
          <option key={ruleset.version} value={ruleset.version}>
            {ruleset.version}
            {ruleset.is_official ? "" : " (unofficial draft)"}
          </option>
        ))}
      </select>
      <span className="text-content-dim">
        Changing the ruleset recalculates the score from current events, deductions, and freestyle
        evaluation.
      </span>
    </label>
  );
}
