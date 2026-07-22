"use client";

import type { ScoreBreakdown } from "@/lib/types";

import { useRecomputeScore } from "@/hooks/useAnalysis";

interface ScoreBreakdownPanelProps {
  analysisId: string;
  score: ScoreBreakdown | null;
}

/** Renders the deterministic `ScoreBreakdown` -- always with its
 * `ruleset_version` and any `warnings`, so scoring is never opaque (Core
 * Product Principle #1: "Do not implement an opaque model that directly
 * predicts only a final score"). */
export function ScoreBreakdownPanel({ analysisId, score }: ScoreBreakdownPanelProps): JSX.Element {
  const recomputeScore = useRecomputeScore(analysisId);

  if (!score) {
    return (
      <div className="rounded-m border border-outline-soft bg-surface-default p-4 text-sm text-content-dim">
        No score computed yet.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 rounded-m border border-outline-soft bg-surface-default p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-content-default">Score breakdown</h3>
        <button
          type="button"
          onClick={() => recomputeScore.mutate()}
          disabled={recomputeScore.isPending}
          className="rounded-full border border-outline-default px-3 py-1.5 text-xs font-semibold text-content-default hover:bg-surface-alt disabled:opacity-60"
        >
          {recomputeScore.isPending ? "Recalculating..." : "Recalculate"}
        </button>
      </div>

      <div className="flex items-baseline gap-2">
        <span className="text-4xl font-bold text-brand-boldest">
          {score.final_score.toFixed(1)}
        </span>
        <span className="text-sm text-content-dim">final score (unofficial)</span>
      </div>

      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <dt className="text-content-dim">Technical (raw)</dt>
          <dd className="font-semibold text-content-default">{score.technical_raw.toFixed(2)}</dd>
        </div>
        <div>
          <dt className="text-content-dim">Technical (scaled)</dt>
          <dd className="font-semibold text-content-default">{score.technical_scaled.toFixed(2)}</dd>
        </div>
        <div>
          <dt className="text-content-dim">Freestyle eval. (raw)</dt>
          <dd className="font-semibold text-content-default">
            {score.freestyle_evaluation_raw.toFixed(2)}
          </dd>
        </div>
        <div>
          <dt className="text-content-dim">Freestyle eval. (scaled)</dt>
          <dd className="font-semibold text-content-default">
            {score.freestyle_evaluation_scaled.toFixed(2)}
          </dd>
        </div>
        <div>
          <dt className="text-content-dim">Major deductions</dt>
          <dd className="font-semibold text-status-alert">-{score.major_deductions.toFixed(2)}</dd>
        </div>
        <div>
          <dt className="text-content-dim">Confidence</dt>
          <dd className="font-semibold text-content-default">
            {Math.round(score.confidence * 100)}%
          </dd>
        </div>
      </dl>

      <p className="text-xs text-content-dim">
        Ruleset <code>{score.ruleset_version}</code>. See the ruleset panel below for its full,
        versioned definition.
      </p>

      {score.warnings.length > 0 ? (
        <ul className="flex flex-col gap-1 rounded-s bg-status-notice/10 p-3 text-sm text-status-notice">
          {score.warnings.map((warning, index) => (
            <li key={index}>{warning}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
