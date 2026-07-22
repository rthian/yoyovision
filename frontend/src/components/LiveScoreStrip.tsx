"use client";

import { formatMsAsTimecode } from "@/lib/format";
import type { LiveScorePreview } from "@/lib/live-score-preview";
import type { Ruleset } from "@/lib/types";

interface LiveScoreStripProps {
  preview: LiveScorePreview;
  ruleset: Ruleset | null;
  activeEventLabel: string | null;
}

/** Running score at the current playhead (completed tricks only). */
export function LiveScoreStrip({
  preview,
  ruleset,
  activeEventLabel,
}: LiveScoreStripProps): JSX.Element {
  const technicalWeight = ruleset?.technical_weight ?? 0.6;
  const freestyleWeight = ruleset?.freestyle_evaluation_weight ?? 0.4;

  return (
    <div className="flex flex-col gap-3 rounded-m border border-outline-soft bg-surface-default p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-content-dim">
            Live at {formatMsAsTimecode(preview.up_to_ms)}
          </p>
          <p className="text-3xl font-bold tabular-nums text-brand-boldest">
            {preview.final_score.toFixed(1)}
          </p>
          <p className="text-xs text-content-dim">
            {preview.completed_event_count} trick
            {preview.completed_event_count === 1 ? "" : "s"} credited so far
          </p>
        </div>
        <div className="text-right text-sm text-content-subtle">
          {activeEventLabel ? (
            <p>
              In progress: <span className="font-semibold text-content-default">{activeEventLabel}</span>
            </p>
          ) : (
            <p className="text-content-dim">No trick in progress</p>
          )}
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div>
          <dt className="text-content-dim">Technical (raw)</dt>
          <dd className="font-semibold text-content-default">{preview.technical_raw.toFixed(2)}</dd>
        </div>
        <div>
          <dt className="text-content-dim">Technical (scaled)</dt>
          <dd className="font-semibold text-content-default">{preview.technical_scaled.toFixed(2)}</dd>
        </div>
        <div>
          <dt className="text-content-dim">Deductions so far</dt>
          <dd className="font-semibold text-status-alert">-{preview.major_deductions.toFixed(2)}</dd>
        </div>
        <div>
          <dt className="text-content-dim">Freestyle (scaled)</dt>
          <dd className="font-semibold text-content-default">
            {preview.freestyle_evaluation_scaled.toFixed(2)}
          </dd>
        </div>
      </dl>

      <p className="text-xs text-content-dim">
        Completed tricks only ({technicalWeight.toFixed(1)} × technical + {freestyleWeight.toFixed(1)} ×
        freestyle − deductions). In-progress tricks credit after they end.
      </p>
    </div>
  );
}
