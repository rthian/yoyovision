"use client";

import { useQuery } from "@tanstack/react-query";

import { getJudgingEntryResults } from "@/lib/api-client";
import type { FeCategoryScores } from "@/lib/types";
import { FE_CATEGORY_COLUMNS } from "@/lib/types";

function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "—";
  }
  return value.toFixed(1);
}

function ScoreCells({
  scores,
  highlightCategories = [],
}: {
  scores: FeCategoryScores;
  highlightCategories?: string[];
}): JSX.Element {
  return (
    <>
      {FE_CATEGORY_COLUMNS.map(({ key }) => {
        const highlighted = highlightCategories.includes(key);
        return (
          <td
            key={key}
            className={`px-2 py-2 text-center text-sm tabular-nums ${
              highlighted ? "bg-status-notice-softest font-semibold" : ""
            }`}
            title={highlighted ? "AI gap-fill" : undefined}
          >
            {formatScore(scores[key])}
          </td>
        );
      })}
    </>
  );
}

interface JudgingEntryResultsProps {
  entryId: string;
}

export function JudgingEntryResults({ entryId }: JudgingEntryResultsProps): JSX.Element {
  const resultsQuery = useQuery({
    queryKey: ["judgingEntryResults", entryId],
    queryFn: () => getJudgingEntryResults(entryId),
  });

  if (resultsQuery.isLoading) {
    return <p className="text-sm text-content-dim">Loading results…</p>;
  }
  if (resultsQuery.isError || !resultsQuery.data) {
    return <p className="text-sm text-status-alert">Could not load results.</p>;
  }

  const results = resultsQuery.data;

  return (
    <section className="flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold text-content-default">Results</h2>
        <p className="text-sm text-content-dim">
          Profile {results.ai_mix_profile} · aggregation {results.aggregation_mode}
        </p>
      </div>

      {results.videos.map((video) => (
        <div key={video.entry_video_id} className="overflow-x-auto rounded-m border border-outline-soft">
          <div className="border-b border-outline-soft bg-surface-alt px-4 py-3">
            <p className="font-semibold text-content-default">
              {video.sort_order + 1}. {video.original_filename}
            </p>
            <p className="text-xs text-content-dim">
              Mode: {video.effective_aggregation_mode}
              {video.ai_virtual_judge_included ? " · AI virtual judge included" : ""}
            </p>
          </div>
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="border-b border-outline-soft text-left text-xs text-content-dim">
                <th className="px-4 py-2">Judge</th>
                {FE_CATEGORY_COLUMNS.map((col) => (
                  <th key={col.key} className="px-2 py-2 text-center">
                    {col.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {video.judges.map((judge) => (
                <tr
                  key={judge.assignment_id}
                  className={`border-b border-outline-softest ${
                    judge.included_in_aggregate ? "" : "opacity-60"
                  }`}
                >
                  <td className="px-4 py-2 text-sm">
                    {judge.display_name}
                    {judge.is_shadow ? " (shadow)" : ""}
                    {!judge.is_submitted ? " (draft)" : ""}
                  </td>
                  <ScoreCells scores={judge.scores} />
                </tr>
              ))}
              <tr className="border-b border-outline-soft bg-brand-softest font-semibold">
                <td className="px-4 py-2 text-sm">Panel aggregate</td>
                <ScoreCells
                  scores={video.panel_aggregate}
                  highlightCategories={video.ai_filled_categories}
                />
              </tr>
              {video.ai_fe ? (
                <tr className="border-b border-outline-softest">
                  <td className="px-4 py-2 text-sm text-content-subtle">
                    AI{video.ai_virtual_judge_included ? " (virtual judge)" : ""}
                  </td>
                  <ScoreCells scores={video.ai_fe} />
                </tr>
              ) : null}
              {video.shadow_fe ? (
                <tr>
                  <td className="px-4 py-2 text-sm text-content-dim">Shadow (compare only)</td>
                  <ScoreCells scores={video.shadow_fe} />
                </tr>
              ) : null}
            </tbody>
          </table>
          {video.warnings.length > 0 ? (
            <ul className="list-disc px-6 py-3 text-xs text-status-notice-boldest">
              {video.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ))}
    </section>
  );
}
