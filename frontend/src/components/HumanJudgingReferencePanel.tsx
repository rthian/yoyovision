"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";

import { getAnalysisHumanJudgingReference } from "@/lib/api-client";
import { formatMsAsTimecode } from "@/lib/format";

interface HumanJudgingReferencePanelProps {
  analysisId: string;
  onSeek: (ms: number) => void;
}

export function HumanJudgingReferencePanel({
  analysisId,
  onSeek,
}: HumanJudgingReferencePanelProps): JSX.Element {
  const referenceQuery = useQuery({
    queryKey: ["analyses", analysisId, "human-judging-reference"],
    queryFn: () => getAnalysisHumanJudgingReference(analysisId),
  });

  if (referenceQuery.isLoading) {
    return <p className="text-sm text-content-dim">Loading human judging reference…</p>;
  }
  if (referenceQuery.isError || !referenceQuery.data) {
    return (
      <p className="rounded-s border border-status-alert/20 bg-status-alert/5 p-3 text-sm text-status-alert">
        Human judging reference could not be loaded.
      </p>
    );
  }

  const entries = referenceQuery.data.entries;
  return (
    <section className="rounded-m border border-outline-soft bg-surface-default p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-semibold text-content-default">Human judging reference</h2>
          <p className="mt-1 text-sm text-content-dim">
            Judge clicks and evaluations stay separate from the AI score and provide comparison evidence.
          </p>
        </div>
        <Link
          href="/admin/judging-entries"
          className="rounded-full border border-outline-default px-4 py-2 text-sm font-semibold hover:bg-surface-alt"
        >
          Manage judging entries
        </Link>
      </div>

      {entries.length === 0 ? (
        <p className="mt-4 rounded-s bg-surface-alt p-3 text-sm text-content-dim">
          No judging entry references this video yet. Create an entry, invite judges, and submit their scores to compare them here.
        </p>
      ) : (
        <div className="mt-4 flex flex-col gap-4">
          {entries.map((entry) => {
            const submitted = entry.judges.filter((judge) => judge.is_submitted);
            return (
              <div key={entry.entry_id} className="rounded-s border border-outline-soft p-3">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <Link href={`/admin/judging-entries/${entry.entry_id}`} className="font-semibold text-brand-boldest hover:underline">
                      {entry.title}
                    </Link>
                    <p className="text-xs text-content-dim">{entry.mode} · {entry.status} · {submitted.length}/{entry.judges.length} submitted</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xl font-bold tabular-nums text-content-default">
                      {entry.panel_net_technical_clicks?.toFixed(1) ?? "—"}
                    </p>
                    <p className="text-xs text-content-dim">panel average net clicks</p>
                  </div>
                </div>
                <div className="mt-3 grid gap-2 sm:grid-cols-2">
                  {entry.judges.map((judge) => (
                    <details key={judge.assignment_id} className="rounded-s bg-surface-alt p-2">
                      <summary className="cursor-pointer text-xs font-semibold text-content-subtle">
                        {judge.display_name}: +{judge.positive_clicks} / −{judge.negative_clicks} = {judge.net_technical_clicks}
                      </summary>
                      <div className="mt-2 flex max-h-28 flex-wrap gap-1 overflow-y-auto">
                        {judge.technical_clicks.map((click) => (
                          <button
                            key={click.id}
                            type="button"
                            onClick={() => onSeek(click.timestamp_ms)}
                            className={`rounded-full px-2 py-1 text-xs font-semibold ${
                              click.kind === "positive"
                                ? "bg-status-positive/10 text-status-positive"
                                : "bg-status-alert/10 text-status-alert"
                            }`}
                          >
                            {click.kind === "positive" ? "+" : "−"} {formatMsAsTimecode(click.timestamp_ms)}
                          </button>
                        ))}
                      </div>
                    </details>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
