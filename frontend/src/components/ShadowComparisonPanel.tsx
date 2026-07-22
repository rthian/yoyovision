"use client";

import { useMemo, useState } from "react";
import { useQueries } from "@tanstack/react-query";

import {
  compareEventCounts,
  compareModelVersions,
  compareScoreBreakdowns,
  pickDefaultComparisonJobs,
} from "@/lib/compare-analyses";
import { getScore, listEvents } from "@/lib/api-client";
import { formatDateTime } from "@/lib/format";
import type { AnalysisJob } from "@/lib/types";

interface ShadowComparisonPanelProps {
  jobs: AnalysisJob[];
  enabled: boolean;
}

function completedJobs(jobs: AnalysisJob[], shadow: boolean): AnalysisJob[] {
  return jobs.filter((job) => job.status === "completed" && job.is_shadow === shadow);
}

export function ShadowComparisonPanel({
  jobs,
  enabled,
}: ShadowComparisonPanelProps): JSX.Element | null {
  const defaults = useMemo(() => pickDefaultComparisonJobs(jobs), [jobs]);
  const [baselineId, setBaselineId] = useState<string | null>(defaults.baseline?.id ?? null);
  const [candidateId, setCandidateId] = useState<string | null>(defaults.candidate?.id ?? null);

  const officialJobs = completedJobs(jobs, false);
  const shadowJobs = completedJobs(jobs, true);

  const effectiveBaselineId = baselineId ?? defaults.baseline?.id ?? null;
  const effectiveCandidateId = candidateId ?? defaults.candidate?.id ?? null;

  const baselineJob = jobs.find((job) => job.id === effectiveBaselineId) ?? null;
  const candidateJob = jobs.find((job) => job.id === effectiveCandidateId) ?? null;

  const [baselineScoreQuery, candidateScoreQuery, baselineEventsQuery, candidateEventsQuery] =
    useQueries({
      queries: [
        {
          queryKey: ["analyses", effectiveBaselineId, "score"],
          queryFn: () => getScore(effectiveBaselineId as string),
          enabled: enabled && Boolean(effectiveBaselineId),
        },
        {
          queryKey: ["analyses", effectiveCandidateId, "score"],
          queryFn: () => getScore(effectiveCandidateId as string),
          enabled: enabled && Boolean(effectiveCandidateId),
        },
        {
          queryKey: ["analyses", effectiveBaselineId, "events"],
          queryFn: () => listEvents(effectiveBaselineId as string),
          enabled: enabled && Boolean(effectiveBaselineId),
        },
        {
          queryKey: ["analyses", effectiveCandidateId, "events"],
          queryFn: () => listEvents(effectiveCandidateId as string),
          enabled: enabled && Boolean(effectiveCandidateId),
        },
      ],
    });

  if (officialJobs.length === 0 || shadowJobs.length === 0) {
    return null;
  }

  const isLoading =
    baselineScoreQuery.isLoading ||
    candidateScoreQuery.isLoading ||
    baselineEventsQuery.isLoading ||
    candidateEventsQuery.isLoading;

  const scoreRows =
    baselineScoreQuery.data && candidateScoreQuery.data
      ? compareScoreBreakdowns(baselineScoreQuery.data, candidateScoreQuery.data)
      : [];

  const eventRow =
    baselineEventsQuery.data && candidateEventsQuery.data
      ? compareEventCounts(baselineEventsQuery.data, candidateEventsQuery.data)
      : null;

  const modelRows =
    baselineJob && candidateJob ? compareModelVersions(baselineJob, candidateJob) : [];

  return (
    <section className="flex flex-col gap-4 rounded-m border border-outline-soft bg-surface-default p-4">
      <div>
        <h2 className="text-lg font-semibold text-content-default">Shadow comparison</h2>
        <p className="mt-1 text-sm text-content-dim">
          Compare an official run against a shadow run to evaluate a new adapter or model
          configuration before promoting it.
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-semibold text-content-default">Official baseline</span>
          <select
            className="h-10 rounded-s border border-outline-soft bg-surface-default px-3"
            value={effectiveBaselineId ?? ""}
            onChange={(event) => setBaselineId(event.target.value)}
          >
            {officialJobs.map((job) => (
              <option key={job.id} value={job.id}>
                {formatDateTime(job.created_at)} — {job.pipeline_version}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-semibold text-content-default">Shadow candidate</span>
          <select
            className="h-10 rounded-s border border-outline-soft bg-surface-default px-3"
            value={effectiveCandidateId ?? ""}
            onChange={(event) => setCandidateId(event.target.value)}
          >
            {shadowJobs.map((job) => (
              <option key={job.id} value={job.id}>
                {formatDateTime(job.created_at)} — {job.pipeline_version}
              </option>
            ))}
          </select>
        </label>
      </div>

      {isLoading ? <p className="text-sm text-content-dim">Loading comparison...</p> : null}

      {!isLoading && scoreRows.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-outline-softest text-left text-content-dim">
                <th className="py-2 pr-4 font-semibold">Metric</th>
                <th className="py-2 pr-4 font-semibold">Official</th>
                <th className="py-2 pr-4 font-semibold">Shadow</th>
                <th className="py-2 font-semibold">Delta</th>
              </tr>
            </thead>
            <tbody>
              {[...scoreRows, ...(eventRow ? [eventRow] : [])].map((row) => (
                <tr key={row.label} className="border-b border-outline-softest">
                  <td className="py-2 pr-4 text-content-default">{row.label}</td>
                  <td className="py-2 pr-4 tabular-nums text-content-dim">{row.baseline}</td>
                  <td className="py-2 pr-4 tabular-nums text-content-dim">{row.candidate}</td>
                  <td className="py-2 tabular-nums text-content-default">{row.delta ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {!isLoading && modelRows.length > 0 ? (
        <div>
          <h3 className="text-sm font-semibold text-content-default">Adapter / model versions</h3>
          <ul className="mt-2 flex flex-col gap-1 text-sm text-content-dim">
            {modelRows.map((row) => (
              <li key={row.label}>
                <span className="font-semibold text-content-default">{row.label}:</span>{" "}
                {row.baseline} → {row.candidate}
                {row.delta === "changed" ? (
                  <span className="ml-2 text-status-informative">changed</span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
