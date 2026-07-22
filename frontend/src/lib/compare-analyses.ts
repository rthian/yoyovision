import type { AnalysisEvent, AnalysisJob, ScoreBreakdown } from "@/lib/types";

export interface CompareMetricRow {
  label: string;
  baseline: string;
  candidate: string;
  delta: string | null;
}

function formatScore(value: number): string {
  return value.toFixed(3);
}

function formatDelta(baseline: number, candidate: number): string {
  const delta = candidate - baseline;
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(3)}`;
}

export function compareScoreBreakdowns(
  baseline: ScoreBreakdown,
  candidate: ScoreBreakdown
): CompareMetricRow[] {
  const rows: Array<{ label: string; key: keyof ScoreBreakdown }> = [
    { label: "Final score", key: "final_score" },
    { label: "Technical (raw)", key: "technical_raw" },
    { label: "Technical (scaled)", key: "technical_scaled" },
    { label: "Freestyle eval (scaled)", key: "freestyle_evaluation_scaled" },
    { label: "Major deductions", key: "major_deductions" },
    { label: "Confidence", key: "confidence" },
  ];

  return rows.map(({ label, key }) => {
    const baseValue = baseline[key];
    const candidateValue = candidate[key];
    if (typeof baseValue !== "number" || typeof candidateValue !== "number") {
      return {
        label,
        baseline: String(baseValue),
        candidate: String(candidateValue),
        delta: null,
      };
    }
    return {
      label,
      baseline: formatScore(baseValue),
      candidate: formatScore(candidateValue),
      delta: formatDelta(baseValue, candidateValue),
    };
  });
}

export function compareEventCounts(
  baselineEvents: AnalysisEvent[],
  candidateEvents: AnalysisEvent[]
): CompareMetricRow {
  const delta = candidateEvents.length - baselineEvents.length;
  const sign = delta > 0 ? "+" : "";
  return {
    label: "Detected events",
    baseline: String(baselineEvents.length),
    candidate: String(candidateEvents.length),
    delta: `${sign}${delta}`,
  };
}

export function compareModelVersions(
  baselineJob: AnalysisJob,
  candidateJob: AnalysisJob
): CompareMetricRow[] {
  const baselineVersions = baselineJob.model_versions ?? {};
  const candidateVersions = candidateJob.model_versions ?? {};
  const keys = Array.from(
    new Set([...Object.keys(baselineVersions), ...Object.keys(candidateVersions)])
  ).sort();

  if (keys.length === 0) {
    return [
      {
        label: "Model versions",
        baseline: "n/a",
        candidate: "n/a",
        delta: null,
      },
    ];
  }

  return keys.map((key) => {
    const baseValue = baselineVersions[key] ?? "—";
    const candidateValue = candidateVersions[key] ?? "—";
    const changed = baseValue !== candidateValue;
    return {
      label: key.replaceAll("_", " "),
      baseline: baseValue,
      candidate: candidateValue,
      delta: changed ? "changed" : "same",
    };
  });
}

export function pickDefaultComparisonJobs(jobs: AnalysisJob[]): {
  baseline: AnalysisJob | null;
  candidate: AnalysisJob | null;
} {
  const completed = jobs.filter((job) => job.status === "completed");
  const baseline = completed.find((job) => !job.is_shadow) ?? null;
  const candidate = completed.find((job) => job.is_shadow) ?? null;
  return { baseline, candidate };
}
