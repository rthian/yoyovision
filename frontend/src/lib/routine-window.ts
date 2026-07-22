import type { AnalysisJob } from "@/lib/types";

export interface RoutineWindow {
  startMs: number;
  endMs: number;
}

/** Resolves the judged routine span for scoring and playback. */
export function resolveRoutineWindow(
  job: Pick<AnalysisJob, "routine_start_ms" | "routine_end_ms">,
  videoDurationMs: number
): RoutineWindow {
  const startMs = job.routine_start_ms ?? 0;
  const endMs = job.routine_end_ms ?? (videoDurationMs > 0 ? videoDurationMs : 0);
  return {
    startMs,
    endMs: videoDurationMs > 0 ? Math.min(endMs, videoDurationMs) : endMs,
  };
}

export function eventInRoutine(
  event: { start_ms: number; end_ms: number },
  window: RoutineWindow
): boolean {
  return event.start_ms >= window.startMs && event.end_ms <= window.endMs;
}

export function deductionInRoutine(
  deduction: { timestamp_ms: number },
  window: RoutineWindow
): boolean {
  return deduction.timestamp_ms >= window.startMs && deduction.timestamp_ms <= window.endMs;
}
