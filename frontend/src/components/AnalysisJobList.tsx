"use client";

import Link from "next/link";

import { formatDateTime } from "@/lib/format";
import type { AnalysisJob } from "@/lib/types";

const STATUS_LABELS: Record<AnalysisJob["status"], string> = {
  pending: "Pending",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

const STAGE_LABELS: Record<NonNullable<AnalysisJob["current_stage"]>, string> = {
  queued: "Queued",
  media_validation: "Validating media",
  preprocessing: "Preprocessing",
  pose_extraction: "Extracting pose",
  hand_extraction: "Extracting hands",
  yoyo_detection: "Detecting yo-yo",
  tracking: "Tracking",
  string_analysis: "Analyzing string",
  feature_extraction: "Extracting features",
  temporal_event_detection: "Detecting trick events",
  scoring: "Scoring",
  done: "Done",
};

const CANCELLABLE_STATUSES: ReadonlySet<AnalysisJob["status"]> = new Set(["pending", "running"]);
const DELETABLE_STATUSES: ReadonlySet<AnalysisJob["status"]> = new Set([
  "completed",
  "failed",
  "cancelled",
]);

interface AnalysisJobListProps {
  jobs: AnalysisJob[];
  /** Requests cancellation for a pending/running job (Prompt F). Optional
   * so this component still works read-only (e.g. no ownership context). */
  onCancel?: (analysisId: string) => void;
  /** Permanently removes a finished analysis run. Optional for read-only use. */
  onDelete?: (analysisId: string) => void;
  /** Analysis id whose cancel request is currently in flight, if any --
   * disables that row's button and shows a pending label. */
  cancellingId?: string;
  /** Analysis id whose delete request is currently in flight, if any. */
  deletingId?: string;
}

export function AnalysisJobList({
  jobs,
  onCancel,
  onDelete,
  cancellingId,
  deletingId,
}: AnalysisJobListProps): JSX.Element {
  if (jobs.length === 0) {
    return <p className="text-sm text-content-dim">No analysis runs yet.</p>;
  }

  return (
    <ul className="flex flex-col gap-3">
      {jobs.map((job) => {
        const isCancellable = CANCELLABLE_STATUSES.has(job.status) && !job.cancel_requested;
        const isCancelling = cancellingId === job.id;
        const isDeletable = DELETABLE_STATUSES.has(job.status);
        const isDeleting = deletingId === job.id;

        return (
          <li
            key={job.id}
            className="flex items-center justify-between rounded-m border border-outline-soft bg-surface-default p-4"
          >
            <div className="flex flex-col gap-1">
              <span className="flex items-center gap-2 font-semibold text-content-default">
                {STATUS_LABELS[job.status]}
                {job.current_stage ? ` - ${STAGE_LABELS[job.current_stage]}` : ""}
                {job.is_shadow ? (
                  <span className="rounded-s bg-status-informative/15 px-2 py-0.5 text-xs font-semibold text-status-informative">
                    Shadow
                  </span>
                ) : null}
                {job.cancel_requested && job.status !== "cancelled" ? (
                  <span className="rounded-s bg-status-notice/15 px-2 py-0.5 text-xs font-semibold text-status-notice">
                    Cancelling...
                  </span>
                ) : null}
              </span>
              <span className="text-sm text-content-dim">
                Started {formatDateTime(job.created_at)} - pipeline {job.pipeline_version}
              </span>
              {job.model_versions && Object.keys(job.model_versions).length > 0 ? (
                <span className="text-xs text-content-dim">
                  {Object.entries(job.model_versions)
                    .map(([role, version]) => `${role}: ${version}`)
                    .join(" · ")}
                </span>
              ) : null}
              {job.status === "failed" && job.error_message ? (
                <span role="alert" className="text-sm text-status-alert">
                  {job.error_message}
                </span>
              ) : null}
            </div>
            <div className="flex items-center gap-2">
              {onCancel && isCancellable ? (
                <button
                  type="button"
                  onClick={() => onCancel(job.id)}
                  disabled={isCancelling}
                  className="rounded-full border border-outline-default px-4 py-2 text-sm font-semibold text-status-alert hover:bg-status-alert/10 disabled:opacity-60"
                >
                  {isCancelling ? "Cancelling..." : "Cancel"}
                </button>
              ) : null}
              {onDelete && isDeletable ? (
                <button
                  type="button"
                  onClick={() => {
                    if (
                      window.confirm(
                        "Delete this analysis run? Events, deductions, and scores will be permanently removed."
                      )
                    ) {
                      onDelete(job.id);
                    }
                  }}
                  disabled={isDeleting}
                  className="rounded-full border border-outline-default px-4 py-2 text-sm font-semibold text-status-alert hover:bg-status-alert/10 disabled:opacity-60"
                >
                  {isDeleting ? "Deleting..." : "Delete"}
                </button>
              ) : null}
              {job.status === "completed" ? (
                <Link
                  href={`/analyses/${job.id}`}
                  className="rounded-full bg-brand-primary px-4 py-2 text-sm font-semibold text-white"
                >
                  Review
                </Link>
              ) : null}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
