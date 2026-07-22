"use client";

import type { AnalysisReviewState } from "@/lib/types";

interface ReviewLockBannerProps {
  reviewState: AnalysisReviewState;
  submittedAt: string | null;
  isSubmitting: boolean;
  isReopening: boolean;
  onSubmit: () => void;
  onReopen: () => void;
}

/** Shows draft/submitted status and controls for locking the review. */
export function ReviewLockBanner({
  reviewState,
  submittedAt,
  isSubmitting,
  isReopening,
  onSubmit,
  onReopen,
}: ReviewLockBannerProps): JSX.Element {
  const isLocked = reviewState === "submitted";

  return (
    <div
      className={`flex flex-col gap-3 rounded-m border px-4 py-3 sm:flex-row sm:items-center sm:justify-between ${
        isLocked
          ? "border-status-positive/30 bg-status-positive/10"
          : "border-status-informative/30 bg-status-informative/10"
      }`}
    >
      <div>
        <p className="text-sm font-semibold text-content-default">
          {isLocked ? "Review submitted" : "Review in progress"}
        </p>
        <p className="text-sm text-content-dim">
          {isLocked
            ? submittedAt
              ? `Submitted ${new Date(submittedAt).toLocaleString()}. Editing is locked until you reopen.`
              : "Editing is locked until you reopen."
            : "Submit when you are done reviewing to lock edits and mark the record as adjudicated for export."}
        </p>
      </div>
      {isLocked ? (
        <button
          type="button"
          disabled={isReopening}
          onClick={onReopen}
          className="shrink-0 rounded-full border border-outline-default bg-surface-default px-4 py-2 text-sm font-semibold text-content-default hover:bg-surface-alt disabled:opacity-60"
        >
          {isReopening ? "Reopening..." : "Reopen for edits"}
        </button>
      ) : (
        <button
          type="button"
          disabled={isSubmitting}
          onClick={onSubmit}
          className="shrink-0 rounded-full bg-brand-primary-default px-4 py-2 text-sm font-semibold text-white disabled:opacity-60"
        >
          {isSubmitting ? "Submitting..." : "Submit review"}
        </button>
      )}
    </div>
  );
}
