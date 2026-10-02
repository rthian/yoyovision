"use client";

import { useMemo } from "react";
import { useParams } from "next/navigation";
import { useState } from "react";

import { AuthGate } from "@/components/AuthGate";
import { AnalysisTrainingContext } from "@/components/AnalysisTrainingContext";
import { CompactEventFeed } from "@/components/CompactEventFeed";
import { DeductionTable } from "@/components/DeductionTable";
import { EventTable } from "@/components/EventTable";
import { EventTimeline } from "@/components/EventTimeline";
import { ExportButtons } from "@/components/ExportButtons";
import { FreestyleEvaluationForm } from "@/components/FreestyleEvaluationForm";
import { LiveScoreStrip } from "@/components/LiveScoreStrip";
import { HumanJudgingReferencePanel } from "@/components/HumanJudgingReferencePanel";
import { ReviewLockBanner } from "@/components/ReviewLockBanner";
import { RoutineWindowPanel } from "@/components/RoutineWindowPanel";
import { RulesetPicker } from "@/components/RulesetPicker";
import { RulesetPanel } from "@/components/RulesetPanel";
import { ScoreBreakdownPanel } from "@/components/ScoreBreakdownPanel";
import { VideoPlayerWithOverlay } from "@/components/VideoPlayerWithOverlay";

import { useAnalysisJob, useReopenAnalysis, useScore, useScoreLineItems, useSubmitAnalysis, useUpdateAnalysisRuleset, useUpdateRoutineWindow } from "@/hooks/useAnalysis";
import type { TechnicalLineItem } from "@/lib/types";
import { computeLiveScorePreview } from "@/lib/live-score-preview";
import { formatMsAsTimecode } from "@/lib/format";
import { eventInRoutine, resolveRoutineWindow } from "@/lib/routine-window";
import { useAuth } from "@/hooks/useAuth";
import { useDeductions } from "@/hooks/useDeductions";
import { useEvaluation } from "@/hooks/useEvaluation";
import { useEvents } from "@/hooks/useEvents";
import { useRuleset, useRulesets } from "@/hooks/useRulesets";
import { useTrainingAnnotations } from "@/hooks/useTrainingAnnotations";
import { useVideo } from "@/hooks/useVideos";
import { useVideoBlobUrl } from "@/hooks/useVideoBlobUrl";

type ReviewFilter = "all" | "pending" | "low-confidence" | "uncertain" | "edited";

const REVIEW_FILTER_LABELS: Record<ReviewFilter, string> = {
  all: "All",
  pending: "Pending",
  "low-confidence": "Low confidence",
  uncertain: "Uncertain",
  edited: "Edited",
};

function AnalysisReview({ analysisId }: { analysisId: string }): JSX.Element {
  const { isAuthenticated } = useAuth();
  const [currentMs, setCurrentMs] = useState(0);
  const [seekToMs, setSeekToMs] = useState<number | null>(null);
  const [showFullEventTable, setShowFullEventTable] = useState(false);
  const [reviewFilter, setReviewFilter] = useState<ReviewFilter>("all");

  const jobQuery = useAnalysisJob(analysisId, isAuthenticated);
  const job = jobQuery.data;

  const videoQuery = useVideo(job?.video_id ?? "", isAuthenticated && Boolean(job));
  const { blobUrl } = useVideoBlobUrl(job?.video_id, isAuthenticated && Boolean(job));
  const trainingAnnotationsQuery = useTrainingAnnotations(
    job?.video_id ?? "",
    isAuthenticated && Boolean(job)
  );

  const eventsQuery = useEvents(analysisId, isAuthenticated);
  const deductionsQuery = useDeductions(analysisId, isAuthenticated);
  const scoreQuery = useScore(analysisId, isAuthenticated);
  const lineItemsQuery = useScoreLineItems(
    analysisId,
    isAuthenticated && job?.status === "completed"
  );
  const updateRoutineWindow = useUpdateRoutineWindow(analysisId);
  const submitAnalysis = useSubmitAnalysis(analysisId);
  const reopenAnalysis = useReopenAnalysis(analysisId);
  const evaluationQuery = useEvaluation(analysisId, isAuthenticated);
  const rulesetVersion = job?.ruleset_version ?? scoreQuery.data?.ruleset_version;
  const rulesetQuery = useRuleset(rulesetVersion, isAuthenticated);
  const rulesetsQuery = useRulesets(isAuthenticated);
  const updateRuleset = useUpdateAnalysisRuleset(analysisId);

  const allEvents = eventsQuery.data ?? [];
  const videoDurationMs = videoQuery.data?.duration_ms ?? 0;
  const routineWindow =
    job?.status === "completed" ? resolveRoutineWindow(job, videoDurationMs) : null;
  const events = routineWindow
    ? allEvents.filter((event) => eventInRoutine(event, routineWindow))
    : allEvents;
  const outsideRoutineEventCount = allEvents.length - events.length;
  const reviewedEventCount = events.filter((event) => event.review_status !== "pending").length;
  const reviewProgress = events.length > 0 ? Math.round((reviewedEventCount / events.length) * 100) : 0;
  const filterCounts: Record<ReviewFilter, number> = {
    all: events.length,
    pending: events.filter((event) => event.review_status === "pending").length,
    "low-confidence": events.filter((event) => event.confidence < 0.7).length,
    uncertain: events.filter((event) => event.outcome === "uncertain").length,
    edited: events.filter((event) => event.review_status === "edited").length,
  };
  const filteredEvents = events.filter((event) => {
    if (reviewFilter === "pending") return event.review_status === "pending";
    if (reviewFilter === "low-confidence") return event.confidence < 0.7;
    if (reviewFilter === "uncertain") return event.outcome === "uncertain";
    if (reviewFilter === "edited") return event.review_status === "edited";
    return true;
  });
  const nextEventNeedingReview = events.find(
    (event) =>
      event.review_status === "pending" || event.confidence < 0.7 || event.outcome === "uncertain"
  );
  const lineItemsByEventId = useMemo(() => {
    const map = new Map<string, TechnicalLineItem>();
    for (const item of lineItemsQuery.data?.technical_line_items ?? []) {
      if (item.event_id) {
        map.set(item.event_id, item);
      }
    }
    return map;
  }, [lineItemsQuery.data]);

  const score = scoreQuery.data ?? null;
  const ruleset = rulesetQuery.data ?? null;
  const deductions = deductionsQuery.data ?? [];
  const livePreview =
    score && job?.status === "completed" && routineWindow
      ? computeLiveScorePreview(
          events,
          lineItemsByEventId,
          deductions,
          score,
          ruleset ?? {
            version: score.ruleset_version,
            is_official: false,
            disclaimer: "",
            difficulty_band_points: {},
            repeated_element_decay: {},
            deduction_rules: [],
            freestyle_evaluation_weights: {},
            technical_scale_max: 100,
            freestyle_evaluation_scale_max: 100,
            technical_weight: 0.6,
            freestyle_evaluation_weight: 0.4,
          },
          currentMs,
          routineWindow
        )
      : null;
  const activeEventLabel =
    events.find((event) => event.id === livePreview?.active_event_id)?.label ?? null;
  const lastEventEndMs = events.reduce((max, event) => Math.max(max, event.end_ms), 0);
  const timelineDurationMs = videoDurationMs;
  const eventCoverageShort =
    routineWindow !== null &&
    events.length > 0 &&
    lastEventEndMs < routineWindow.endMs - (routineWindow.endMs - routineWindow.startMs) * 0.1;
  const isLocked = (job?.review_state ?? "draft") === "submitted";

  if (jobQuery.isLoading) {
    return <p className="text-sm text-content-dim">Loading analysis...</p>;
  }
  if (jobQuery.isError || !job) {
    return (
      <p role="alert" className="text-sm text-status-alert">
        Analysis not found.
      </p>
    );
  }
  if (job.status !== "completed") {
    return (
      <p className="text-sm text-content-dim">
        This analysis is still {job.status}. Come back once it has completed.
      </p>
    );
  }

  function handleSeek(ms: number): void {
    setSeekToMs(ms);
    setCurrentMs(ms);
  }

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-brand-boldest">Review and score</p>
          <h1 className="text-2xl font-bold text-content-default">Analysis review</h1>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-content-dim">
            <span className="rounded-full bg-brand-primary-softest px-2.5 py-1 font-semibold text-brand-boldest">
              {job.division}
            </span>
            <span>{videoQuery.data?.player_id || "Unknown performer"}</span>
            <span aria-hidden="true">·</span>
            <span>Pipeline {job.pipeline_version}</span>
          </div>
        </div>
        <ExportButtons analysisId={analysisId} reviewState={job.review_state ?? "draft"} />
      </div>

      <nav aria-label="Video workflow" className="rounded-m border border-outline-soft bg-surface-default px-4 py-3">
        <ol className="grid grid-cols-3 gap-2 text-sm">
          <li className="flex items-center gap-2 text-status-positive">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-status-positive/15 font-bold">✓</span>
            <span className="font-semibold">1. Add video</span>
          </li>
          <li className="flex items-center gap-2 text-status-positive">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-status-positive/15 font-bold">✓</span>
            <span className="font-semibold">2. Analyze</span>
          </li>
          <li aria-current="step" className="flex items-center gap-2 text-brand-boldest">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-brand-primary-softest font-bold">3</span>
            <span className="font-semibold">Review</span>
          </li>
        </ol>
      </nav>

      {job.is_shadow ? (
        <p
          role="status"
          className="rounded-m border border-status-informative/30 bg-status-informative/10 px-4 py-3 text-sm text-status-informative"
        >
          This is a shadow-mode run (Prompt F): its events, deductions, and score are real, but
          it is not this video&apos;s official result.
        </p>
      ) : null}

      <ReviewLockBanner
        reviewState={job.review_state ?? "draft"}
        submittedAt={job.submitted_at}
        isSubmitting={submitAnalysis.isPending}
        isReopening={reopenAnalysis.isPending}
        unresolvedEventCount={filterCounts.pending}
        onSubmit={() => void submitAnalysis.mutateAsync()}
        onReopen={() => void reopenAnalysis.mutateAsync()}
      />

      <section className="rounded-m border border-outline-soft bg-surface-default p-4" aria-labelledby="review-progress-heading">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 id="review-progress-heading" className="font-semibold text-content-default">Review progress</h2>
            <p className="text-sm text-content-dim">
              {reviewedEventCount} of {events.length} events reviewed
              {filterCounts.pending > 0 ? ` · ${filterCounts.pending} still need a decision` : " · Ready to submit"}
            </p>
          </div>
          {nextEventNeedingReview && !isLocked ? (
            <button
              type="button"
              onClick={() => {
                setReviewFilter(
                  nextEventNeedingReview.review_status === "pending"
                    ? "pending"
                    : nextEventNeedingReview.confidence < 0.7
                      ? "low-confidence"
                      : "uncertain"
                );
                setShowFullEventTable(true);
                handleSeek(nextEventNeedingReview.start_ms);
              }}
              className="shrink-0 rounded-full bg-brand-primary px-4 py-2 text-sm font-semibold text-white"
            >
              Review next item
            </button>
          ) : null}
        </div>
        <div
          className="mt-3 h-2 overflow-hidden rounded-full bg-outline-softest"
          role="progressbar"
          aria-label="Event review progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={reviewProgress}
        >
          <div className="h-full rounded-full bg-brand-primary transition-all" style={{ width: `${reviewProgress}%` }} />
        </div>
        <div className="mt-4 flex flex-wrap gap-2" aria-label="Filter review events">
          {(Object.keys(REVIEW_FILTER_LABELS) as ReviewFilter[]).map((filter) => (
            <button
              key={filter}
              type="button"
              aria-pressed={reviewFilter === filter}
              onClick={() => {
                setReviewFilter(filter);
                setShowFullEventTable(true);
              }}
              className={`rounded-full border px-3 py-1.5 text-xs font-semibold ${
                reviewFilter === filter
                  ? "border-brand-primary bg-brand-primary-softest text-brand-boldest"
                  : "border-outline-soft text-content-subtle hover:bg-surface-alt"
              }`}
            >
              {REVIEW_FILTER_LABELS[filter]} · {filterCounts[filter]}
            </button>
          ))}
        </div>
      </section>

      {lineItemsQuery.isError ? (
        <p role="alert" className="rounded-m border border-status-alert/30 bg-status-alert/10 px-4 py-3 text-sm text-status-alert">
          Could not load per-trick scoring rows. Live technical points may stay at 0 until this is
          fixed. Try refreshing the page.
        </p>
      ) : null}

      {eventCoverageShort ? (
        <p
          role="status"
          className="rounded-m border border-status-notice/30 bg-status-notice/10 px-4 py-3 text-sm text-status-notice"
        >
          The last detected trick ends at {formatMsAsTimecode(lastEventEndMs)}, before the routine
          ends at {formatMsAsTimecode(routineWindow?.endMs ?? videoDurationMs)}. Re-run analysis
          to refresh detection across the full routine.
        </p>
      ) : null}

      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(320px,2fr)]">
        <div className="flex min-w-0 flex-col gap-3">
          <VideoPlayerWithOverlay
            src={blobUrl}
            events={events}
            onTimeUpdateMs={setCurrentMs}
            seekToMs={seekToMs}
            routineStartMs={routineWindow?.startMs}
            routineEndMs={routineWindow?.endMs}
          />
          <EventTimeline
            events={events}
            durationMs={timelineDurationMs}
            currentMs={currentMs}
            onSeek={handleSeek}
            routineStartMs={routineWindow?.startMs}
            routineEndMs={routineWindow?.endMs}
          />
          {livePreview ? (
            <LiveScoreStrip
              preview={livePreview}
              ruleset={ruleset}
              activeEventLabel={activeEventLabel}
            />
          ) : null}
        </div>

        <section className="flex min-w-0 flex-col gap-3 xl:sticky xl:top-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-content-default">Trick events</h2>
              <p className="text-sm text-content-dim">
                In routine · {Math.min(8, events.length)} of {events.length}
              </p>
            </div>
            <button
              type="button"
              aria-expanded={showFullEventTable}
              onClick={() => setShowFullEventTable((value) => !value)}
              className="shrink-0 rounded-full border border-outline-soft bg-surface-default px-3 py-1.5 text-xs font-semibold text-content-default hover:bg-surface-alt"
            >
              {showFullEventTable ? "Hide full table" : "View full table"}
            </button>
          </div>
          <CompactEventFeed
            events={events}
            lineItemsByEventId={lineItemsByEventId}
            currentMs={currentMs}
            activeEventId={livePreview?.active_event_id ?? null}
            onSeek={handleSeek}
          />
        </section>
      </div>

      {outsideRoutineEventCount > 0 ? (
        <p className="rounded-s border border-outline-soft bg-surface-alt px-3 py-2 text-sm text-content-dim">
          {outsideRoutineEventCount} detected event{outsideRoutineEventCount === 1 ? " is" : "s are"} outside the routine window and hidden. They receive no technical points.
        </p>
      ) : null}

      <AnalysisTrainingContext
        videoId={job.video_id}
        annotations={trainingAnnotationsQuery.data ?? []}
        isLoading={trainingAnnotationsQuery.isLoading}
        routineWindow={routineWindow}
        modelVersions={job.model_versions}
        onSeek={handleSeek}
      />

      <HumanJudgingReferencePanel analysisId={analysisId} onSeek={handleSeek} />

      {showFullEventTable ? (
        <section className="flex flex-col gap-3">
          <div>
            <h2 className="text-lg font-semibold text-content-default">Full event review</h2>
            <p className="text-sm text-content-dim">
              {filteredEvents.length === events.length
                ? `${events.length} detected events with full editing and review controls`
                : `${filteredEvents.length} of ${events.length} events · ${REVIEW_FILTER_LABELS[reviewFilter]}`}
            </p>
          </div>
          <EventTable
            analysisId={analysisId}
            events={filteredEvents}
            lineItemsByEventId={lineItemsByEventId}
            currentMs={currentMs}
            activeEventId={livePreview?.active_event_id ?? null}
            onSeek={handleSeek}
            readOnly={isLocked}
          />
        </section>
      ) : null}

      {routineWindow ? (
        <RoutineWindowPanel
          key={`${job.routine_start_ms ?? 0}-${job.routine_end_ms ?? videoDurationMs}`}
          window={routineWindow}
          currentMs={currentMs}
          videoDurationMs={videoDurationMs}
          isSaving={updateRoutineWindow.isPending}
          readOnly={isLocked}
          onSetStartToPlayhead={() => handleSeek(currentMs)}
          onSetEndToPlayhead={() => handleSeek(currentMs)}
          onSave={async (startMs, endMs) => {
            await updateRoutineWindow.mutateAsync({
              routine_start_ms: startMs,
              routine_end_ms: endMs,
            });
          }}
        />
      ) : null}

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-content-default">Major deductions</h2>
        <DeductionTable analysisId={analysisId} deductions={deductionsQuery.data ?? []} readOnly={isLocked} />
      </section>

      <section>
        <FreestyleEvaluationForm
          analysisId={analysisId}
          evaluation={evaluationQuery.data ?? null}
          readOnly={isLocked}
        />
      </section>

      <section className="flex flex-col gap-3">
        <ScoreBreakdownPanel
          analysisId={analysisId}
          score={score}
          ruleset={ruleset}
        />
        <RulesetPicker
          rulesets={rulesetsQuery.data ?? []}
          selectedVersion={rulesetVersion ?? "1a-draft-0.1"}
          disabled={isLocked || updateRuleset.isPending}
          onChange={(version) => {
            if (version !== rulesetVersion) {
              updateRuleset.mutate(version);
            }
          }}
        />
        <RulesetPanel ruleset={rulesetQuery.data ?? null} />
      </section>
    </div>
  );
}

export default function AnalysisReviewPage(): JSX.Element {
  const params = useParams<{ analysisId: string }>();
  return (
    <AuthGate>
      <AnalysisReview analysisId={params.analysisId} />
    </AuthGate>
  );
}
