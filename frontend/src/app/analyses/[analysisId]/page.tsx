"use client";

import { useMemo } from "react";
import { useParams } from "next/navigation";
import { useState } from "react";

import { AuthGate } from "@/components/AuthGate";
import { DeductionTable } from "@/components/DeductionTable";
import { EventTable } from "@/components/EventTable";
import { EventTimeline } from "@/components/EventTimeline";
import { ExportButtons } from "@/components/ExportButtons";
import { FreestyleEvaluationForm } from "@/components/FreestyleEvaluationForm";
import { LiveScoreStrip } from "@/components/LiveScoreStrip";
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
import { resolveRoutineWindow } from "@/lib/routine-window";
import { useAuth } from "@/hooks/useAuth";
import { useDeductions } from "@/hooks/useDeductions";
import { useEvaluation } from "@/hooks/useEvaluation";
import { useEvents } from "@/hooks/useEvents";
import { useRuleset, useRulesets } from "@/hooks/useRulesets";
import { useVideo } from "@/hooks/useVideos";
import { useVideoBlobUrl } from "@/hooks/useVideoBlobUrl";

function AnalysisReview({ analysisId }: { analysisId: string }): JSX.Element {
  const { isAuthenticated } = useAuth();
  const [currentMs, setCurrentMs] = useState(0);
  const [seekToMs, setSeekToMs] = useState<number | null>(null);

  const jobQuery = useAnalysisJob(analysisId, isAuthenticated);
  const job = jobQuery.data;

  const videoQuery = useVideo(job?.video_id ?? "", isAuthenticated && Boolean(job));
  const { blobUrl } = useVideoBlobUrl(job?.video_id, isAuthenticated && Boolean(job));

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

  const events = eventsQuery.data ?? [];
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
  const videoDurationMs = videoQuery.data?.duration_ms ?? 0;
  const routineWindow =
    job?.status === "completed" ? resolveRoutineWindow(job, videoDurationMs) : null;
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
  const timelineDurationMs = Math.max(videoDurationMs, lastEventEndMs);
  const eventCoverageShort =
    videoDurationMs > 0 && lastEventEndMs > 0 && lastEventEndMs < videoDurationMs * 0.9;
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
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-content-default">Analysis review</h1>
        <ExportButtons analysisId={analysisId} reviewState={job.review_state ?? "draft"} />
      </div>

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
        onSubmit={() => void submitAnalysis.mutateAsync()}
        onReopen={() => void reopenAnalysis.mutateAsync()}
      />

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
          Detected tricks only cover about {formatMsAsTimecode(lastEventEndMs)} of this{" "}
          {formatMsAsTimecode(videoDurationMs)} video. Re-run analysis on this video to refresh
          event detection across the full routine.
        </p>
      ) : null}

      <div className="flex flex-col gap-3">
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
        <h2 className="text-lg font-semibold text-content-default">Trick events</h2>
        <EventTable
          analysisId={analysisId}
          events={events}
          lineItemsByEventId={lineItemsByEventId}
          currentMs={currentMs}
          activeEventId={livePreview?.active_event_id ?? null}
          onSeek={handleSeek}
          readOnly={isLocked}
        />
      </section>

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
