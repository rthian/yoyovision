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
import { RulesetPanel } from "@/components/RulesetPanel";
import { ScoreBreakdownPanel } from "@/components/ScoreBreakdownPanel";
import { VideoPlayerWithOverlay } from "@/components/VideoPlayerWithOverlay";

import { useAnalysisJob, useScore, useScoreLineItems } from "@/hooks/useAnalysis";
import type { TechnicalLineItem } from "@/lib/types";
import { computeLiveScorePreview } from "@/lib/live-score-preview";
import { useAuth } from "@/hooks/useAuth";
import { useDeductions } from "@/hooks/useDeductions";
import { useEvaluation } from "@/hooks/useEvaluation";
import { useEvents } from "@/hooks/useEvents";
import { useRuleset } from "@/hooks/useRulesets";
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
  const lineItemsQuery = useScoreLineItems(analysisId, isAuthenticated);
  const evaluationQuery = useEvaluation(analysisId, isAuthenticated);
  const rulesetQuery = useRuleset(scoreQuery.data?.ruleset_version, isAuthenticated);

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
  const livePreview =
    score && ruleset
      ? computeLiveScorePreview(
          events,
          lineItemsByEventId,
          deductions,
          score,
          ruleset,
          currentMs
        )
      : null;
  const activeEventLabel =
    events.find((event) => event.id === livePreview?.active_event_id)?.label ?? null;

  function handleSeek(ms: number): void {
    setSeekToMs(ms);
    setCurrentMs(ms);
  }

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-content-default">Analysis review</h1>
        <ExportButtons analysisId={analysisId} />
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

      <div className="flex flex-col gap-3">
        <VideoPlayerWithOverlay
          src={blobUrl}
          events={events}
          onTimeUpdateMs={setCurrentMs}
          seekToMs={seekToMs}
        />
        <EventTimeline
          events={events}
          durationMs={videoQuery.data?.duration_ms ?? 0}
          currentMs={currentMs}
          onSeek={handleSeek}
        />
        {livePreview ? (
          <LiveScoreStrip
            preview={livePreview}
            ruleset={ruleset}
            activeEventLabel={activeEventLabel}
          />
        ) : null}
      </div>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-content-default">Trick events</h2>
        <EventTable
          analysisId={analysisId}
          events={events}
          lineItemsByEventId={lineItemsByEventId}
          currentMs={currentMs}
          activeEventId={livePreview?.active_event_id ?? null}
          onSeek={handleSeek}
        />
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-content-default">Major deductions</h2>
        <DeductionTable analysisId={analysisId} deductions={deductionsQuery.data ?? []} />
      </section>

      <section>
        <FreestyleEvaluationForm
          analysisId={analysisId}
          evaluation={evaluationQuery.data ?? null}
        />
      </section>

      <section className="flex flex-col gap-3">
        <ScoreBreakdownPanel
          analysisId={analysisId}
          score={score}
          ruleset={ruleset}
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
