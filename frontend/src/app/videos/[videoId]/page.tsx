"use client";

import { useParams } from "next/navigation";
import { useState } from "react";

import { AnalysisJobList } from "@/components/AnalysisJobList";
import { ShadowComparisonPanel } from "@/components/ShadowComparisonPanel";
import { AuthGate } from "@/components/AuthGate";

import { useAuth } from "@/hooks/useAuth";
import {
  useCancelAnalysis,
  useDeleteAnalysis,
  useTriggerVideoAnalysis,
  useVideo,
  useVideoAnalyses,
} from "@/hooks/useVideos";
import { ApiError } from "@/lib/api-client";
import { formatBytes, formatDateTime } from "@/lib/format";

function VideoDetail({ videoId }: { videoId: string }): JSX.Element {
  const { isAuthenticated } = useAuth();
  const [shadowMode, setShadowMode] = useState(false);
  const videoQuery = useVideo(videoId, isAuthenticated);
  const analysesQuery = useVideoAnalyses(videoId, isAuthenticated);
  const triggerAnalysis = useTriggerVideoAnalysis(videoId);
  const cancelAnalysisMutation = useCancelAnalysis(videoId);
  const deleteAnalysisMutation = useDeleteAnalysis(videoId);

  if (videoQuery.isLoading) {
    return <p className="text-sm text-content-dim">Loading...</p>;
  }
  if (videoQuery.isError || !videoQuery.data) {
    return (
      <p role="alert" className="text-sm text-status-alert">
        {videoQuery.error instanceof ApiError
          ? videoQuery.error.message
          : "Video not found."}
      </p>
    );
  }

  const video = videoQuery.data;

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-8">
      <div>
        <h1 className="text-2xl font-bold text-content-default">{video.original_filename}</h1>
        <p className="mt-1 text-sm text-content-dim">
          {video.status} - {formatBytes(video.file_size)} - uploaded{" "}
          {formatDateTime(video.created_at)}
          {video.width && video.height ? ` - ${video.width}x${video.height}` : ""}
          {video.fps ? ` @ ${video.fps.toFixed(1)}fps` : ""}
        </p>
      </div>

      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-content-default">Analysis runs</h2>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-content-dim">
            <input
              type="checkbox"
              checked={shadowMode}
              onChange={(event) => setShadowMode(event.target.checked)}
              className="h-4 w-4 rounded-s"
            />
            Shadow mode
          </label>
          <button
            type="button"
            onClick={() => triggerAnalysis.mutate({ shadow: shadowMode })}
            disabled={triggerAnalysis.isPending || video.status !== "ready"}
            className="rounded-full bg-brand-primary px-4 py-2 text-sm font-semibold text-white disabled:opacity-60"
          >
            {triggerAnalysis.isPending ? "Starting..." : "Run analysis"}
          </button>
        </div>
      </div>
      <p className="text-xs text-content-dim">
        Shadow mode runs the full pipeline and persists real results, but keeps the job flagged
        as non-official (e.g. for trying a new model version without affecting a video&apos;s
        canonical score).
      </p>

      {analysesQuery.isLoading ? (
        <p className="text-sm text-content-dim">Loading...</p>
      ) : (
        <>
        <ShadowComparisonPanel jobs={analysesQuery.data ?? []} enabled={isAuthenticated} />
        <AnalysisJobList
          jobs={analysesQuery.data ?? []}
          onCancel={(analysisId) => cancelAnalysisMutation.mutate(analysisId)}
          onDelete={(analysisId) => deleteAnalysisMutation.mutate(analysisId)}
          cancellingId={
            cancelAnalysisMutation.isPending ? cancelAnalysisMutation.variables : undefined
          }
          deletingId={
            deleteAnalysisMutation.isPending ? deleteAnalysisMutation.variables : undefined
          }
        />
        </>
      )}
    </div>
  );
}

export default function VideoDetailPage(): JSX.Element {
  const params = useParams<{ videoId: string }>();
  return (
    <AuthGate>
      <VideoDetail videoId={params.videoId} />
    </AuthGate>
  );
}
