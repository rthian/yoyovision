"use client";

import { useMemo, useState } from "react";

import { JudgeFreestyleForm } from "@/components/JudgeFreestyleForm";
import { ApiError } from "@/lib/api-client";
import {
  useJudgeAccess,
  useSubmitJudgeFe,
  useUpsertJudgeFe,
} from "@/hooks/useJudgeAccess";
import { useJudgeVideoBlobUrl } from "@/hooks/useJudgeVideoBlobUrl";

interface JudgePageProps {
  params: { token: string };
}

export default function JudgePage({ params }: JudgePageProps): JSX.Element {
  const token = params.token;
  const accessQuery = useJudgeAccess(token);
  const videos = accessQuery.data?.videos ?? [];
  const [selectedVideoId, setSelectedVideoId] = useState<string | undefined>();

  const activeVideoId = useMemo(() => {
    if (selectedVideoId) {
      return selectedVideoId;
    }
    return videos[0]?.entry_video_id;
  }, [selectedVideoId, videos]);

  const activeVideo = videos.find((video) => video.entry_video_id === activeVideoId);
  const upsert = useUpsertJudgeFe(token, activeVideoId ?? "");
  const submit = useSubmitJudgeFe(token, activeVideoId ?? "");
  const videoBlob = useJudgeVideoBlobUrl(token, activeVideoId, Boolean(activeVideoId));

  if (accessQuery.isLoading) {
    return <p className="text-sm text-content-dim">Loading your judging session…</p>;
  }

  if (accessQuery.isError) {
    const err = accessQuery.error;
    const message =
      err instanceof ApiError
        ? err.message
        : "This invite link is invalid or no longer available.";
    const isExpired = err instanceof ApiError && (err.status === 410 || err.status === 401);
    return (
      <div className="mx-auto max-w-lg rounded-m border border-outline-soft bg-surface-default p-6">
        <h1 className="text-xl font-bold text-content-default">Invite unavailable</h1>
        <p className="mt-2 text-sm text-content-dim">{message}</p>
        {isExpired ? (
          <p className="mt-2 text-sm text-content-subtle">Ask your admin for a new link.</p>
        ) : null}
      </div>
    );
  }

  const data = accessQuery.data!;

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6">
      <header>
        <p className="text-sm text-content-dim">Judging as {data.display_name}</p>
        <h1 className="text-2xl font-bold text-content-default">{data.entry_title}</h1>
        <p className="text-sm text-content-dim">Mode: {data.entry_mode}</p>
      </header>

      <section>
        <h2 className="mb-2 text-sm font-semibold text-content-subtle">Videos</h2>
        <ul className="flex flex-col gap-2">
          {videos.map((video) => {
            const submitted = video.my_score?.is_submitted ?? false;
            const active = video.entry_video_id === activeVideoId;
            return (
              <li key={video.entry_video_id}>
                <button
                  type="button"
                  onClick={() => setSelectedVideoId(video.entry_video_id)}
                  className={`w-full rounded-s border px-4 py-3 text-left text-sm ${
                    active
                      ? "border-brand-bold bg-brand-softest"
                      : "border-outline-soft bg-surface-default"
                  }`}
                >
                  <span className="font-semibold text-content-default">
                    {video.sort_order + 1}. {video.original_filename}
                  </span>
                  <span className="ml-2 text-content-dim">
                    {submitted ? "Submitted" : "Not submitted"}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </section>

      {activeVideo ? (
        <section className="flex flex-col gap-4">
          {videoBlob.isLoading ? (
            <p className="text-sm text-content-dim">Loading video…</p>
          ) : videoBlob.error ? (
            <p role="alert" className="text-sm text-status-alert">
              Could not load video.
            </p>
          ) : videoBlob.blobUrl ? (
            <video controls src={videoBlob.blobUrl} className="w-full rounded-m bg-black" />
          ) : null}

          <JudgeFreestyleForm
            score={activeVideo.my_score}
            readOnly={activeVideo.my_score?.is_submitted ?? false}
            isSaving={upsert.isPending}
            isSubmitting={submit.isPending}
            onSaveDraft={(payload) => upsert.mutate(payload)}
            onSubmit={(payload) => submit.mutate(payload)}
          />
        </section>
      ) : null}
    </div>
  );
}
