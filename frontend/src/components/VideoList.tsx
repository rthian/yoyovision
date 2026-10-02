"use client";

import Link from "next/link";

import { formatBytes, formatDateTime, formatMsAsTimecode } from "@/lib/format";
import type { VideoAsset } from "@/lib/types";

import { useDeleteVideo } from "@/hooks/useVideos";

const STATUS_LABELS: Record<VideoAsset["status"], string> = {
  uploaded: "Uploaded",
  validating: "Validating",
  ready: "Ready",
  rejected: "Rejected",
  deleted: "Deleted",
};

const STATUS_STYLES: Record<VideoAsset["status"], string> = {
  uploaded: "bg-status-informative/10 text-status-informative",
  validating: "bg-status-notice/15 text-status-notice",
  ready: "bg-status-positive/15 text-status-positive",
  rejected: "bg-status-alert/10 text-status-alert",
  deleted: "bg-surface-alt text-content-dim",
};

export function VideoList({ videos }: { videos: VideoAsset[] }): JSX.Element {
  const deleteVideo = useDeleteVideo();

  if (videos.length === 0) {
    return (
      <div className="rounded-m border border-dashed border-outline-default bg-surface-default p-8 text-center">
        <p className="font-semibold text-content-default">Your video library is empty</p>
        <p className="mt-1 text-sm text-content-dim">Add a video above to start annotating tricks.</p>
      </div>
    );
  }

  return (
    <ul className="flex flex-col gap-3">
      {videos.map((video) => (
        <li
          key={video.id}
          className="group flex items-center gap-3 rounded-m border border-outline-soft bg-surface-default p-3 transition hover:border-outline-default hover:shadow-sm"
        >
          <Link
            href={`/videos/${video.id}`}
            aria-hidden="true"
            tabIndex={-1}
            className="flex h-16 w-24 shrink-0 items-center justify-center rounded-s bg-content-default text-xs font-bold tracking-wide text-white"
          >
            {video.division}
          </Link>
          <div className="min-w-0 flex-1">
            <Link
              href={`/videos/${video.id}`}
              className="block truncate font-semibold text-content-default hover:text-brand-boldest"
            >
              {video.original_filename}
            </Link>
            <p className="mt-1 truncate text-sm text-content-dim">
              {video.player_id || "Unknown performer"} · {video.source_type === "youtube" ? "YouTube" : "File upload"}
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-content-dim">
              <span className={`rounded-full px-2 py-0.5 font-semibold ${STATUS_STYLES[video.status]}`}>
                {STATUS_LABELS[video.status]}
              </span>
              {video.duration_ms ? <span>{formatMsAsTimecode(video.duration_ms)}</span> : null}
              <span>{formatBytes(video.file_size)}</span>
              <span>{formatDateTime(video.created_at)}</span>
            </div>
          </div>
          <details className="relative shrink-0">
            <summary
              aria-label={`More actions for ${video.original_filename}`}
              className="flex h-11 w-11 cursor-pointer list-none items-center justify-center rounded-full text-xl text-content-dim hover:bg-surface-alt"
            >
              ⋯
            </summary>
            <div className="absolute right-0 top-12 z-10 w-36 rounded-m border border-outline-soft bg-surface-default p-2 shadow-lg">
              <button
                type="button"
                onClick={() => {
                  if (window.confirm(`Delete "${video.original_filename}"?`)) {
                    deleteVideo.mutate({ videoId: video.id });
                  }
                }}
                disabled={deleteVideo.isPending}
                className="w-full rounded-s px-3 py-2 text-left text-sm font-semibold text-status-alert hover:bg-status-alert/10 disabled:opacity-60"
              >
                Delete video
              </button>
            </div>
          </details>
        </li>
      ))}
    </ul>
  );
}
