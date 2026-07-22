"use client";

import Link from "next/link";

import { formatBytes, formatDateTime } from "@/lib/format";
import type { VideoAsset } from "@/lib/types";

import { useDeleteVideo } from "@/hooks/useVideos";

const STATUS_LABELS: Record<VideoAsset["status"], string> = {
  uploaded: "Uploaded",
  validating: "Validating",
  ready: "Ready",
  rejected: "Rejected",
  deleted: "Deleted",
};

export function VideoList({ videos }: { videos: VideoAsset[] }): JSX.Element {
  const deleteVideo = useDeleteVideo();

  if (videos.length === 0) {
    return <p className="text-sm text-content-dim">No videos uploaded yet.</p>;
  }

  return (
    <ul className="flex flex-col gap-3">
      {videos.map((video) => (
        <li
          key={video.id}
          className="flex items-center justify-between rounded-m border border-outline-soft bg-surface-default p-4"
        >
          <div className="flex flex-col gap-1">
            <Link
              href={`/videos/${video.id}`}
              className="font-semibold text-content-default hover:text-brand-boldest"
            >
              {video.original_filename}
            </Link>
            <span className="text-sm text-content-dim">
              {STATUS_LABELS[video.status]} - {formatBytes(video.file_size)} -{" "}
              {formatDateTime(video.created_at)}
            </span>
          </div>
          <button
            type="button"
            onClick={() => {
              if (window.confirm(`Delete "${video.original_filename}"?`)) {
                deleteVideo.mutate({ videoId: video.id });
              }
            }}
            disabled={deleteVideo.isPending}
            className="rounded-full px-4 py-2 text-sm font-semibold text-status-alert hover:bg-status-alert/10 disabled:opacity-60"
          >
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}
