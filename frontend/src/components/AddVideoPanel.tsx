"use client";

import { useRef, useState, type FormEvent } from "react";

import { useImportYoutubeVideo, useUploadVideo } from "@/hooks/useVideos";
import { ApiError } from "@/lib/api-client";
import { DIVISIONS, type Division } from "@/lib/types";

const ACCEPTED_MIME_TYPES = ["video/mp4", "video/quicktime", "video/webm"];
type AddMethod = "upload" | "youtube";

export function AddVideoPanel(): JSX.Element {
  const uploadVideo = useUploadVideo();
  const importVideo = useImportYoutubeVideo();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [method, setMethod] = useState<AddMethod>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [division, setDivision] = useState<Division>("1A");
  const [performer, setPerformer] = useState("");
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const isPending = uploadVideo.isPending || importVideo.isPending;
  const hasSource = method === "upload" ? Boolean(file) : Boolean(url.trim());
  const canSubmit = hasSource && Boolean(performer.trim()) && rightsConfirmed && !isPending;

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!canSubmit) return;

    setError(null);
    setSuccess(null);
    try {
      if (method === "upload" && file) {
        await uploadVideo.mutateAsync({
          file,
          division,
          playerId: performer.trim(),
          rightsConfirmed,
        });
        setFile(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
        setSuccess("Video uploaded. Open it below to annotate or run an analysis.");
      } else if (method === "youtube") {
        await importVideo.mutateAsync({
          url: url.trim(),
          division,
          player_id: performer.trim(),
          rights_confirmed: true,
        });
        setUrl("");
        setSuccess("YouTube video imported. Open it below to continue.");
      }
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : method === "youtube"
            ? "YouTube import failed. Retry or upload the video file instead."
            : "Upload failed. Check the video file and try again."
      );
    }
  }

  return (
    <section className="overflow-hidden rounded-m border border-outline-soft bg-surface-default">
      <div className="border-b border-outline-softest px-5 pt-5">
        <h2 className="text-lg font-semibold text-content-default">Add video</h2>
        <p className="mt-1 text-sm text-content-dim">
          Add permitted footage for annotation and AI-assisted analysis.
        </p>
        <div className="mt-4 flex gap-5" role="tablist" aria-label="Video source">
          {(["upload", "youtube"] as const).map((value) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={method === value}
              onClick={() => {
                setMethod(value);
                setError(null);
                setSuccess(null);
              }}
              className={`border-b-2 pb-3 text-sm font-semibold ${
                method === value
                  ? "border-brand-primary text-brand-boldest"
                  : "border-transparent text-content-dim hover:text-content-default"
              }`}
            >
              {value === "upload" ? "Upload file" : "YouTube link"}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4 p-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="flex flex-col gap-1.5 text-sm font-medium text-content-subtle">
            Performer name or ID
            <input
              required
              value={performer}
              onChange={(event) => setPerformer(event.target.value)}
              placeholder="e.g. competitor-042"
              disabled={isPending}
              className="h-10 rounded-s border border-outline-default px-3 font-normal"
            />
          </label>
          <label className="flex flex-col gap-1.5 text-sm font-medium text-content-subtle">
            Division
            <select
              value={division}
              onChange={(event) => setDivision(event.target.value as Division)}
              disabled={isPending}
              className="h-10 rounded-s border border-outline-default bg-surface-default px-3 font-normal"
            >
              {DIVISIONS.map((value) => <option key={value}>{value}</option>)}
            </select>
          </label>
        </div>

        {method === "upload" ? (
          <label className="flex flex-col gap-1.5 text-sm font-medium text-content-subtle">
            Video file
            <input
              ref={fileInputRef}
              type="file"
              required
              accept={ACCEPTED_MIME_TYPES.join(",")}
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              disabled={isPending}
              className="rounded-s border border-dashed border-outline-default bg-surface-alt p-3 font-normal"
            />
            <span className="text-xs font-normal text-content-dim">MP4, MOV, or WebM</span>
          </label>
        ) : (
          <label className="flex flex-col gap-1.5 text-sm font-medium text-content-subtle">
            YouTube link
            <input
              type="url"
              required
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              disabled={isPending}
              className="h-10 rounded-s border border-outline-default px-3 font-normal"
            />
            <span className="text-xs font-normal text-content-dim">
              If YouTube blocks the import, switch to Upload file.
            </span>
          </label>
        )}

        <label className="flex items-start gap-2 text-sm text-content-dim">
          <input
            type="checkbox"
            checked={rightsConfirmed}
            onChange={(event) => setRightsConfirmed(event.target.checked)}
            disabled={isPending}
            className="mt-1"
          />
          I have permission to use this video for model training.
        </label>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded-full bg-brand-primary px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-50"
          >
            {isPending ? (method === "youtube" ? "Importing video…" : "Uploading video…") : "Add video"}
          </button>
          <span className="text-xs text-content-dim">Automated analysis currently supports 1A.</span>
        </div>
        {error ? <p role="alert" className="text-sm text-status-alert">{error}</p> : null}
        {success ? <p role="status" className="text-sm text-status-positive">{success}</p> : null}
      </form>
    </section>
  );
}
