"use client";

import { useRef, useState, type ChangeEvent } from "react";

import { ApiError } from "@/lib/api-client";

import { useUploadVideo } from "@/hooks/useVideos";

/** Accepted per MVP scope ("Uploaded MP4, MOV or WebM video"); the API
 * re-validates MIME type and file signature server-side regardless (never
 * trust the client), this is purely a UX affordance. */
const ACCEPTED_MIME_TYPES = ["video/mp4", "video/quicktime", "video/webm"];

export function VideoUploadForm(): JSX.Element {
  const uploadVideo = useUploadVideo();
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setError(null);
    try {
      await uploadVideo.mutateAsync(file);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Upload failed. Please check the file and try again."
      );
    } finally {
      if (inputRef.current) {
        inputRef.current.value = "";
      }
    }
  }

  return (
    <div className="flex flex-col gap-2 rounded-m border border-dashed border-outline-default bg-surface-default p-6">
      <label className="flex flex-col gap-2">
        <span className="text-base font-semibold text-content-default">
          Upload a 1A freestyle video
        </span>
        <span className="text-sm text-content-dim">MP4, MOV, or WebM. Analysis runs offline.</span>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_MIME_TYPES.join(",")}
          onChange={handleFileChange}
          disabled={uploadVideo.isPending}
          className="mt-2 text-sm"
        />
      </label>
      {uploadVideo.isPending ? (
        <p className="text-sm text-content-dim">Uploading and validating...</p>
      ) : null}
      {error ? (
        <p role="alert" className="text-sm text-status-alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
