"use client";

import { useRef, useState, type ChangeEvent } from "react";

import { ApiError } from "@/lib/api-client";

import { useUploadVideo } from "@/hooks/useVideos";
import { DIVISIONS, type Division } from "@/lib/types";

/** Accepted per MVP scope ("Uploaded MP4, MOV or WebM video"); the API
 * re-validates MIME type and file signature server-side regardless (never
 * trust the client), this is purely a UX affordance. */
const ACCEPTED_MIME_TYPES = ["video/mp4", "video/quicktime", "video/webm"];

export function VideoUploadForm(): JSX.Element {
  const uploadVideo = useUploadVideo();
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [division, setDivision] = useState<Division>("1A");
  const [playerId, setPlayerId] = useState("");
  const [rightsConfirmed, setRightsConfirmed] = useState(false);

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setError(null);
    try {
      await uploadVideo.mutateAsync({ file, division, playerId, rightsConfirmed });
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
          Upload a freestyle video
        </span>
        <span className="text-sm text-content-dim">
          MP4, MOV, or WebM. Automated analysis is currently available for 1A only.
        </span>
        <select
          aria-label="Competition division"
          value={division}
          onChange={(event) => setDivision(event.target.value as Division)}
          disabled={uploadVideo.isPending}
          className="mt-2 h-10 rounded-s border border-outline-default bg-surface-default px-3 text-sm"
        >
          {DIVISIONS.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
        <input
          type="text"
          required
          value={playerId}
          onChange={(event) => setPlayerId(event.target.value)}
          placeholder="Performer ID, e.g. competitor-042"
          disabled={uploadVideo.isPending}
          className="h-10 rounded-s border border-outline-default px-3 text-sm"
        />
        <span className="flex items-start gap-2 text-sm text-content-dim">
          <input
            type="checkbox"
            checked={rightsConfirmed}
            onChange={(event) => setRightsConfirmed(event.target.checked)}
            className="mt-1"
          />
          I have permission to use this video for model training.
        </span>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_MIME_TYPES.join(",")}
          onChange={handleFileChange}
          disabled={uploadVideo.isPending || !playerId.trim() || !rightsConfirmed}
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
