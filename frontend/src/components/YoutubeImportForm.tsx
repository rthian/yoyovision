"use client";

import { useState, type FormEvent } from "react";

import { useImportYoutubeVideo } from "@/hooks/useVideos";
import { ApiError } from "@/lib/api-client";
import { DIVISIONS, type Division } from "@/lib/types";

export function YoutubeImportForm(): JSX.Element {
  const importVideo = useImportYoutubeVideo();
  const [url, setUrl] = useState("");
  const [division, setDivision] = useState<Division>("1A");
  const [playerId, setPlayerId] = useState("");
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    try {
      await importVideo.mutateAsync({
        url,
        division,
        player_id: playerId,
        rights_confirmed: true,
      });
      setUrl("");
      setRightsConfirmed(false);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "YouTube import failed.");
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-3 rounded-m border border-outline-soft bg-surface-default p-5"
    >
      <div>
        <h2 className="font-semibold text-content-default">Import from YouTube</h2>
        <p className="text-sm text-content-dim">
          Import one video for annotation. Playlists and non-YouTube URLs are rejected.
        </p>
      </div>
      <label className="flex flex-col gap-1 text-sm">
        YouTube link
        <input
          type="url"
          required
          value={url}
          placeholder="https://www.youtube.com/watch?v=..."
          onChange={(event) => setUrl(event.target.value)}
          className="h-10 rounded-s border border-outline-default px-3"
        />
      </label>
      <label className="flex flex-col gap-1 text-sm">
        Performer ID
        <input
          required
          value={playerId}
          onChange={(event) => setPlayerId(event.target.value)}
          placeholder="Stable pseudonym, e.g. competitor-042"
          className="h-10 rounded-s border border-outline-default px-3"
        />
      </label>
      <label className="flex flex-col gap-1 text-sm">
        Division
        <select
          value={division}
          onChange={(event) => setDivision(event.target.value as Division)}
          className="h-10 rounded-s border border-outline-default px-3"
        >
          {DIVISIONS.map((value) => (
            <option key={value} value={value}>{value}</option>
          ))}
        </select>
      </label>
      <label className="flex items-start gap-2 text-sm text-content-dim">
        <input
          type="checkbox"
          checked={rightsConfirmed}
          onChange={(event) => setRightsConfirmed(event.target.checked)}
          className="mt-1"
        />
        I have permission to download and use this video for model training.
      </label>
      <button
        type="submit"
        disabled={!url || !playerId.trim() || !rightsConfirmed || importVideo.isPending}
        className="self-start rounded-full bg-brand-default px-5 py-2 text-sm font-semibold text-content-on-brand disabled:opacity-50"
      >
        {importVideo.isPending ? "Importing…" : "Import video"}
      </button>
      {error ? <p role="alert" className="text-sm text-status-alert">{error}</p> : null}
    </form>
  );
}
