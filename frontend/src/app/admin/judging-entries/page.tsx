"use client";

import Link from "next/link";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { AuthGate } from "@/components/AuthGate";
import {
  createJudgingEntry,
  listJudgingEntries,
  updateJudgingEntry,
} from "@/lib/api-client";
import type { JudgingEntryMode } from "@/lib/types";
import { useVideos } from "@/hooks/useVideos";

function JudgingEntriesAdmin(): JSX.Element {
  const queryClient = useQueryClient();
  const entriesQuery = useQuery({
    queryKey: ["judgingEntries"],
    queryFn: listJudgingEntries,
  });
  const videosQuery = useVideos(true);
  const [title, setTitle] = useState("");
  const [mode, setMode] = useState<JudgingEntryMode>("training");
  const [selectedVideoIds, setSelectedVideoIds] = useState<string[]>([]);

  const createMutation = useMutation({
    mutationFn: () =>
      createJudgingEntry({ title, mode, video_ids: selectedVideoIds }),
    onSuccess: async (entry) => {
      await updateJudgingEntry(entry.id, { status: "open" });
      void queryClient.invalidateQueries({ queryKey: ["judgingEntries"] });
      setTitle("");
      setSelectedVideoIds([]);
    },
  });

  function toggleVideo(videoId: string): void {
    setSelectedVideoIds((prev) =>
      prev.includes(videoId) ? prev.filter((id) => id !== videoId) : [...prev, videoId]
    );
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-8">
      <div>
        <h1 className="text-2xl font-bold text-content-default">Judging entries</h1>
        <p className="mt-1 text-sm text-content-dim">
          Create multi-video panels and share private judge invites.
        </p>
      </div>

      <form
        className="flex flex-col gap-4 rounded-m border border-outline-soft p-4"
        onSubmit={(e) => {
          e.preventDefault();
          createMutation.mutate();
        }}
      >
        <h2 className="text-lg font-semibold">New entry</h2>
        <label className="flex flex-col gap-1 text-sm">
          Title
          <input
            required
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="h-10 rounded-s border border-outline-default px-3"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Mode
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as JudgingEntryMode)}
            className="h-10 rounded-s border border-outline-default px-3"
          >
            <option value="training">Training</option>
            <option value="contest">Contest</option>
          </select>
        </label>
        <fieldset>
          <legend className="mb-2 text-sm font-semibold">Videos</legend>
          <div className="flex max-h-48 flex-col gap-2 overflow-y-auto">
            {(videosQuery.data ?? []).map((video) => (
              <label key={video.id} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={selectedVideoIds.includes(video.id)}
                  onChange={() => toggleVideo(video.id)}
                />
                {video.original_filename}
              </label>
            ))}
          </div>
        </fieldset>
        <button
          type="submit"
          disabled={!title || selectedVideoIds.length === 0 || createMutation.isPending}
          className="self-start rounded-full bg-brand-default px-5 py-2 text-sm font-semibold text-content-on-brand disabled:opacity-50"
        >
          {createMutation.isPending ? "Creating…" : "Create & open entry"}
        </button>
      </form>

      <section>
        <h2 className="mb-3 text-lg font-semibold">Entries</h2>
        {entriesQuery.isLoading ? (
          <p className="text-sm text-content-dim">Loading…</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {(entriesQuery.data ?? []).map((entry) => (
              <li key={entry.id}>
                <Link
                  href={`/admin/judging-entries/${entry.id}`}
                  className="block rounded-s border border-outline-soft px-4 py-3 hover:bg-surface-alt"
                >
                  <span className="font-semibold text-content-default">{entry.title}</span>
                  <span className="ml-2 text-sm text-content-dim">
                    {entry.status} · {entry.judges.length} judges
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

export default function JudgingEntriesPage(): JSX.Element {
  return (
    <AuthGate>
      <JudgingEntriesAdmin />
    </AuthGate>
  );
}
