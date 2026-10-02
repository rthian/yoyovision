"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { AuthGate } from "@/components/AuthGate";
import {
  createJudgingEntry,
  listJudgingEntries,
  updateJudgingEntry,
} from "@/lib/api-client";
import { DIVISIONS, type Division, type JudgingEntryMode } from "@/lib/types";
import { useVideos } from "@/hooks/useVideos";

function JudgingEntriesAdmin(): JSX.Element {
  const router = useRouter();
  const queryClient = useQueryClient();
  const entriesQuery = useQuery({
    queryKey: ["judgingEntries"],
    queryFn: listJudgingEntries,
  });
  const videosQuery = useVideos(true);
  const [title, setTitle] = useState("");
  const [mode, setMode] = useState<JudgingEntryMode>("training");
  const [division, setDivision] = useState<Division>("1A");
  const [videoSearch, setVideoSearch] = useState("");
  const [selectedVideoIds, setSelectedVideoIds] = useState<string[]>([]);
  const availableVideos = (videosQuery.data ?? []).filter(
    (video) => video.division === division
  );
  const searchTerm = videoSearch.trim().toLocaleLowerCase();
  const filteredVideos = availableVideos.filter((video) =>
    video.original_filename.toLocaleLowerCase().includes(searchTerm)
  );

  const createMutation = useMutation({
    mutationFn: () =>
      createJudgingEntry({ title, mode, division, video_ids: selectedVideoIds }),
    onSuccess: async (entry) => {
      await updateJudgingEntry(entry.id, { status: "open" });
      void queryClient.invalidateQueries({ queryKey: ["judgingEntries"] });
      setTitle("");
      setVideoSearch("");
      setSelectedVideoIds([]);
      router.push(`/admin/judging-entries/${entry.id}`);
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
        <h2 className="text-lg font-semibold">Create judging entry</h2>
        <label className="flex flex-col gap-1 text-sm">
          Entry name
          <input
            required
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Example: Mir Kim · 1A Final"
            className="h-10 rounded-s border border-outline-default px-3"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Division
          <select
            value={division}
            onChange={(e) => {
              setDivision(e.target.value as Division);
              setVideoSearch("");
              setSelectedVideoIds([]);
            }}
            className="h-10 rounded-s border border-outline-default px-3"
          >
            {DIVISIONS.map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
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
        <fieldset className="rounded-s border border-outline-soft p-3">
          <legend className="px-1 text-sm font-semibold">Videos</legend>
          <div className="flex justify-end">
            <span className="rounded-full bg-brand-softest px-2.5 py-1 text-xs font-semibold text-brand-boldest">
              {selectedVideoIds.length} selected
            </span>
          </div>
          <label className="mt-3 block text-sm">
            <span className="sr-only">Search videos</span>
            <input
              type="search"
              value={videoSearch}
              onChange={(event) => setVideoSearch(event.target.value)}
              placeholder={`Search ${availableVideos.length} ${division} videos by filename`}
              className="h-10 w-full rounded-s border border-outline-default px-3"
            />
          </label>
          <div className="mt-2 flex items-center justify-between text-xs text-content-dim">
            <span>
              Showing {filteredVideos.length} of {availableVideos.length}
            </span>
            {selectedVideoIds.length > 0 ? (
              <button
                type="button"
                onClick={() => setSelectedVideoIds([])}
                className="rounded-full border border-outline-default px-3 py-1.5 font-semibold text-content-subtle hover:bg-surface-alt"
              >
                Clear selection
              </button>
            ) : null}
          </div>
          <div className="mt-2 flex max-h-72 flex-col gap-2 overflow-y-auto pr-1">
            {videosQuery.isLoading ? (
              <p className="rounded-s bg-surface-alt p-3 text-sm text-content-dim">
                Loading videos…
              </p>
            ) : null}
            {filteredVideos.map((video) => (
              <label
                key={video.id}
                className={`flex cursor-pointer items-start gap-3 rounded-s border px-3 py-2.5 text-sm transition hover:bg-surface-alt ${
                  selectedVideoIds.includes(video.id)
                    ? "border-brand-default bg-brand-softest"
                    : "border-outline-soft"
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedVideoIds.includes(video.id)}
                  onChange={() => toggleVideo(video.id)}
                  className="mt-0.5 h-4 w-4 shrink-0"
                />
                <span className="min-w-0 break-words">{video.original_filename}</span>
              </label>
            ))}
            {!videosQuery.isLoading && filteredVideos.length === 0 ? (
              <p className="rounded-s bg-surface-alt p-3 text-sm text-content-dim">
                {availableVideos.length === 0
                  ? `No ${division} videos are available. Upload a video first.`
                  : "No videos match this search."}
              </p>
            ) : null}
          </div>
        </fieldset>
        <button
          type="submit"
          disabled={!title.trim() || selectedVideoIds.length === 0 || createMutation.isPending}
          className="self-start rounded-full bg-brand-default px-5 py-2 text-sm font-semibold text-content-on-brand disabled:opacity-50"
        >
          {createMutation.isPending ? "Creating…" : "Create & open entry"}
        </button>
      </form>

      <section>
        <h2 className="mb-3 text-lg font-semibold">Entries</h2>
        {entriesQuery.isLoading ? (
          <p className="text-sm text-content-dim">Loading…</p>
        ) : (entriesQuery.data ?? []).length === 0 ? (
          <p className="rounded-s border border-dashed border-outline-default p-4 text-sm text-content-dim">
            No judging entries yet. Create the first entry above.
          </p>
        ) : (
          <ul className="flex flex-col gap-3">
            {(entriesQuery.data ?? []).map((entry) => (
              <li
                key={entry.id}
                className="flex flex-wrap items-center justify-between gap-3 rounded-s border border-outline-soft px-4 py-3"
              >
                <div>
                  <p className="font-semibold text-content-default">{entry.title}</p>
                  <p className="text-sm text-content-dim">
                    {entry.division} · {entry.status} · {entry.videos.length} videos ·{" "}
                    {entry.judges.length} judges
                  </p>
                </div>
                <Link
                  href={`/admin/judging-entries/${entry.id}`}
                  className="rounded-full bg-brand-default px-4 py-2 text-sm font-semibold text-content-on-brand hover:opacity-90"
                >
                  Open entry
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
