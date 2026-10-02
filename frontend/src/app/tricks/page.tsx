"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { AuthGate } from "@/components/AuthGate";
import { VideoPlayerWithOverlay } from "@/components/VideoPlayerWithOverlay";
import { useAuth } from "@/hooks/useAuth";
import { useVideoBlobUrl } from "@/hooks/useVideoBlobUrl";
import { useVideos } from "@/hooks/useVideos";
import {
  ApiError,
  createTrick,
  createTrickExample,
  deleteTrick,
  deleteTrickExample,
  listTricks,
} from "@/lib/api-client";
import { titleCaseFromSnakeCase } from "@/lib/format";
import {
  DIVISIONS,
  TRICK_VIEW_TYPES,
  type Division,
  type TrickViewType,
} from "@/lib/types";

const trickLibraryKey = ["trickLibrary"] as const;

function formatTime(milliseconds: number): string {
  const totalSeconds = milliseconds / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  return `${minutes}:${(totalSeconds % 60).toFixed(2).padStart(5, "0")}`;
}

function TrickLibrary(): JSX.Element {
  const { isAuthenticated } = useAuth();
  const queryClient = useQueryClient();
  const searchParams = useSearchParams();
  const initialDivision = searchParams.get("division");
  const initialVideoId = searchParams.get("videoId") ?? "";
  const [division, setDivision] = useState<Division>(
    DIVISIONS.includes(initialDivision as Division) ? (initialDivision as Division) : "1A"
  );
  const [search, setSearch] = useState("");
  const [name, setName] = useState("");
  const [aliases, setAliases] = useState("");
  const [description, setDescription] = useState("");
  const [selectedTrickId, setSelectedTrickId] = useState("");
  const [videoId, setVideoId] = useState(initialVideoId);
  const [startSeconds, setStartSeconds] = useState(0);
  const [endSeconds, setEndSeconds] = useState(5);
  const [currentMs, setCurrentMs] = useState(0);
  const [viewType, setViewType] = useState<TrickViewType>("tutorial");
  const [cameraAngle, setCameraAngle] = useState("");
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [exampleNotes, setExampleNotes] = useState("");
  const [isPrimary, setIsPrimary] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tricksQuery = useQuery({
    queryKey: [...trickLibraryKey, division],
    queryFn: () => listTricks({ division }),
    enabled: isAuthenticated,
  });
  const videosQuery = useVideos(isAuthenticated);
  const divisionVideos = useMemo(
    () => (videosQuery.data ?? []).filter((video) => video.division === division),
    [division, videosQuery.data]
  );
  const selectedVideo = divisionVideos.find((video) => video.id === videoId);
  const videoBlob = useVideoBlobUrl(videoId || undefined, Boolean(videoId));
  const filteredTricks = useMemo(() => {
    const term = search.trim().toLocaleLowerCase();
    if (!term) return tricksQuery.data ?? [];
    return (tricksQuery.data ?? []).filter(
      (trick) =>
        trick.name.toLocaleLowerCase().includes(term) ||
        trick.aliases.some((alias) => alias.toLocaleLowerCase().includes(term))
    );
  }, [search, tricksQuery.data]);

  useEffect(() => {
    if (videoId && !divisionVideos.some((video) => video.id === videoId)) setVideoId("");
  }, [divisionVideos, videoId]);

  async function refresh(): Promise<void> {
    await queryClient.invalidateQueries({ queryKey: trickLibraryKey });
  }

  const createMutation = useMutation({
    mutationFn: createTrick,
    onSuccess: async (trick) => {
      await refresh();
      setSelectedTrickId(trick.id);
      setName("");
      setAliases("");
      setDescription("");
    },
  });
  const addExampleMutation = useMutation({
    mutationFn: () =>
      createTrickExample(selectedTrickId, {
        video_id: videoId,
        start_ms: Math.round(startSeconds * 1000),
        end_ms: Math.round(endSeconds * 1000),
        view_type: viewType,
        camera_angle: cameraAngle,
        playback_speed: playbackSpeed,
        notes: exampleNotes,
        is_primary: isPrimary,
      }),
    onSuccess: async () => {
      await refresh();
      setCameraAngle("");
      setExampleNotes("");
      setIsPrimary(false);
    },
  });
  const deleteTrickMutation = useMutation({
    mutationFn: deleteTrick,
    onSuccess: refresh,
  });
  const deleteExampleMutation = useMutation({
    mutationFn: ({ trickId, exampleId }: { trickId: string; exampleId: string }) =>
      deleteTrickExample(trickId, exampleId),
    onSuccess: refresh,
  });

  async function submitTrick(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    try {
      await createMutation.mutateAsync({
        name,
        division,
        aliases: aliases.split(",").map((value) => value.trim()).filter(Boolean),
        description,
      });
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "Could not create trick.");
    }
  }

  async function submitExample(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    try {
      await addExampleMutation.mutateAsync();
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "Could not add example.");
    }
  }

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-8">
      <div>
        <h1 className="text-2xl font-bold text-content-default">Trick Library</h1>
        <p className="mt-1 text-sm text-content-dim">
          Group tutorial, alternate-angle, slow-motion, and stage segments under one
          canonical trick. These labels prepare cross-view examples for future matching.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <form onSubmit={submitTrick} className="flex flex-col gap-4 rounded-m border border-outline-soft p-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-content-dim">Step 1</p>
            <h2 className="text-lg font-semibold">Create a catalog trick</h2>
          </div>
          <label className="flex flex-col gap-1 text-sm">
            Division
            <select
              value={division}
              onChange={(event) => {
                setDivision(event.target.value as Division);
                setSelectedTrickId("");
              }}
              className="h-10 rounded-s border border-outline-default px-3"
            >
              {DIVISIONS.map((value) => <option key={value}>{value}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Canonical trick name
            <input required value={name} onChange={(event) => setName(event.target.value)} className="h-10 rounded-s border border-outline-default px-3" placeholder="Mir Kim Over the Head" />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Aliases <span className="text-xs text-content-dim">Separate names with commas.</span>
            <input value={aliases} onChange={(event) => setAliases(event.target.value)} className="h-10 rounded-s border border-outline-default px-3" />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Description
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} className="min-h-24 rounded-s border border-outline-default p-3" />
          </label>
          <button type="submit" disabled={!name.trim() || createMutation.isPending} className="self-start rounded-full bg-brand-default px-5 py-2 text-sm font-semibold text-content-on-brand disabled:opacity-50">
            {createMutation.isPending ? "Creating…" : "Create trick"}
          </button>
        </form>

        <form onSubmit={submitExample} className="flex flex-col gap-4 rounded-m border border-outline-soft p-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-content-dim">Step 2</p>
            <h2 className="text-lg font-semibold">Add a video example</h2>
          </div>
          <label className="flex flex-col gap-1 text-sm">
            Catalog trick
            <select required value={selectedTrickId} onChange={(event) => setSelectedTrickId(event.target.value)} className="h-10 rounded-s border border-outline-default px-3">
              <option value="">Choose a trick</option>
              {(tricksQuery.data ?? []).map((trick) => <option key={trick.id} value={trick.id}>{trick.name}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Source video
            <select required value={videoId} onChange={(event) => setVideoId(event.target.value)} className="h-10 rounded-s border border-outline-default px-3">
              <option value="">Choose an uploaded or imported video</option>
              {divisionVideos.map((video) => <option key={video.id} value={video.id}>{video.original_filename}</option>)}
            </select>
          </label>
          <p className="text-xs text-content-dim">
            Need another source? <Link href="/" className="font-semibold text-brand-boldest underline">Upload or import a YouTube video</Link>, then return here.
          </p>
          {videoBlob.isLoading ? <p className="text-sm text-content-dim">Loading source video…</p> : null}
          {videoBlob.blobUrl ? (
            <div className="flex flex-col gap-2">
              <VideoPlayerWithOverlay
                src={videoBlob.blobUrl}
                events={[]}
                onTimeUpdateMs={setCurrentMs}
              />
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-s bg-surface-alt p-2 text-xs text-content-dim">
                <span>Playhead: {formatTime(currentMs)}</span>
                <div className="flex gap-2">
                  <button type="button" onClick={() => setStartSeconds(currentMs / 1000)} className="rounded-full border border-outline-default px-3 py-1.5 font-semibold text-content-subtle">Mark start</button>
                  <button type="button" onClick={() => setEndSeconds(currentMs / 1000)} className="rounded-full bg-brand-primary px-3 py-1.5 font-semibold text-white">Mark end</button>
                </div>
              </div>
              {selectedVideo?.duration_ms ? <p className="text-xs text-content-dim">Source duration: {formatTime(selectedVideo.duration_ms)}</p> : null}
            </div>
          ) : null}
          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1 text-sm">Start (seconds)<input type="number" min={0} step="0.01" value={startSeconds} onChange={(event) => setStartSeconds(Number(event.target.value))} className="h-10 rounded-s border border-outline-default px-3" /></label>
            <label className="flex flex-col gap-1 text-sm">End (seconds)<input type="number" min={0} step="0.01" value={endSeconds} onChange={(event) => setEndSeconds(Number(event.target.value))} className="h-10 rounded-s border border-outline-default px-3" /></label>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1 text-sm">View type<select value={viewType} onChange={(event) => setViewType(event.target.value as TrickViewType)} className="h-10 rounded-s border border-outline-default px-3">{TRICK_VIEW_TYPES.map((value) => <option key={value} value={value}>{titleCaseFromSnakeCase(value)}</option>)}</select></label>
            <label className="flex flex-col gap-1 text-sm">Playback speed<input type="number" min={0.1} max={4} step={0.1} value={playbackSpeed} onChange={(event) => setPlaybackSpeed(Number(event.target.value))} className="h-10 rounded-s border border-outline-default px-3" /></label>
          </div>
          <label className="flex flex-col gap-1 text-sm">Camera angle<input value={cameraAngle} onChange={(event) => setCameraAngle(event.target.value)} placeholder="Front, side, overhead…" className="h-10 rounded-s border border-outline-default px-3" /></label>
          <label className="flex flex-col gap-1 text-sm">Notes<textarea value={exampleNotes} onChange={(event) => setExampleNotes(event.target.value)} className="min-h-20 rounded-s border border-outline-default p-3" /></label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={isPrimary} onChange={(event) => setIsPrimary(event.target.checked)} />Use as the primary reference view</label>
          <button type="submit" disabled={!selectedTrickId || !videoId || endSeconds <= startSeconds || addExampleMutation.isPending} className="self-start rounded-full bg-brand-primary px-5 py-2 text-sm font-semibold text-white disabled:opacity-50">
            {addExampleMutation.isPending ? "Adding…" : "Add example"}
          </button>
        </form>
      </div>

      {error ? <p role="alert" className="rounded-s bg-status-alert/10 p-3 text-sm text-status-alert">{error}</p> : null}

      <section className="flex flex-col gap-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div><h2 className="text-lg font-semibold">Catalog</h2><p className="text-sm text-content-dim">{filteredTricks.length} tricks in {division}</p></div>
          <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search names or aliases" className="h-10 rounded-s border border-outline-default px-3 sm:w-72" />
        </div>
        {tricksQuery.isLoading ? <p className="text-sm text-content-dim">Loading library…</p> : null}
        {filteredTricks.map((trick) => (
          <article key={trick.id} className="rounded-m border border-outline-soft p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2"><h3 className="font-semibold text-content-default">{trick.name}</h3><span className="rounded-full bg-brand-softest px-2 py-0.5 text-xs font-semibold text-brand-boldest">{trick.division}</span></div>
                {trick.aliases.length ? <p className="mt-1 text-sm text-content-dim">Also: {trick.aliases.join(", ")}</p> : null}
                {trick.description ? <p className="mt-2 text-sm text-content-subtle">{trick.description}</p> : null}
              </div>
              <button type="button" onClick={() => { if (window.confirm(`Delete ${trick.name} and its catalog examples? Source videos and annotations will be kept.`)) deleteTrickMutation.mutate(trick.id); }} className="rounded-full border border-status-alert/40 px-3 py-1.5 text-xs font-semibold text-status-alert hover:bg-status-alert/5">Delete</button>
            </div>
            <div className="mt-4 grid gap-2 md:grid-cols-2">
              {trick.examples.map((example) => (
                <div key={example.id} className="rounded-s bg-surface-alt p-3 text-sm">
                  <div className="flex items-start justify-between gap-3">
                    <div><p className="font-semibold">{titleCaseFromSnakeCase(example.view_type)}{example.is_primary ? " · Primary" : ""}</p><p className="break-all text-xs text-content-dim">{example.original_filename}</p></div>
                    <button type="button" onClick={() => deleteExampleMutation.mutate({ trickId: trick.id, exampleId: example.id })} className="text-xs font-semibold text-status-alert">Remove</button>
                  </div>
                  <p className="mt-2 tabular-nums text-content-subtle">{formatTime(example.start_ms)}–{formatTime(example.end_ms)} · {example.playback_speed}×{example.camera_angle ? ` · ${example.camera_angle}` : ""}</p>
                  {example.notes ? <p className="mt-1 text-xs text-content-dim">{example.notes}</p> : null}
                  <Link href={`/videos/${example.video_id}`} className="mt-2 inline-block text-xs font-semibold text-brand-boldest underline">Open video and annotations</Link>
                </div>
              ))}
              {trick.examples.length === 0 ? <p className="text-sm text-content-dim">No examples yet. Select this trick in Step 2.</p> : null}
            </div>
          </article>
        ))}
        {!tricksQuery.isLoading && filteredTricks.length === 0 ? <p className="rounded-s bg-surface-alt p-4 text-sm text-content-dim">No catalog tricks match this view.</p> : null}
      </section>
    </div>
  );
}

export default function TrickLibraryPage(): JSX.Element {
  return <AuthGate><TrickLibrary /></AuthGate>;
}
