"use client";

import { useState, type FormEvent } from "react";

import { VideoPlayerWithOverlay } from "@/components/VideoPlayerWithOverlay";
import {
  useCreateTrainingAnnotation,
  useDeleteTrainingAnnotation,
  useTrainingAnnotations,
  useUpdateTrainingAnnotation,
} from "@/hooks/useTrainingAnnotations";
import { useVideoBlobUrl } from "@/hooks/useVideoBlobUrl";
import { ApiError, getTrainingRecord } from "@/lib/api-client";
import { EVENT_FAMILIES, type Outcome, type TechnicalCredit } from "@/lib/types";
import { titleCaseFromSnakeCase } from "@/lib/format";

interface TrainingAnnotationPanelProps {
  videoId: string;
  durationMs?: number;
}

function formatTimestamp(milliseconds: number): string {
  return `${(milliseconds / 1000).toFixed(2)}s`;
}

const EXTRA_TRICK_CATEGORIES = [
  ["loop_combo", "Loop combo"],
  ["wrap", "Wrap"],
  ["tangler", "Tangler"],
  ["offstring_catch", "Offstring catch"],
  ["counterweight_release", "Counterweight release"],
] as const;

export function TrainingAnnotationPanel({
  videoId,
  durationMs = 0,
}: TrainingAnnotationPanelProps): JSX.Element {
  const videoBlob = useVideoBlobUrl(videoId, true);
  const annotations = useTrainingAnnotations(videoId, true);
  const createAnnotation = useCreateTrainingAnnotation(videoId);
  const updateAnnotation = useUpdateTrainingAnnotation(videoId);
  const deleteAnnotation = useDeleteTrainingAnnotation(videoId);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [currentMs, setCurrentMs] = useState(0);
  const [seekToMs, setSeekToMs] = useState<number | null>(null);
  const [seekRequestId, setSeekRequestId] = useState(0);
  const [label, setLabel] = useState("");
  const [elementType, setElementType] = useState("");
  const [startMs, setStartMs] = useState(0);
  const [endMs, setEndMs] = useState(1000);
  const [outcome, setOutcome] = useState<Outcome>("success");
  const [technicalCredit, setTechnicalCredit] =
    useState<TechnicalCredit>("uncertain");
  const [notes, setNotes] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  function seekTo(milliseconds: number): void {
    setSeekToMs(milliseconds);
    setSeekRequestId((value) => value + 1);
  }

  function updateStart(milliseconds: number): void {
    const nextStart = Math.max(0, Math.round(milliseconds));
    setStartMs(nextStart);
    if (endMs <= nextStart) setEndMs(Math.min(durationMs || nextStart + 1000, nextStart + 1000));
  }

  function updateEnd(milliseconds: number): void {
    setEndMs(Math.max(0, Math.round(milliseconds)));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    try {
      const payload = {
        label,
        element_type: elementType,
        start_ms: startMs,
        end_ms: endMs,
        outcome,
        technical_credit: technicalCredit,
        notes,
      };
      if (editingId) {
        await updateAnnotation.mutateAsync({ annotationId: editingId, payload });
      } else {
        await createAnnotation.mutateAsync(payload);
      }
      setEditingId(null);
      setLabel("");
      setElementType("");
      setNotes("");
      setStartMs(endMs);
      setEndMs(durationMs > 0 ? Math.min(durationMs, endMs + 1000) : endMs + 1000);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save annotation.");
    }
  }

  async function exportRecord(): Promise<void> {
    setIsExporting(true);
    setError(null);
    try {
      const record = await getTrainingRecord(videoId);
      const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${videoId}-training-record.json`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not export training data.");
    } finally {
      setIsExporting(false);
    }
  }

  return (
    <section className="flex flex-col gap-5 rounded-m border border-outline-soft p-4">
      <div>
        <h2 className="text-lg font-semibold text-content-default">Training annotations</h2>
        <p className="text-sm text-content-dim">
          Play or scrub to the beginning of a trick and mark Start, then move to the end and
          mark End. This labels a segment for training; it does not cut the original video.
        </p>
      </div>

      {videoBlob.isLoading ? (
        <p className="text-sm text-content-dim">Loading video…</p>
      ) : videoBlob.blobUrl ? (
        <VideoPlayerWithOverlay
          src={videoBlob.blobUrl}
          events={[]}
          onTimeUpdateMs={setCurrentMs}
          seekToMs={seekToMs}
          seekRequestId={seekRequestId}
        />
      ) : (
        <p className="text-sm text-status-alert">Could not load the video.</p>
      )}

      <div className="rounded-s border border-status-informative/25 bg-status-informative/5 p-3">
        <p className="text-sm font-semibold text-content-default">
          Current playhead: {formatTimestamp(currentMs)}
        </p>
        <ol className="mt-1 list-decimal space-y-1 pl-5 text-sm text-content-dim">
          <li>Pause at the first frame of the trick and choose <strong>Mark start</strong>.</li>
          <li>Move to the final frame and choose <strong>Mark end</strong>.</li>
          <li>Name the trick, choose its category and scoring result, then save.</li>
        </ol>
      </div>

      <form onSubmit={handleSubmit} className="grid gap-4 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm">
          Trick name
          <input
            required
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="Alternating loop combo"
            className="h-10 rounded-s border border-outline-default px-3"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Trick category
          <select
            required
            value={elementType}
            onChange={(event) => setElementType(event.target.value)}
            className="h-10 rounded-s border border-outline-default px-3"
          >
            <option value="">Choose a category</option>
            {elementType &&
            !EXTRA_TRICK_CATEGORIES.some(([value]) => value === elementType) &&
            !EVENT_FAMILIES.some((value) => value === elementType) ? (
              <option value={elementType}>{titleCaseFromSnakeCase(elementType)}</option>
            ) : null}
            {EXTRA_TRICK_CATEGORIES.map(([value, categoryLabel]) => (
              <option key={value} value={value}>{categoryLabel}</option>
            ))}
            {EVENT_FAMILIES.map((family) => (
              <option key={family} value={family}>{titleCaseFromSnakeCase(family)}</option>
            ))}
          </select>
          <span className="text-xs text-content-dim">
            The reusable family of movement. For example, “Black Hop” is the trick name and
            “Hop” is its category.
          </span>
        </label>
        <div className="flex flex-col gap-2 rounded-s border border-outline-soft bg-surface-alt p-3 text-sm">
          <div className="flex items-center justify-between gap-3">
            <span className="font-semibold">Start</span>
            <input
              aria-label="Trick start time in seconds"
              type="number"
              min={0}
              max={durationMs > 0 ? durationMs / 1000 : undefined}
              step="0.01"
              value={(startMs / 1000).toFixed(2)}
              onChange={(event) => updateStart(Number(event.target.value) * 1000)}
              className="h-9 w-24 rounded-s border border-outline-default bg-surface-default px-2 text-right"
            />
          </div>
          <button
            type="button"
            onClick={() => updateStart(currentMs)}
            className="h-10 rounded-full bg-brand-primary px-3 font-semibold text-white"
          >
            Mark start at {formatTimestamp(currentMs)}
          </button>
        </div>
        <div className="flex flex-col gap-2 rounded-s border border-outline-soft bg-surface-alt p-3 text-sm">
          <div className="flex items-center justify-between gap-3">
            <span className="font-semibold">End</span>
            <input
              aria-label="Trick end time in seconds"
              type="number"
              min={0}
              max={durationMs > 0 ? durationMs / 1000 : undefined}
              step="0.01"
              value={(endMs / 1000).toFixed(2)}
              onChange={(event) => updateEnd(Number(event.target.value) * 1000)}
              className="h-9 w-24 rounded-s border border-outline-default bg-surface-default px-2 text-right"
            />
          </div>
          <button
            type="button"
            onClick={() => updateEnd(currentMs)}
            className="h-10 rounded-full bg-brand-primary px-3 font-semibold text-white"
          >
            Mark end at {formatTimestamp(currentMs)}
          </button>
        </div>
        {durationMs > 0 ? (
          <div className="sm:col-span-2">
            <div className="relative h-3 overflow-hidden rounded-full bg-outline-softest" aria-label="Selected trick segment">
              <div
                className="absolute h-full rounded-full bg-status-informative"
                style={{
                  left: `${Math.min(100, (startMs / durationMs) * 100)}%`,
                  width: `${Math.max(0, Math.min(100, ((endMs - startMs) / durationMs) * 100))}%`,
                }}
              />
            </div>
            <div className="mt-2 flex items-center justify-between text-xs text-content-dim">
              <span>Selected: {formatTimestamp(startMs)}–{formatTimestamp(endMs)}</span>
              <button type="button" onClick={() => seekTo(startMs)} className="font-semibold text-brand-boldest hover:underline">
                Preview from start
              </button>
            </div>
            <div className="mt-3 grid gap-2 rounded-s bg-surface-alt p-3 text-xs text-content-dim">
              <label className="grid grid-cols-[80px_1fr_54px] items-center gap-2">
                Adjust start
                <input
                  type="range"
                  min={0}
                  max={durationMs}
                  step={10}
                  value={Math.min(startMs, durationMs)}
                  onChange={(event) => updateStart(Number(event.target.value))}
                  className="accent-status-informative"
                />
                <span className="text-right">{formatTimestamp(startMs)}</span>
              </label>
              <label className="grid grid-cols-[80px_1fr_54px] items-center gap-2">
                Adjust end
                <input
                  type="range"
                  min={0}
                  max={durationMs}
                  step={10}
                  value={Math.min(endMs, durationMs)}
                  onChange={(event) => updateEnd(Number(event.target.value))}
                  className="accent-status-informative"
                />
                <span className="text-right">{formatTimestamp(endMs)}</span>
              </label>
            </div>
          </div>
        ) : null}
        <label className="flex flex-col gap-1 text-sm">
          Outcome
          <select
            value={outcome}
            onChange={(event) => setOutcome(event.target.value as Outcome)}
            className="h-10 rounded-s border border-outline-default px-3"
          >
            <option value="success">Success</option>
            <option value="miss">Miss</option>
            <option value="uncertain">Uncertain</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Scoring label
          <select
            value={technicalCredit}
            onChange={(event) =>
              setTechnicalCredit(event.target.value as TechnicalCredit)
            }
            className="h-10 rounded-s border border-outline-default px-3"
          >
            <option value="positive_click">Award technical point</option>
            <option value="negative_click">Record a miss</option>
            <option value="no_click">No technical point</option>
            <option value="uncertain">Needs review</option>
          </select>
          <span className="text-xs text-content-dim">Choose how a technical judge should treat this segment.</span>
        </label>
        <label className="flex flex-col gap-1 text-sm sm:col-span-2">
          Notes
          <input
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            className="h-10 rounded-s border border-outline-default px-3"
          />
        </label>
        <button
          type="submit"
          disabled={
            !label ||
            !elementType ||
            endMs <= startMs ||
            createAnnotation.isPending ||
            updateAnnotation.isPending
          }
          className="self-start rounded-full bg-brand-default px-5 py-2 text-sm font-semibold text-content-on-brand disabled:opacity-50"
        >
          {createAnnotation.isPending || updateAnnotation.isPending
            ? "Saving…"
            : editingId
              ? "Save changes"
              : "Add annotation"}
        </button>
        {endMs <= startMs ? (
          <p role="alert" className="self-center text-sm text-status-alert">
            End time must be after start time.
          </p>
        ) : null}
        {editingId ? (
          <button
            type="button"
            onClick={() => setEditingId(null)}
            className="self-start px-5 py-2 text-sm font-semibold text-content-dim"
          >
            Cancel edit
          </button>
        ) : null}
      </form>

      {error ? <p role="alert" className="text-sm text-status-alert">{error}</p> : null}

      <div className="flex flex-col gap-2">
        {(annotations.data ?? []).map((annotation) => (
          <div
            key={annotation.id}
            onClick={(event) => {
              if (!(event.target as HTMLElement).closest("button")) seekTo(annotation.start_ms);
            }}
            title={`Go to ${formatTimestamp(annotation.start_ms)}`}
            className="flex cursor-pointer items-center justify-between rounded-s border border-outline-soft p-3 hover:bg-surface-alt"
          >
            <div>
              <p className="font-semibold text-content-default">{annotation.label}</p>
              <p className="text-xs text-content-dim">
                {formatTimestamp(annotation.start_ms)}–{formatTimestamp(annotation.end_ms)} ·{" "}
                {titleCaseFromSnakeCase(annotation.element_type)} · {annotation.outcome} ·{" "}
                {annotation.technical_credit.replaceAll("_", " ")}
              </p>
            </div>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => {
                  setEditingId(annotation.id);
                  setLabel(annotation.label);
                  setElementType(annotation.element_type);
                  setStartMs(annotation.start_ms);
                  setEndMs(annotation.end_ms);
                  setOutcome(annotation.outcome);
                  setTechnicalCredit(annotation.technical_credit);
                  setNotes(annotation.notes);
                }}
                className="text-sm font-semibold text-brand-boldest"
              >
                Edit
              </button>
              <button
                type="button"
                onClick={() => deleteAnnotation.mutate(annotation.id)}
                className="text-sm font-semibold text-status-alert"
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={() => void exportRecord()}
        disabled={(annotations.data?.length ?? 0) === 0 || isExporting}
        className="self-start rounded-full border border-outline-default px-5 py-2 text-sm font-semibold disabled:opacity-50"
      >
        {isExporting ? "Exporting…" : "Export training record"}
      </button>
    </section>
  );
}
