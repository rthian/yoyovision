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
import type { Outcome, TechnicalCredit } from "@/lib/types";

interface TrainingAnnotationPanelProps {
  videoId: string;
}

function formatTimestamp(milliseconds: number): string {
  return `${(milliseconds / 1000).toFixed(2)}s`;
}

export function TrainingAnnotationPanel({
  videoId,
}: TrainingAnnotationPanelProps): JSX.Element {
  const videoBlob = useVideoBlobUrl(videoId, true);
  const annotations = useTrainingAnnotations(videoId, true);
  const createAnnotation = useCreateTrainingAnnotation(videoId);
  const updateAnnotation = useUpdateTrainingAnnotation(videoId);
  const deleteAnnotation = useDeleteTrainingAnnotation(videoId);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [currentMs, setCurrentMs] = useState(0);
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
      setEndMs(endMs + 1000);
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
          Mark exact trick boundaries, give the trick a reusable name, and record what a
          technical judge would click.
        </p>
      </div>

      {videoBlob.isLoading ? (
        <p className="text-sm text-content-dim">Loading video…</p>
      ) : videoBlob.blobUrl ? (
        <VideoPlayerWithOverlay
          src={videoBlob.blobUrl}
          events={[]}
          onTimeUpdateMs={setCurrentMs}
        />
      ) : (
        <p className="text-sm text-status-alert">Could not load the video.</p>
      )}

      <form onSubmit={handleSubmit} className="grid gap-3 sm:grid-cols-2">
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
          Element type
          <input
            required
            value={elementType}
            onChange={(event) => setElementType(event.target.value)}
            placeholder="loop_combo"
            className="h-10 rounded-s border border-outline-default px-3"
          />
        </label>
        <div className="flex flex-col gap-1 text-sm">
          Start: {formatTimestamp(startMs)}
          <button
            type="button"
            onClick={() => setStartMs(currentMs)}
            className="h-10 rounded-s border border-outline-default px-3 text-left"
          >
            Set start to {formatTimestamp(currentMs)}
          </button>
        </div>
        <div className="flex flex-col gap-1 text-sm">
          End: {formatTimestamp(endMs)}
          <button
            type="button"
            onClick={() => setEndMs(currentMs)}
            className="h-10 rounded-s border border-outline-default px-3 text-left"
          >
            Set end to {formatTimestamp(currentMs)}
          </button>
        </div>
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
          Technical judge action
          <select
            value={technicalCredit}
            onChange={(event) =>
              setTechnicalCredit(event.target.value as TechnicalCredit)
            }
            className="h-10 rounded-s border border-outline-default px-3"
          >
            <option value="positive_click">Positive click</option>
            <option value="negative_click">Negative click</option>
            <option value="no_click">No click</option>
            <option value="uncertain">Needs review</option>
          </select>
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
            className="flex items-center justify-between rounded-s border border-outline-soft p-3"
          >
            <div>
              <p className="font-semibold text-content-default">{annotation.label}</p>
              <p className="text-xs text-content-dim">
                {formatTimestamp(annotation.start_ms)}–{formatTimestamp(annotation.end_ms)} ·{" "}
                {annotation.element_type} · {annotation.outcome} ·{" "}
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
