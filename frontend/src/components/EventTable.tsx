"use client";

import { useState } from "react";

import { formatConfidence, formatMsAsTimecode, titleCaseFromSnakeCase } from "@/lib/format";
import { lineItemReasonLabel, nonScoringFamilyBadge } from "@/lib/scoring-labels";
import { DIFFICULTY_BANDS, EVENT_FAMILIES } from "@/lib/types";
import type {
  AnalysisEvent,
  DifficultyBand,
  EventFamily,
  Outcome,
  TechnicalLineItem,
} from "@/lib/types";

import {
  useConfirmEvent,
  useCreateEvent,
  useDeleteEvent,
  useRejectEvent,
  useUpdateEvent,
} from "@/hooks/useEvents";

const OUTCOMES: Outcome[] = ["success", "miss", "uncertain"];

const REVIEW_STATUS_STYLES: Record<AnalysisEvent["review_status"], string> = {
  pending: "bg-status-notice/15 text-status-notice",
  confirmed: "bg-status-positive/15 text-status-positive",
  rejected: "bg-status-alert/15 text-status-alert",
  edited: "bg-status-informative/15 text-status-informative",
};

interface EventTableProps {
  analysisId: string;
  events: AnalysisEvent[];
  lineItemsByEventId: Map<string, TechnicalLineItem>;
  currentMs: number;
  activeEventId: string | null;
  onSeek: (ms: number) => void;
}

/** Full add/edit/delete/confirm/reject editor for `AnalysisEvent` rows, per
 * Core Product Principle #4 ("Users must be able to add, edit, delete and
 * confirm every detected event"). Every field edit is a full-row PATCH; the
 * API flips `source` to human and `review_status` to `edited` server-side. */
export function EventTable({
  analysisId,
  events,
  lineItemsByEventId,
  currentMs,
  activeEventId,
  onSeek,
}: EventTableProps): JSX.Element {
  const updateEvent = useUpdateEvent(analysisId);
  const confirmEvent = useConfirmEvent(analysisId);
  const rejectEvent = useRejectEvent(analysisId);
  const deleteEvent = useDeleteEvent(analysisId);
  const createEvent = useCreateEvent(analysisId);

  const [isAdding, setIsAdding] = useState(false);
  const [newEvent, setNewEvent] = useState({
    label: "",
    family: "unknown_technical_element" as EventFamily,
    start_ms: 0,
    end_ms: 1000,
    outcome: "success" as Outcome,
    difficulty_band: "unknown" as DifficultyBand,
  });

  async function handleAdd(): Promise<void> {
    await createEvent.mutateAsync(newEvent);
    setIsAdding(false);
    setNewEvent({
      label: "",
      family: "unknown_technical_element",
      start_ms: 0,
      end_ms: 1000,
      outcome: "success",
      difficulty_band: "unknown",
    });
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto rounded-m border border-outline-soft">
        <table className="w-full min-w-[880px] text-left text-sm">
          <thead className="bg-surface-alt text-xs uppercase text-content-dim">
            <tr>
              <th className="px-3 py-2">Time</th>
              <th className="px-3 py-2">Label</th>
              <th className="px-3 py-2">Family</th>
              <th className="px-3 py-2">Outcome</th>
              <th className="px-3 py-2">Difficulty</th>
              <th className="px-3 py-2">Confidence</th>
              <th className="px-3 py-2">Pts</th>
              <th className="px-3 py-2">Source</th>
              <th className="px-3 py-2">Status</th>
              <th className="px-3 py-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {events.map((event) => {
              const isActive = event.id === activeEventId;
              const isCompleted = event.end_ms <= currentMs;
              const rowClass = isActive
                ? "bg-status-informative/10"
                : isCompleted
                  ? "bg-status-positive/5"
                  : "";
              return (
              <tr key={event.id} className={`border-t border-outline-softest ${rowClass}`}>
                <td className="px-3 py-2">
                  <button
                    type="button"
                    onClick={() => onSeek(event.start_ms)}
                    className="text-content-subtle underline-offset-2 hover:underline"
                  >
                    {formatMsAsTimecode(event.start_ms)}
                  </button>
                </td>
                <td className="px-3 py-2">
                  <input
                    defaultValue={event.label}
                    onBlur={(e) => {
                      if (e.target.value !== event.label) {
                        updateEvent.mutate({
                          eventId: event.id,
                          payload: { label: e.target.value },
                        });
                      }
                    }}
                    className="w-32 rounded-s border border-transparent bg-transparent px-1 hover:border-outline-default focus:border-outline-default"
                  />
                </td>
                <td className="px-3 py-2">
                  <select
                    value={event.family}
                    onChange={(e) =>
                      updateEvent.mutate({
                        eventId: event.id,
                        payload: { family: e.target.value as EventFamily },
                      })
                    }
                    className="rounded-s border border-outline-softest bg-transparent px-1"
                  >
                    {EVENT_FAMILIES.map((family) => (
                      <option key={family} value={family}>
                        {titleCaseFromSnakeCase(family)}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="px-3 py-2">
                  <select
                    value={event.outcome}
                    onChange={(e) =>
                      updateEvent.mutate({
                        eventId: event.id,
                        payload: { outcome: e.target.value as Outcome },
                      })
                    }
                    className="rounded-s border border-outline-softest bg-transparent px-1"
                  >
                    {OUTCOMES.map((outcome) => (
                      <option key={outcome} value={outcome}>
                        {titleCaseFromSnakeCase(outcome)}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="px-3 py-2">
                  <select
                    value={event.difficulty_band}
                    onChange={(e) =>
                      updateEvent.mutate({
                        eventId: event.id,
                        payload: { difficulty_band: e.target.value as DifficultyBand },
                      })
                    }
                    className="rounded-s border border-outline-softest bg-transparent px-1"
                  >
                    {DIFFICULTY_BANDS.map((band) => (
                      <option key={band} value={band}>
                        {titleCaseFromSnakeCase(band)}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="px-3 py-2 text-content-dim">{formatConfidence(event.confidence)}</td>
                <td className="px-3 py-2">
                  {(() => {
                    const lineItem = lineItemsByEventId.get(event.id);
                    const familyBadge = nonScoringFamilyBadge(event.family);
                    if (!lineItem) {
                      return <span className="text-content-dim">—</span>;
                    }
                    return (
                      <div className="flex flex-col gap-0.5">
                        <span
                          className={
                            lineItem.points > 0
                              ? "font-semibold text-content-default"
                              : "text-content-dim"
                          }
                        >
                          {lineItem.points.toFixed(2)}
                        </span>
                        <span className="text-xs text-content-dim">
                          {lineItemReasonLabel(lineItem.reason)}
                        </span>
                        {familyBadge ? (
                          <span className="text-xs font-semibold text-status-notice">
                            {familyBadge}
                          </span>
                        ) : null}
                      </div>
                    );
                  })()}
                </td>
                <td className="px-3 py-2 text-content-dim">{titleCaseFromSnakeCase(event.source)}</td>
                <td className="px-3 py-2">
                  <span
                    className={`rounded-s px-2 py-0.5 text-xs font-semibold ${REVIEW_STATUS_STYLES[event.review_status]}`}
                  >
                    {titleCaseFromSnakeCase(event.review_status)}
                  </span>
                </td>
                <td className="px-3 py-2">
                  <div className="flex gap-2">
                    <button
                      type="button"
                      title="Confirm"
                      onClick={() => confirmEvent.mutate(event.id)}
                      className="text-status-positive hover:underline"
                    >
                      Confirm
                    </button>
                    <button
                      type="button"
                      title="Reject"
                      onClick={() => rejectEvent.mutate(event.id)}
                      className="text-status-notice hover:underline"
                    >
                      Reject
                    </button>
                    <button
                      type="button"
                      title="Delete"
                      onClick={() => {
                        if (window.confirm(`Delete event "${event.label}"?`)) {
                          deleteEvent.mutate(event.id);
                        }
                      }}
                      className="text-status-alert hover:underline"
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            );
            })}
            {events.length === 0 ? (
              <tr>
                <td colSpan={10} className="px-3 py-4 text-center text-content-dim">
                  No events detected yet.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      {isAdding ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void handleAdd();
          }}
          className="flex flex-wrap items-end gap-3 rounded-m border border-outline-soft bg-surface-default p-4"
        >
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            Label
            <input
              required
              value={newEvent.label}
              onChange={(e) => setNewEvent((prev) => ({ ...prev, label: e.target.value }))}
              className="h-9 rounded-s border border-outline-default px-2 text-sm"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            Family
            <select
              value={newEvent.family}
              onChange={(e) =>
                setNewEvent((prev) => ({ ...prev, family: e.target.value as EventFamily }))
              }
              className="h-9 rounded-s border border-outline-default px-2 text-sm"
            >
              {EVENT_FAMILIES.map((family) => (
                <option key={family} value={family}>
                  {titleCaseFromSnakeCase(family)}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            Start (ms)
            <input
              type="number"
              min={0}
              value={newEvent.start_ms}
              onChange={(e) =>
                setNewEvent((prev) => ({ ...prev, start_ms: Number(e.target.value) }))
              }
              className="h-9 w-24 rounded-s border border-outline-default px-2 text-sm"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            End (ms)
            <input
              type="number"
              min={0}
              value={newEvent.end_ms}
              onChange={(e) =>
                setNewEvent((prev) => ({ ...prev, end_ms: Number(e.target.value) }))
              }
              className="h-9 w-24 rounded-s border border-outline-default px-2 text-sm"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            Outcome
            <select
              value={newEvent.outcome}
              onChange={(e) =>
                setNewEvent((prev) => ({ ...prev, outcome: e.target.value as Outcome }))
              }
              className="h-9 rounded-s border border-outline-default px-2 text-sm"
            >
              {OUTCOMES.map((outcome) => (
                <option key={outcome} value={outcome}>
                  {titleCaseFromSnakeCase(outcome)}
                </option>
              ))}
            </select>
          </label>
          <button
            type="submit"
            disabled={createEvent.isPending}
            className="h-9 rounded-full bg-brand-primary px-4 text-sm font-semibold text-white disabled:opacity-60"
          >
            Add event
          </button>
          <button
            type="button"
            onClick={() => setIsAdding(false)}
            className="h-9 rounded-full px-4 text-sm font-semibold text-content-subtle hover:bg-surface-alt"
          >
            Cancel
          </button>
        </form>
      ) : (
        <button
          type="button"
          onClick={() => setIsAdding(true)}
          className="self-start rounded-full border border-outline-default px-4 py-2 text-sm font-semibold text-content-default hover:bg-surface-alt"
        >
          + Add event manually
        </button>
      )}
    </div>
  );
}
