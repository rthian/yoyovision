"use client";

import { formatMsAsTimecode } from "@/lib/format";
import type { AnalysisEvent } from "@/lib/types";

const OUTCOME_COLORS: Record<AnalysisEvent["outcome"], string> = {
  success: "bg-status-positive",
  miss: "bg-status-alert",
  uncertain: "bg-status-notice",
};

interface EventTimelineProps {
  events: AnalysisEvent[];
  durationMs: number;
  currentMs: number;
  onSeek: (ms: number) => void;
  routineStartMs?: number;
  routineEndMs?: number;
}

/** A clickable timeline strip: each event renders as a colored segment
 * (green=success, red=miss, amber=uncertain) positioned/sized by its
 * start/end ms, and a playhead marker tracks the current video position. */
export function EventTimeline({
  events,
  durationMs,
  currentMs,
  onSeek,
  routineStartMs = 0,
  routineEndMs,
}: EventTimelineProps): JSX.Element {
  const safeDuration = durationMs > 0 ? durationMs : 1;
  const routineEnd = routineEndMs ?? durationMs;
  const startPct = (routineStartMs / safeDuration) * 100;
  const endPct = (routineEnd / safeDuration) * 100;

  return (
    <div className="flex flex-col gap-2">
      <div
        className="relative h-8 w-full cursor-pointer rounded-full bg-outline-softest"
        role="slider"
        aria-label="Event timeline"
        aria-valuemin={0}
        aria-valuemax={durationMs}
        aria-valuenow={currentMs}
        tabIndex={0}
        onClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientX - rect.left) / rect.width;
          onSeek(Math.round(ratio * safeDuration));
        }}
      >
        {routineStartMs > 0 ? (
          <div
            className="pointer-events-none absolute inset-y-0 left-0 bg-content-default/10"
            style={{ width: `${startPct}%` }}
          />
        ) : null}
        {routineEnd < durationMs ? (
          <div
            className="pointer-events-none absolute inset-y-0 right-0 bg-content-default/10"
            style={{ width: `${100 - endPct}%` }}
          />
        ) : null}
        <div
          className="pointer-events-none absolute inset-y-1 border-l-2 border-brand-primary-bold"
          style={{ left: `${startPct}%` }}
          title="Measure start"
        />
        <div
          className="pointer-events-none absolute inset-y-1 border-r-2 border-brand-primary-bold"
          style={{ left: `${endPct}%` }}
          title="Music stop"
        />
        {events.map((eventItem) => {
          const left = (eventItem.start_ms / safeDuration) * 100;
          const width = Math.max(
            0.5,
            ((eventItem.end_ms - eventItem.start_ms) / safeDuration) * 100
          );
          const isActive = currentMs >= eventItem.start_ms && currentMs <= eventItem.end_ms;
          return (
            <button
              key={eventItem.id}
              type="button"
              title={`${eventItem.label} (${formatMsAsTimecode(eventItem.start_ms)})`}
              aria-label={`Seek to ${eventItem.label}`}
              onClick={(clickEvent) => {
                clickEvent.stopPropagation();
                onSeek(eventItem.start_ms);
              }}
              className={`absolute top-1 h-6 rounded-full opacity-80 ${OUTCOME_COLORS[eventItem.outcome]} ${
                isActive ? "ring-2 ring-status-informative ring-offset-1" : ""
              }`}
              style={{ left: `${left}%`, width: `${width}%` }}
            />
          );
        })}
        <div
          className="absolute top-0 h-8 w-0.5 bg-content-default"
          style={{ left: `${(currentMs / safeDuration) * 100}%` }}
        />
      </div>
      <div className="flex gap-4 text-xs text-content-dim">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-status-positive" /> Success
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-status-alert" /> Miss
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-status-notice" /> Uncertain
        </span>
      </div>
    </div>
  );
}
