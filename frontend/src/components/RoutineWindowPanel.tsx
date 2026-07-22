"use client";

import { useState } from "react";

import { formatMsAsTimecode } from "@/lib/format";
import type { RoutineWindow } from "@/lib/routine-window";

interface RoutineWindowPanelProps {
  window: RoutineWindow;
  currentMs: number;
  videoDurationMs: number;
  isSaving: boolean;
  readOnly?: boolean;
  onSetStartToPlayhead: () => void;
  onSetEndToPlayhead: () => void;
  onSave: (startMs: number, endMs: number) => Promise<void>;
}

/** Lets a judge mark measure start and music stop within a longer upload. */
export function RoutineWindowPanel({
  window,
  currentMs,
  videoDurationMs,
  isSaving,
  readOnly = false,
  onSetStartToPlayhead,
  onSetEndToPlayhead,
  onSave,
}: RoutineWindowPanelProps): JSX.Element {
  const [startMs, setStartMs] = useState(window.startMs);
  const [endMs, setEndMs] = useState(window.endMs);

  async function handleSave(): Promise<void> {
    await onSave(startMs, endMs);
  }

  return (
    <section className="flex flex-col gap-3 rounded-m border border-outline-soft bg-surface-default p-4">
      <div>
        <h2 className="text-lg font-semibold text-content-default">Routine window</h2>
        <p className="text-sm text-content-dim">
          Mark where the measure starts and where the music stops. Playback pauses at the end, and
          scoring only counts tricks inside this span.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-content-dim">Measure start</span>
          <div className="flex gap-2">
            <input
              type="number"
              min={0}
              max={videoDurationMs}
              disabled={readOnly}
              value={startMs}
              onChange={(event) => setStartMs(Number(event.target.value))}
              className="w-full rounded-s border border-outline-soft px-3 py-2"
            />
            <button
              type="button"
              disabled={readOnly}
              onClick={() => {
                onSetStartToPlayhead();
                setStartMs(currentMs);
              }}
              className="shrink-0 rounded-full bg-brand-secondary-softest px-3 py-2 text-xs font-semibold text-brand-secondary-boldest"
            >
              Use playhead
            </button>
          </div>
          <span className="text-xs text-content-dim">{formatMsAsTimecode(startMs)}</span>
        </label>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-content-dim">Music stop</span>
          <div className="flex gap-2">
            <input
              type="number"
              min={0}
              max={videoDurationMs}
              disabled={readOnly}
              value={endMs}
              onChange={(event) => setEndMs(Number(event.target.value))}
              className="w-full rounded-s border border-outline-soft px-3 py-2"
            />
            <button
              type="button"
              disabled={readOnly}
              onClick={() => {
                onSetEndToPlayhead();
                setEndMs(currentMs);
              }}
              className="shrink-0 rounded-full bg-brand-secondary-softest px-3 py-2 text-xs font-semibold text-brand-secondary-boldest"
            >
              Use playhead
            </button>
          </div>
          <span className="text-xs text-content-dim">{formatMsAsTimecode(endMs)}</span>
        </label>
      </div>

      {readOnly ? null : (
      <button
        type="button"
        disabled={isSaving || endMs <= startMs}
        onClick={() => void handleSave()}
        className="self-start rounded-full bg-brand-primary-default px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
      >
        {isSaving ? "Saving..." : "Save routine window"}
      </button>
      )}
    </section>
  );
}
