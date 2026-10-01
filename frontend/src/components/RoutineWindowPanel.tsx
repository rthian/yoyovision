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
  const [isOpen, setIsOpen] = useState(false);

  async function handleSave(): Promise<void> {
    await onSave(startMs, endMs);
    setIsOpen(false);
  }

  return (
    <section className="flex flex-col gap-3 rounded-m border border-outline-soft bg-surface-default p-4">
      <button
        type="button"
        aria-expanded={isOpen}
        onClick={() => setIsOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-4 text-left"
      >
        <span>
          <span className="block text-lg font-semibold text-content-default">Routine window</span>
          <span className="mt-0.5 block text-sm text-content-dim">
            {formatMsAsTimecode(startMs)} → {formatMsAsTimecode(endMs)}
          </span>
        </span>
        <span className="flex items-center gap-2">
          <span className="rounded-full bg-status-positive/15 px-2.5 py-1 text-xs font-semibold text-status-positive">
            Set
          </span>
          <span aria-hidden="true" className="text-lg text-content-dim">
            {isOpen ? "−" : "+"}
          </span>
        </span>
      </button>

      {isOpen ? (
        <div className="flex flex-col gap-3 border-t border-outline-softest pt-3">
          <p className="text-sm text-content-dim">
            Mark where the measure starts and where the music stops. Playback pauses at the end,
            and scoring only counts tricks inside this span.
          </p>

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
        </div>
      ) : null}
    </section>
  );
}
