"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";

import { formatMsAsTimecode } from "@/lib/format";
import type { ClickMode, JudgeClick } from "@/lib/types";
import { useCreateJudgeClick, useDeleteJudgeClick } from "@/hooks/useJudgeClicks";

interface JudgeClickerProps {
  token: string;
  entryVideoId: string;
  clickMode: ClickMode;
  durationMs: number;
  clicks: JudgeClick[];
  readOnly: boolean;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}


function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    target.isContentEditable
  );
}

function lastAddedClick(clicks: JudgeClick[]): JudgeClick | undefined {
  if (clicks.length === 0) {
    return undefined;
  }
  return [...clicks].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  )[0];
}

export function JudgeClicker({
  token,
  entryVideoId,
  clickMode,
  durationMs,
  clicks,
  readOnly,
  videoRef,
}: JudgeClickerProps): JSX.Element | null {
  const createClick = useCreateJudgeClick(token, entryVideoId);
  const deleteClick = useDeleteJudgeClick(token, entryVideoId);
  const busyRef = useRef(false);

  const sortedClicks = useMemo(
    () => [...clicks].sort((a, b) => a.timestamp_ms - b.timestamp_ms),
    [clicks]
  );
  const undoTarget = useMemo(() => lastAddedClick(clicks), [clicks]);

  const addClick = useCallback(() => {
    const video = videoRef.current;
    if (!video || readOnly || clickMode === "off" || busyRef.current) {
      return;
    }
    busyRef.current = true;
    const timestampMs = Math.round(video.currentTime * 1000);
    createClick.mutate(
      { timestamp_ms: timestampMs },
      { onSettled: () => { busyRef.current = false; } }
    );
  }, [videoRef, readOnly, clickMode, createClick]);

  const removeLastClick = useCallback(() => {
    if (!undoTarget || readOnly || clickMode === "off" || busyRef.current) {
      return;
    }
    busyRef.current = true;
    deleteClick.mutate(undoTarget.id, { onSettled: () => { busyRef.current = false; } });
  }, [undoTarget, readOnly, clickMode, deleteClick]);
  useEffect(() => {
    if (readOnly || clickMode === "off") {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTypingTarget(event.target)) {
        return;
      }
      const addKey =
        event.key === "+" ||
        event.key === "=" ||
        event.code === "NumpadAdd";
      const removeKey = event.key === "-" || event.code === "NumpadSubtract";
      if (!addKey && !removeKey) {
        return;
      }
      event.preventDefault();
      if (addKey) {
        addClick();
      } else {
        removeLastClick();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [addClick, removeLastClick, readOnly, clickMode]);

  if (clickMode === "off") {
    return null;
  }

  const safeDuration = durationMs > 0 ? durationMs : 1;
  const pending = createClick.isPending || deleteClick.isPending;
  const clickError =
    (createClick.error instanceof Error ? createClick.error.message : null) ||
    (deleteClick.error instanceof Error ? deleteClick.error.message : null);

  return (
    <section className="flex flex-col gap-4 rounded-m border border-outline-soft bg-surface-alt p-4">
      <div>
        <h3 className="text-sm font-semibold text-content-default">Timestamp clicker</h3>
        <p className="text-xs text-content-dim">
          {clickMode === "technical_score"
            ? "Tap + when you see a trick, − to undo. Keyboard: + / −"
            : "Tap + / − to mark or undo. Keyboard: + / −"}
        </p>
      </div>

      {clickError ? (
        <p role="alert" className="text-sm text-status-alert">
          {clickError}
        </p>
      ) : null}

      {!readOnly ? (
        <div className="flex items-center justify-center gap-6">
          <button
            type="button"
            aria-label="Remove last click"
            onClick={removeLastClick}
            disabled={pending || !undoTarget}
            className="flex h-16 w-16 items-center justify-center rounded-full bg-surface-default text-3xl font-bold text-content-default shadow-sm ring-1 ring-outline-soft disabled:opacity-40"
          >
            −
          </button>
          <div className="flex min-w-[4rem] flex-col items-center">
            <span className="text-3xl font-bold tabular-nums text-content-default">
              {clicks.length}
            </span>
            <span className="text-xs text-content-dim">clicks</span>
          </div>
          <button
            type="button"
            aria-label="Add click at current time"
            onClick={addClick}
            disabled={pending}
            className="flex h-16 w-16 items-center justify-center rounded-full bg-brand-default text-3xl font-bold text-content-on-brand disabled:opacity-50"
          >
            +
          </button>
        </div>
      ) : (
        <p className="text-center text-2xl font-bold tabular-nums text-content-default">
          {clicks.length} clicks
        </p>
      )}

      <div
        className="relative h-6 w-full rounded-full bg-outline-softest"
        aria-hidden
      >
        {sortedClicks.map((click) => (
          <div
            key={click.id}
            className="absolute top-0 h-full w-0.5 bg-brand-primary-bold"
            style={{ left: `${(click.timestamp_ms / safeDuration) * 100}%` }}
            title={formatMsAsTimecode(click.timestamp_ms)}
          />
        ))}
      </div>

      {sortedClicks.length > 0 ? (
        <ul className="flex max-h-32 flex-col gap-1 overflow-y-auto text-xs text-content-subtle">
          {sortedClicks.map((click) => (
            <li key={click.id} className="tabular-nums">
              {formatMsAsTimecode(click.timestamp_ms)}
              {click.label ? ` · ${click.label}` : ""}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
