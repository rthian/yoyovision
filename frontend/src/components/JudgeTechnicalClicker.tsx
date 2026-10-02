"use client";

import { useEffect, useRef } from "react";

import { formatMsAsTimecode } from "@/lib/format";
import type { JudgeTechnicalClick, TechnicalClickKind } from "@/lib/types";

interface JudgeTechnicalClickerProps {
  clicks: JudgeTechnicalClick[];
  currentMs: number;
  readOnly: boolean;
  isSaving?: boolean;
  onClick: (kind: TechnicalClickKind) => void;
  onUndo: (clickId: string) => void;
}

export function JudgeTechnicalClicker({
  clicks,
  currentMs,
  readOnly,
  isSaving = false,
  onClick,
  onUndo,
}: JudgeTechnicalClickerProps): JSX.Element {
  const onClickRef = useRef(onClick);
  onClickRef.current = onClick;

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent): void {
      if (readOnly || event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, textarea, select") || target?.isContentEditable) return;
      if (["ArrowRight", "+", "="].includes(event.key)) {
        event.preventDefault();
        onClickRef.current("positive");
      } else if (["ArrowLeft", "-", "_"].includes(event.key)) {
        event.preventDefault();
        onClickRef.current("negative");
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [readOnly]);

  const positives = clicks.filter((click) => click.kind === "positive").length;
  const negatives = clicks.length - positives;
  const net = positives - negatives;
  const latestClick = [...clicks].sort((a, b) => b.created_at.localeCompare(a.created_at))[0];

  return (
    <section className="rounded-m border border-outline-soft bg-surface-default p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-content-default">Technical scoring</h2>
          <p className="text-sm text-content-dim">
            Click while the trick happens. Every action is saved with the video timestamp.
          </p>
        </div>
        <div className="text-right">
          <p className="font-mono text-xs text-content-dim">{formatMsAsTimecode(currentMs)}</p>
          <p className="text-3xl font-bold tabular-nums text-content-default">{net}</p>
          <p className="text-xs text-content-dim">net technical clicks</p>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <button
          type="button"
          disabled={readOnly}
          onClick={() => onClick("negative")}
          className="min-h-40 rounded-m border-2 border-status-alert/40 bg-status-alert/10 text-status-alert transition active:scale-[0.98] disabled:opacity-50"
        >
          <span className="block text-5xl font-bold">−</span>
          <span className="block text-base font-bold">NEGATIVE</span>
          <span className="mt-1 block text-sm">{negatives} clicks · ← key</span>
        </button>
        <button
          type="button"
          disabled={readOnly}
          onClick={() => onClick("positive")}
          className="min-h-40 rounded-m border-2 border-status-positive/40 bg-status-positive/10 text-status-positive transition active:scale-[0.98] disabled:opacity-50"
        >
          <span className="block text-5xl font-bold">+</span>
          <span className="block text-base font-bold">POSITIVE</span>
          <span className="mt-1 block text-sm">{positives} clicks · → key</span>
        </button>
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-content-dim">
        <span>{isSaving ? "Saving click…" : `${clicks.length} timestamped actions saved`}</span>
        {latestClick && !readOnly ? (
          <button
            type="button"
            onClick={() => onUndo(latestClick.id)}
            className="rounded-full border border-outline-default px-3 py-1.5 font-semibold text-content-subtle hover:bg-surface-alt"
          >
            Undo last ({latestClick.kind} at {formatMsAsTimecode(latestClick.timestamp_ms)})
          </button>
        ) : null}
      </div>
      {readOnly ? (
        <p className="mt-3 text-sm font-semibold text-status-positive">Submitted — technical clicks are locked</p>
      ) : null}
    </section>
  );
}
