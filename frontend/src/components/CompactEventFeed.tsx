import { formatConfidence, formatMsAsTimecode, titleCaseFromSnakeCase } from "@/lib/format";
import { lineItemReasonLabel } from "@/lib/scoring-labels";
import type { AnalysisEvent, TechnicalLineItem } from "@/lib/types";

interface CompactEventFeedProps {
  events: AnalysisEvent[];
  lineItemsByEventId: Map<string, TechnicalLineItem>;
  currentMs: number;
  activeEventId: string | null;
  onSeek: (ms: number) => void;
  maxRows?: number;
}

/** A recording-friendly event feed that follows the playhead. */
export function CompactEventFeed({
  events,
  lineItemsByEventId,
  currentMs,
  activeEventId,
  onSeek,
  maxRows = 8,
}: CompactEventFeedProps): JSX.Element {
  const sortedEvents = [...events].sort((a, b) => a.start_ms - b.start_ms);
  const firstFutureIndex = sortedEvents.findIndex((event) => event.start_ms > currentMs);
  const focusIndex = firstFutureIndex === -1 ? sortedEvents.length - 1 : firstFutureIndex;
  const startIndex = Math.max(
    0,
    Math.min(focusIndex - 3, Math.max(0, sortedEvents.length - maxRows))
  );
  const visibleEvents = sortedEvents.slice(startIndex, startIndex + maxRows);

  return (
    <div className="overflow-hidden rounded-m border border-outline-soft bg-surface-default">
      <div className="grid grid-cols-[7rem_minmax(0,1fr)_7rem_6rem_5rem] gap-3 bg-surface-alt px-4 py-2 text-xs font-semibold uppercase tracking-wide text-content-dim">
        <span>Time</span>
        <span>Trick</span>
        <span>Outcome</span>
        <span>Confidence</span>
        <span className="text-right">Points</span>
      </div>
      <div aria-live="polite">
        {visibleEvents.map((event) => {
          const isActive = event.id === activeEventId;
          const isCompleted = event.end_ms <= currentMs;
          const lineItem = lineItemsByEventId.get(event.id);
          const rowClass = isActive
            ? "border-brand-primary-default bg-brand-primary-softest"
            : isCompleted
              ? "border-transparent bg-status-positive/5"
              : "border-transparent";

          return (
            <button
              type="button"
              key={event.id}
              onClick={() => onSeek(event.start_ms)}
              className={`grid w-full grid-cols-[7rem_minmax(0,1fr)_7rem_6rem_5rem] items-center gap-3 border-l-4 border-t border-t-outline-softest px-3 py-2 text-left text-sm transition-colors hover:bg-surface-alt ${rowClass}`}
            >
              <span className="font-mono text-xs text-content-subtle">
                {formatMsAsTimecode(event.start_ms)}
              </span>
              <span className="truncate font-medium text-content-default">{event.label}</span>
              <span className="text-content-subtle">
                {titleCaseFromSnakeCase(event.outcome)}
              </span>
              <span className="text-content-dim">{formatConfidence(event.confidence)}</span>
              <span className="text-right">
                {lineItem && isCompleted ? (
                  <span
                    className={
                      lineItem.points > 0
                        ? "font-semibold text-content-default"
                        : "text-content-dim"
                    }
                  >
                    {lineItem.points.toFixed(2)}
                    <span className="sr-only"> {lineItemReasonLabel(lineItem.reason)}</span>
                  </span>
                ) : (
                  <span className="text-content-dim">—</span>
                )}
              </span>
            </button>
          );
        })}
        {visibleEvents.length === 0 ? (
          <p className="px-4 py-6 text-center text-sm text-content-dim">No events detected yet.</p>
        ) : null}
      </div>
    </div>
  );
}
