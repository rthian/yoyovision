import type {
  AnalysisEvent,
  HumanJudgingEntryReference,
  JudgeTechnicalClick,
} from "@/lib/types";

export const HUMAN_CLICK_MATCH_TOLERANCE_MS = 1_500;

export interface MatchedHumanClick extends JudgeTechnicalClick {
  judgeName: string;
  entryTitle: string;
}

export interface HumanEventEvidence {
  clicks: MatchedHumanClick[];
  positive: number;
  negative: number;
  net: number;
  judgeCount: number;
}

export interface HumanClickComparison {
  byEventId: Map<string, HumanEventEvidence>;
  unmatchedClicks: MatchedHumanClick[];
  totalClicks: number;
}

function distanceFromEvent(timestampMs: number, event: AnalysisEvent): number {
  if (timestampMs < event.start_ms) return event.start_ms - timestampMs;
  if (timestampMs > event.end_ms) return timestampMs - event.end_ms;
  return 0;
}

function eventForClick(
  events: AnalysisEvent[],
  timestampMs: number,
  toleranceMs: number
): AnalysisEvent | undefined {
  return events
    .map((event) => ({
      event,
      distance: distanceFromEvent(timestampMs, event),
      midpointDistance: Math.abs(timestampMs - (event.start_ms + event.end_ms) / 2),
    }))
    .filter(({ distance }) => distance <= toleranceMs)
    .sort((left, right) =>
      left.distance === right.distance
        ? left.midpointDistance - right.midpointDistance
        : left.distance - right.distance
    )[0]?.event;
}

export function buildHumanClickComparison(
  events: AnalysisEvent[],
  entries: HumanJudgingEntryReference[],
  toleranceMs = HUMAN_CLICK_MATCH_TOLERANCE_MS
): HumanClickComparison {
  const clicks: MatchedHumanClick[] = [];
  const seenClickIds = new Set<string>();

  for (const entry of entries) {
    for (const judge of entry.judges) {
      if (!judge.include_in_results || judge.is_shadow) continue;
      for (const click of judge.technical_clicks) {
        if (seenClickIds.has(click.id)) continue;
        seenClickIds.add(click.id);
        clicks.push({ ...click, judgeName: judge.display_name, entryTitle: entry.title });
      }
    }
  }

  const grouped = new Map<string, MatchedHumanClick[]>();
  const unmatchedClicks: MatchedHumanClick[] = [];
  for (const click of clicks) {
    const event = eventForClick(events, click.timestamp_ms, toleranceMs);
    if (!event) {
      unmatchedClicks.push(click);
      continue;
    }
    grouped.set(event.id, [...(grouped.get(event.id) ?? []), click]);
  }

  const byEventId = new Map<string, HumanEventEvidence>();
  for (const [eventId, matchedClicks] of grouped) {
    const positive = matchedClicks.filter((click) => click.kind === "positive").length;
    const negative = matchedClicks.length - positive;
    byEventId.set(eventId, {
      clicks: matchedClicks.sort((left, right) => left.timestamp_ms - right.timestamp_ms),
      positive,
      negative,
      net: positive - negative,
      judgeCount: new Set(matchedClicks.map((click) => click.judgeName)).size,
    });
  }

  return { byEventId, unmatchedClicks, totalClicks: clicks.length };
}

export function humanEvidenceDisagrees(
  event: AnalysisEvent,
  evidence: HumanEventEvidence | undefined
): boolean {
  if (!evidence || evidence.clicks.length === 0) return false;
  if (event.outcome === "success") return evidence.net <= 0;
  return evidence.net > 0;
}
