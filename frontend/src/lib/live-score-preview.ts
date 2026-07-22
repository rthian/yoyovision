import type {
  AnalysisEvent,
  DeductionType,
  MajorDeduction,
  ReviewStatus,
  Ruleset,
  ScoreBreakdown,
  TechnicalLineItem,
} from "@/lib/types";

export interface LiveScorePreview {
  up_to_ms: number;
  completed_event_count: number;
  active_event_id: string | null;
  technical_raw: number;
  technical_scaled: number;
  freestyle_evaluation_scaled: number;
  major_deductions: number;
  final_score: number;
}

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function deductionRuleFor(ruleset: Ruleset, type: DeductionType) {
  return ruleset.deduction_rules.find((rule) => rule.type === type) ?? null;
}

function deductionIsScorable(
  deduction: MajorDeduction,
  ruleset: Ruleset,
  reviewStatus: ReviewStatus
): boolean {
  if (reviewStatus === "rejected") {
    return false;
  }
  const rule = deductionRuleFor(ruleset, deduction.type);
  if (!rule?.requires_manual_confirmation) {
    return true;
  }
  return reviewStatus === "confirmed";
}

function majorDeductionsUpTo(
  deductions: MajorDeduction[],
  ruleset: Ruleset,
  upToMs: number
): number {
  const scorable = deductions.filter(
    (deduction) =>
      deduction.timestamp_ms <= upToMs &&
      deductionIsScorable(deduction, ruleset, deduction.review_status)
  );
  const sorted = [...scorable].sort((left, right) => left.timestamp_ms - right.timestamp_ms);
  const quantityUsedByType = new Map<DeductionType, number>();
  let total = 0;

  for (const deduction of sorted) {
    const rule = deductionRuleFor(ruleset, deduction.type);
    if (!rule) {
      if (deduction.review_status !== "rejected") {
        total += deduction.points;
      }
      continue;
    }

    let rowPoints = deduction.points ?? rule.points_per_occurrence * deduction.quantity;
    let allowedQuantity = deduction.quantity;

    if (rule.max_occurrences_penalized != null) {
      const cap = rule.max_occurrences_penalized;
      const prior = quantityUsedByType.get(deduction.type) ?? 0;
      allowedQuantity = Math.max(0, Math.min(deduction.quantity, cap - prior));
      quantityUsedByType.set(deduction.type, prior + deduction.quantity);
      if (allowedQuantity === 0) {
        continue;
      }
      if (allowedQuantity < deduction.quantity) {
        rowPoints *= allowedQuantity / deduction.quantity;
      }
    } else {
      quantityUsedByType.set(
        deduction.type,
        (quantityUsedByType.get(deduction.type) ?? 0) + deduction.quantity
      );
    }

    total += rowPoints;
  }

  return round3(total);
}

/** Client-side mirror of `score_preview_at_ms` for smooth playback updates. */
export function computeLiveScorePreview(
  events: AnalysisEvent[],
  lineItemsByEventId: Map<string, TechnicalLineItem>,
  deductions: MajorDeduction[],
  score: ScoreBreakdown,
  ruleset: Ruleset,
  currentMs: number,
  routineWindow?: { startMs: number; endMs: number }
): LiveScorePreview {
  const window = routineWindow ?? { startMs: 0, endMs: Number.POSITIVE_INFINITY };
  const completedEvents = events.filter(
    (event) =>
      event.review_status !== "rejected" &&
      event.end_ms <= currentMs &&
      event.start_ms >= window.startMs &&
      event.end_ms <= window.endMs
  );
  let technicalRaw = 0;
  for (const event of completedEvents) {
    technicalRaw += lineItemsByEventId.get(event.id)?.points ?? 0;
  }
  technicalRaw = round3(technicalRaw);
  const technicalScaled = round3(Math.min(technicalRaw, ruleset.technical_scale_max));
  const majorDeductions = majorDeductionsUpTo(
    deductions.filter(
      (deduction) =>
        deduction.timestamp_ms >= window.startMs && deduction.timestamp_ms <= window.endMs
    ),
    ruleset,
    currentMs
  );
  const finalScore = round3(
    Math.max(
      0,
      ruleset.technical_weight * technicalScaled +
        ruleset.freestyle_evaluation_weight * score.freestyle_evaluation_scaled -
        majorDeductions
    )
  );
  const activeEvent =
    events.find(
      (event) =>
        event.review_status !== "rejected" &&
        currentMs >= event.start_ms &&
        currentMs <= event.end_ms &&
        event.start_ms >= window.startMs &&
        event.end_ms <= window.endMs
    ) ?? null;

  return {
    up_to_ms: currentMs,
    completed_event_count: completedEvents.length,
    active_event_id: activeEvent?.id ?? null,
    technical_raw: technicalRaw,
    technical_scaled: technicalScaled,
    freestyle_evaluation_scaled: score.freestyle_evaluation_scaled,
    major_deductions: majorDeductions,
    final_score: finalScore,
  };
}
