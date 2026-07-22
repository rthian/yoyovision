import type { EventFamily } from "@/lib/types";

/** Mirrors `yoyovision_ml.domain.MISTAKE_EVENT_FAMILIES`. */
export const MISTAKE_EVENT_FAMILIES: ReadonlySet<EventFamily> = new Set([
  "control_miss",
  "landing_miss",
  "catch_miss",
]);

/** Mirrors `yoyovision_ml.domain.EQUIPMENT_EVENT_FAMILIES`. */
export const EQUIPMENT_EVENT_FAMILIES: ReadonlySet<EventFamily> = new Set([
  "yoyo_stop",
  "yoyo_change",
  "yoyo_detach",
]);

const LINE_ITEM_REASON_LABELS: Record<string, string> = {
  credited: "Credited",
  excluded_mistake: "Mistake",
  excluded_equipment: "Equipment",
  excluded_unknown: "Unclassified",
  excluded_uncertain: "Uncertain",
  excluded_outcome_miss: "Miss",
};

export function lineItemReasonLabel(reason: string): string {
  if (reason.startsWith("repeat_occurrence_")) {
    const occurrence = reason.replace("repeat_occurrence_", "");
    return `Repeat #${occurrence}`;
  }
  return LINE_ITEM_REASON_LABELS[reason] ?? reason;
}

export function nonScoringFamilyBadge(family: EventFamily): string | null {
  if (MISTAKE_EVENT_FAMILIES.has(family)) {
    return "Mistake";
  }
  if (EQUIPMENT_EVENT_FAMILIES.has(family)) {
    return "Equipment";
  }
  return null;
}
