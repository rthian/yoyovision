"use client";

import { useEffect, useState } from "react";

import { FREESTYLE_EVALUATION_FIELDS } from "@/lib/types";
import type { FreestyleEvaluation, FreestyleEvaluationUpsert } from "@/lib/types";

import { useUpsertEvaluation } from "@/hooks/useEvaluation";

function toFormState(evaluation: FreestyleEvaluation | null): FreestyleEvaluationUpsert {
  if (!evaluation) {
    return { notes: "" };
  }
  return {
    execution: evaluation.execution,
    control: evaluation.control,
    trick_diversity: evaluation.trick_diversity,
    space_use_emphasis: evaluation.space_use_emphasis,
    music_choreography: evaluation.music_choreography,
    music_construction: evaluation.music_construction,
    body_control: evaluation.body_control,
    showmanship: evaluation.showmanship,
    notes: evaluation.notes,
  };
}

interface FreestyleEvaluationFormProps {
  analysisId: string;
  evaluation: FreestyleEvaluation | null;
}

/** Manual-entry form for the Freestyle Evaluation (MVP scope: "Freestyle
 * Evaluation placeholders and manual values" -- there is no model for this
 * yet, every field is a human-entered 0-10 score). */
export function FreestyleEvaluationForm({
  analysisId,
  evaluation,
}: FreestyleEvaluationFormProps): JSX.Element {
  const upsertEvaluation = useUpsertEvaluation(analysisId);
  const [form, setForm] = useState<FreestyleEvaluationUpsert>(() => toFormState(evaluation));

  useEffect(() => {
    setForm(toFormState(evaluation));
  }, [evaluation]);

  function handleSubmit(): void {
    upsertEvaluation.mutate(form);
  }

  return (
    <div className="flex flex-col gap-4 rounded-m border border-outline-soft bg-surface-default p-4">
      <div>
        <h3 className="text-base font-semibold text-content-default">Freestyle evaluation</h3>
        <p className="text-sm text-content-dim">
          Manual judge-style scores (0-10). No model currently produces these.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {FREESTYLE_EVALUATION_FIELDS.map(({ key, label }) => (
          <label key={key} className="flex flex-col gap-1 text-xs text-content-dim">
            {label}
            <input
              type="number"
              min={0}
              max={10}
              step={0.5}
              value={form[key] ?? ""}
              onChange={(e) =>
                setForm((prev) => ({
                  ...prev,
                  [key]: e.target.value === "" ? null : Number(e.target.value),
                }))
              }
              className="h-9 rounded-s border border-outline-default px-2 text-sm"
            />
          </label>
        ))}
      </div>
      <label className="flex flex-col gap-1 text-xs text-content-dim">
        Notes
        <textarea
          value={form.notes ?? ""}
          onChange={(e) => setForm((prev) => ({ ...prev, notes: e.target.value }))}
          maxLength={4096}
          className="h-20 rounded-m border border-outline-default p-2 text-sm"
        />
      </label>
      <button
        type="button"
        onClick={handleSubmit}
        disabled={upsertEvaluation.isPending}
        className="self-start rounded-full bg-brand-primary px-4 py-2 text-sm font-semibold text-white disabled:opacity-60"
      >
        {upsertEvaluation.isPending ? "Saving..." : "Save evaluation"}
      </button>
    </div>
  );
}
