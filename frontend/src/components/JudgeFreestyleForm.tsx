"use client";

import { useEffect, useState } from "react";

import type { FreestyleEvaluationUpsert, JudgeFreestyleScore } from "@/lib/types";
import { FREESTYLE_EVALUATION_FIELDS } from "@/lib/types";

function toFormState(score: JudgeFreestyleScore | null): FreestyleEvaluationUpsert {
  if (!score) {
    return { notes: "" };
  }
  return {
    execution: score.execution,
    control: score.control,
    trick_diversity: score.trick_diversity,
    space_use_emphasis: score.space_use_emphasis,
    music_choreography: score.music_choreography,
    music_construction: score.music_construction,
    body_control: score.body_control,
    showmanship: score.showmanship,
    notes: score.notes,
  };
}

interface JudgeFreestyleFormProps {
  score: JudgeFreestyleScore | null;
  readOnly?: boolean;
  isSaving?: boolean;
  isSubmitting?: boolean;
  onSaveDraft: (payload: FreestyleEvaluationUpsert) => void;
  onSubmit: (payload: FreestyleEvaluationUpsert) => void;
}

export function JudgeFreestyleForm({
  score,
  readOnly = false,
  isSaving = false,
  isSubmitting = false,
  onSaveDraft,
  onSubmit,
}: JudgeFreestyleFormProps): JSX.Element {
  const [form, setForm] = useState<FreestyleEvaluationUpsert>(() => toFormState(score));

  useEffect(() => {
    setForm(toFormState(score));
  }, [score]);

  function handleSaveDraft(): void {
    onSaveDraft(form);
  }

  function handleSubmit(): void {
    const confirmed = window.confirm(
      "Submit scores for this video? You will not be able to edit them after submitting."
    );
    if (!confirmed) {
      return;
    }
    onSubmit(form);
  }

  const disabled = readOnly || isSaving || isSubmitting;

  return (
    <div className="flex flex-col gap-4 rounded-m border border-outline-soft bg-surface-default p-4">
      <div>
        <h3 className="text-base font-semibold text-content-default">Freestyle evaluation</h3>
        <p className="text-sm text-content-dim">Score each category from 0 to 10.</p>
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
              disabled={disabled}
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
          disabled={disabled}
          value={form.notes ?? ""}
          onChange={(e) => setForm((prev) => ({ ...prev, notes: e.target.value }))}
          className="min-h-[72px] rounded-m border border-outline-default p-3 text-sm"
        />
      </label>
      {!readOnly ? (
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            disabled={disabled}
            onClick={handleSaveDraft}
            className="rounded-full bg-surface-alt px-4 py-2 text-sm font-semibold text-content-default disabled:opacity-50"
          >
            {isSaving ? "Saving…" : "Save draft"}
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={handleSubmit}
            className="rounded-full bg-brand-default px-4 py-2 text-sm font-semibold text-content-on-brand disabled:opacity-50"
          >
            {isSubmitting ? "Submitting…" : "Submit scores"}
          </button>
        </div>
      ) : (
        <p className="text-sm font-semibold text-status-positive">Submitted — read only</p>
      )}
    </div>
  );
}
