"use client";

import Link from "next/link";

import { formatMsAsTimecode, titleCaseFromSnakeCase } from "@/lib/format";
import { eventInRoutine } from "@/lib/routine-window";
import type { TrainingAnnotation } from "@/lib/types";

interface AnalysisTrainingContextProps {
  videoId: string;
  annotations: TrainingAnnotation[];
  isLoading: boolean;
  routineWindow: { startMs: number; endMs: number } | null;
  modelVersions: Record<string, string> | null;
  onSeek: (ms: number) => void;
}

export function AnalysisTrainingContext({
  videoId,
  annotations,
  isLoading,
  routineWindow,
  modelVersions,
  onSeek,
}: AnalysisTrainingContextProps): JSX.Element {
  const visibleAnnotations = routineWindow
    ? annotations.filter((annotation) => eventInRoutine(annotation, routineWindow))
    : annotations;
  const modelNames = modelVersions ? Object.values(modelVersions) : [];

  return (
    <section className="rounded-m border border-outline-soft bg-surface-default p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="font-semibold text-content-default">Human training annotations</h2>
            <span className="rounded-full bg-status-informative/10 px-2 py-0.5 text-xs font-semibold text-status-informative">
              {isLoading ? "Loading…" : `${visibleAnnotations.length} in routine`}
            </span>
          </div>
          <p className="mt-1 max-w-3xl text-sm text-content-dim">
            These are ground-truth labels created by a person. They are shown for comparison,
            but they do not alter this analysis run. Using labels from the same video as answers
            would make the model evaluation unreliable.
          </p>
        </div>
        <Link
          href={`/videos/${videoId}`}
          className="shrink-0 rounded-full border border-outline-default px-4 py-2 text-sm font-semibold text-content-default hover:bg-surface-alt"
        >
          Edit annotations
        </Link>
      </div>

      {visibleAnnotations.length > 0 ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {visibleAnnotations.map((annotation) => (
            <button
              key={annotation.id}
              type="button"
              onClick={() => onSeek(annotation.start_ms)}
              title={`Seek to ${annotation.label}`}
              className="rounded-full border border-status-informative/25 bg-status-informative/5 px-3 py-1.5 text-left text-xs text-content-subtle hover:bg-status-informative/10"
            >
              <span className="font-semibold text-status-informative">
                {formatMsAsTimecode(annotation.start_ms)}
              </span>{" "}
              {annotation.label} · {titleCaseFromSnakeCase(annotation.element_type)}
            </button>
          ))}
        </div>
      ) : !isLoading ? (
        <p className="mt-4 text-sm text-content-dim">
          No human annotations fall inside the selected routine window.
        </p>
      ) : null}

      <div className="mt-4 rounded-s bg-surface-alt px-3 py-2 text-xs text-content-dim">
        <strong className="text-content-subtle">How annotations affect future analyses:</strong>{" "}
        export reviewed labels to the training corpus, train and validate a new temporal-event
        model, register its version, then rerun analysis with that model.
        {modelNames.length > 0 ? ` This run reports: ${modelNames.join(", ")}.` : " This run does not report a trained model version."}
      </div>
    </section>
  );
}
