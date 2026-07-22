"use client";

import { useState } from "react";

import { formatMsAsTimecode, titleCaseFromSnakeCase } from "@/lib/format";
import { DEDUCTION_TYPES } from "@/lib/types";
import type { DeductionType, MajorDeduction } from "@/lib/types";

import {
  useConfirmDeduction,
  useCreateDeduction,
  useDeleteDeduction,
  useRejectDeduction,
  useUpdateDeduction,
} from "@/hooks/useDeductions";

const REVIEW_STATUS_STYLES: Record<MajorDeduction["review_status"], string> = {
  pending: "bg-status-notice/15 text-status-notice",
  confirmed: "bg-status-positive/15 text-status-positive",
  rejected: "bg-status-alert/15 text-status-alert",
  edited: "bg-status-informative/15 text-status-informative",
};

interface DeductionTableProps {
  analysisId: string;
  deductions: MajorDeduction[];
  readOnly?: boolean;
}

/** Editor for `MajorDeduction` rows (yo-yo stop/change/detach, etc.) --
 * "multiple deductions per routine" and "manual override" per the Scoring
 * Requirements. */
export function DeductionTable({ analysisId, deductions, readOnly = false }: DeductionTableProps): JSX.Element {
  const updateDeduction = useUpdateDeduction(analysisId);
  const confirmDeduction = useConfirmDeduction(analysisId);
  const rejectDeduction = useRejectDeduction(analysisId);
  const deleteDeduction = useDeleteDeduction(analysisId);
  const createDeduction = useCreateDeduction(analysisId);

  const [isAdding, setIsAdding] = useState(false);
  const [newDeduction, setNewDeduction] = useState({
    type: "yoyo_stop" as DeductionType,
    timestamp_ms: 0,
    quantity: 1,
    points: 1,
  });

  async function handleAdd(): Promise<void> {
    await createDeduction.mutateAsync(newDeduction);
    setIsAdding(false);
    setNewDeduction({ type: "yoyo_stop", timestamp_ms: 0, quantity: 1, points: 1 });
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto rounded-m border border-outline-soft">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="bg-surface-alt text-xs uppercase text-content-dim">
            <tr>
              <th className="px-3 py-2">Time</th>
              <th className="px-3 py-2">Type</th>
              <th className="px-3 py-2">Qty</th>
              <th className="px-3 py-2">Points</th>
              <th className="px-3 py-2">Status</th>
              <th className="px-3 py-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {deductions.map((deduction) => (
              <tr key={deduction.id} className="border-t border-outline-softest">
                <td className="px-3 py-2 text-content-dim">
                  {formatMsAsTimecode(deduction.timestamp_ms)}
                </td>
                <td className="px-3 py-2">{titleCaseFromSnakeCase(deduction.type)}</td>
                <td className="px-3 py-2">
                  <input
                    type="number"
                    min={1}
                    disabled={readOnly}
                    defaultValue={deduction.quantity}
                    onBlur={(e) => {
                      const quantity = Number(e.target.value);
                      if (quantity !== deduction.quantity) {
                        updateDeduction.mutate({ deductionId: deduction.id, payload: { quantity } });
                      }
                    }}
                    className="w-16 rounded-s border border-transparent bg-transparent px-1 hover:border-outline-default focus:border-outline-default"
                  />
                </td>
                <td className="px-3 py-2">
                  <input
                    type="number"
                    min={0}
                    step={0.5}
                    disabled={readOnly}
                    defaultValue={deduction.points}
                    onBlur={(e) => {
                      const points = Number(e.target.value);
                      if (points !== deduction.points) {
                        updateDeduction.mutate({ deductionId: deduction.id, payload: { points } });
                      }
                    }}
                    className="w-16 rounded-s border border-transparent bg-transparent px-1 hover:border-outline-default focus:border-outline-default"
                  />
                </td>
                <td className="px-3 py-2">
                  <span
                    className={`rounded-s px-2 py-0.5 text-xs font-semibold ${REVIEW_STATUS_STYLES[deduction.review_status]}`}
                  >
                    {titleCaseFromSnakeCase(deduction.review_status)}
                  </span>
                </td>
                <td className="px-3 py-2">
                  {readOnly ? (
                    <span className="text-content-dim">—</span>
                  ) : (
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => confirmDeduction.mutate(deduction.id)}
                      className="text-status-positive hover:underline"
                    >
                      Confirm
                    </button>
                    <button
                      type="button"
                      onClick={() => rejectDeduction.mutate(deduction.id)}
                      className="text-status-notice hover:underline"
                    >
                      Reject
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        if (window.confirm("Delete this deduction?")) {
                          deleteDeduction.mutate(deduction.id);
                        }
                      }}
                      className="text-status-alert hover:underline"
                    >
                      Delete
                    </button>
                  </div>
                  )}
                </td>
              </tr>
            ))}
            {deductions.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-3 py-4 text-center text-content-dim">
                  No major deductions recorded.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      {readOnly ? null : isAdding ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void handleAdd();
          }}
          className="flex flex-wrap items-end gap-3 rounded-m border border-outline-soft bg-surface-default p-4"
        >
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            Type
            <select
              value={newDeduction.type}
              onChange={(e) =>
                setNewDeduction((prev) => ({ ...prev, type: e.target.value as DeductionType }))
              }
              className="h-9 rounded-s border border-outline-default px-2 text-sm"
            >
              {DEDUCTION_TYPES.map((type) => (
                <option key={type} value={type}>
                  {titleCaseFromSnakeCase(type)}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            Time (ms)
            <input
              type="number"
              min={0}
              value={newDeduction.timestamp_ms}
              onChange={(e) =>
                setNewDeduction((prev) => ({ ...prev, timestamp_ms: Number(e.target.value) }))
              }
              className="h-9 w-24 rounded-s border border-outline-default px-2 text-sm"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            Quantity
            <input
              type="number"
              min={1}
              value={newDeduction.quantity}
              onChange={(e) =>
                setNewDeduction((prev) => ({ ...prev, quantity: Number(e.target.value) }))
              }
              className="h-9 w-20 rounded-s border border-outline-default px-2 text-sm"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            Points
            <input
              type="number"
              min={0}
              step={0.5}
              value={newDeduction.points}
              onChange={(e) =>
                setNewDeduction((prev) => ({ ...prev, points: Number(e.target.value) }))
              }
              className="h-9 w-20 rounded-s border border-outline-default px-2 text-sm"
            />
          </label>
          <button
            type="submit"
            disabled={createDeduction.isPending}
            className="h-9 rounded-full bg-brand-primary px-4 text-sm font-semibold text-white disabled:opacity-60"
          >
            Add deduction
          </button>
          <button
            type="button"
            onClick={() => setIsAdding(false)}
            className="h-9 rounded-full px-4 text-sm font-semibold text-content-subtle hover:bg-surface-alt"
          >
            Cancel
          </button>
        </form>
      ) : (
        <button
          type="button"
          onClick={() => setIsAdding(true)}
          className="self-start rounded-full border border-outline-default px-4 py-2 text-sm font-semibold text-content-default hover:bg-surface-alt"
        >
          + Add deduction manually
        </button>
      )}
    </div>
  );
}
