"use client";

import { useState } from "react";

import { exportDeductionsCsv, exportEventsCsv, exportReportJson } from "@/lib/api-client";

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

interface ExportButtonsProps {
  analysisId: string;
}

/** JSON and CSV export per MVP scope. Filenames come from the server's
 * `Content-Disposition` header (already sanitized server-side via
 * `sanitize_export_filename`), with a sane fallback if that header is
 * somehow missing. */
export function ExportButtons({ analysisId }: ExportButtonsProps): JSX.Element {
  const [pendingExport, setPendingExport] = useState<string | null>(null);

  async function handleExport(
    key: string,
    exportFn: () => Promise<{ blob: Blob; filename: string | null }>,
    fallbackFilename: string
  ): Promise<void> {
    setPendingExport(key);
    try {
      const { blob, filename } = await exportFn();
      triggerDownload(blob, filename ?? fallbackFilename);
    } finally {
      setPendingExport(null);
    }
  }

  return (
    <div className="flex flex-wrap gap-2">
      <button
        type="button"
        disabled={pendingExport !== null}
        onClick={() =>
          handleExport(
            "report",
            () => exportReportJson(analysisId),
            `yoyovision-analysis-${analysisId}.json`
          )
        }
        className="rounded-full border border-outline-default px-4 py-2 text-sm font-semibold text-content-default hover:bg-surface-alt disabled:opacity-60"
      >
        {pendingExport === "report" ? "Exporting..." : "Export report (JSON)"}
      </button>
      <button
        type="button"
        disabled={pendingExport !== null}
        onClick={() =>
          handleExport(
            "events",
            () => exportEventsCsv(analysisId),
            `yoyovision-events-${analysisId}.csv`
          )
        }
        className="rounded-full border border-outline-default px-4 py-2 text-sm font-semibold text-content-default hover:bg-surface-alt disabled:opacity-60"
      >
        {pendingExport === "events" ? "Exporting..." : "Export events (CSV)"}
      </button>
      <button
        type="button"
        disabled={pendingExport !== null}
        onClick={() =>
          handleExport(
            "deductions",
            () => exportDeductionsCsv(analysisId),
            `yoyovision-deductions-${analysisId}.csv`
          )
        }
        className="rounded-full border border-outline-default px-4 py-2 text-sm font-semibold text-content-default hover:bg-surface-alt disabled:opacity-60"
      >
        {pendingExport === "deductions" ? "Exporting..." : "Export deductions (CSV)"}
      </button>
    </div>
  );
}
