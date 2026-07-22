import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ExportButtons } from "@/components/ExportButtons";
import * as apiClient from "@/lib/api-client";

describe("ExportButtons", () => {
  let createObjectUrlSpy: ReturnType<typeof vi.fn>;
  let revokeObjectUrlSpy: ReturnType<typeof vi.fn>;
  let clickSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    createObjectUrlSpy = vi.fn(() => "blob:mock-url");
    revokeObjectUrlSpy = vi.fn();
    vi.stubGlobal("URL", { createObjectURL: createObjectUrlSpy, revokeObjectURL: revokeObjectUrlSpy });
    clickSpy = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(clickSpy);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("downloads the JSON report using the server-provided filename", async () => {
    const blob = new Blob(["{}"], { type: "application/json" });
    vi.spyOn(apiClient, "exportReportJson").mockResolvedValue({ blob, filename: "report.json" });

    render(<ExportButtons analysisId="analysis-1" />);
    fireEvent.click(screen.getByRole("button", { name: /Export report \(JSON\)/ }));

    await waitFor(() => expect(clickSpy).toHaveBeenCalledTimes(1));
    expect(createObjectUrlSpy).toHaveBeenCalledWith(blob);
    expect(revokeObjectUrlSpy).toHaveBeenCalledWith("blob:mock-url");
  });

  it("falls back to a default filename when the server omits Content-Disposition", async () => {
    const blob = new Blob(["a,b"], { type: "text/csv" });
    vi.spyOn(apiClient, "exportEventsCsv").mockResolvedValue({ blob, filename: null });

    render(<ExportButtons analysisId="analysis-1" />);
    fireEvent.click(screen.getByRole("button", { name: /Export events \(CSV\)/ }));

    await waitFor(() => expect(clickSpy).toHaveBeenCalledTimes(1));
  });

  it("disables the other export buttons while one export is pending", async () => {
    let resolveExport: (value: { blob: Blob; filename: string | null }) => void = () => {};
    vi.spyOn(apiClient, "exportDeductionsCsv").mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveExport = resolve;
        })
    );

    render(<ExportButtons analysisId="analysis-1" />);
    fireEvent.click(screen.getByRole("button", { name: /Export deductions \(CSV\)/ }));

    expect(screen.getByRole("button", { name: /Export report \(JSON\)/ })).toBeDisabled();

    resolveExport({ blob: new Blob(["a"]), filename: "deductions.csv" });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Export report \(JSON\)/ })).not.toBeDisabled()
    );
  });
});
