"use client";

import { useQuery } from "@tanstack/react-query";

import { getJudgingEntryCalibration } from "@/lib/api-client";
import { formatMsAsTimecode } from "@/lib/format";

interface JudgingEntryCalibrationProps {
  entryId: string;
}

export function JudgingEntryCalibration({
  entryId,
}: JudgingEntryCalibrationProps): JSX.Element | null {
  const calibrationQuery = useQuery({
    queryKey: ["judgingEntryCalibration", entryId],
    queryFn: () => getJudgingEntryCalibration(entryId),
    enabled: Boolean(entryId),
  });

  if (calibrationQuery.isLoading) {
    return <p className="text-sm text-content-dim">Loading click calibration…</p>;
  }
  if (calibrationQuery.isError || !calibrationQuery.data) {
    return null;
  }

  const data = calibrationQuery.data;
  if (data.click_mode === "off") {
    return (
      <p className="text-sm text-content-dim">
        Clicker is off for this entry. Set click mode to training or technical to enable.
      </p>
    );
  }

  return (
    <section className="flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold text-content-default">Click calibration</h2>
        <p className="text-sm text-content-dim">
          Mode: {data.click_mode} · tolerance {data.tolerance_ms}ms
        </p>
      </div>

      {data.videos.map((video) => (
        <div key={video.entry_video_id} className="rounded-m border border-outline-soft">
          <div className="border-b border-outline-soft bg-surface-alt px-4 py-3">
            <p className="font-semibold text-content-default">{video.original_filename}</p>
            <p className="text-xs text-content-dim">
              Model events: {video.model_event_count} · panel clicks: {video.panel_click_count}
              {video.panel_mean_clicks !== null
                ? ` · mean ${video.panel_mean_clicks.toFixed(1)} per judge`
                : ""}
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-outline-soft text-left text-xs text-content-dim">
                  <th className="px-4 py-2">Judge</th>
                  <th className="px-2 py-2">Clicks</th>
                  <th className="px-2 py-2">Precision</th>
                  <th className="px-2 py-2">Recall</th>
                  <th className="px-2 py-2">Mean boundary err.</th>
                </tr>
              </thead>
              <tbody>
                {video.judges.map((judge) => (
                  <tr key={judge.assignment_id} className="border-b border-outline-softest">
                    <td className="px-4 py-2">{judge.display_name}</td>
                    <td className="px-2 py-2 tabular-nums">{judge.click_count}</td>
                    <td className="px-2 py-2 tabular-nums">
                      {judge.precision !== null ? judge.precision.toFixed(2) : "—"}
                    </td>
                    <td className="px-2 py-2 tabular-nums">
                      {judge.recall !== null ? judge.recall.toFixed(2) : "—"}
                    </td>
                    <td className="px-2 py-2 tabular-nums">
                      {judge.mean_boundary_error_ms !== null
                        ? `${judge.mean_boundary_error_ms.toFixed(0)} ms`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {video.judges.some((j) => j.matches.length > 0) ? (
            <ul className="divide-y divide-outline-softest px-4 py-2 text-xs text-content-subtle">
              {video.judges.flatMap((judge) =>
                judge.matches.map((match) => (
                  <li key={match.click_id} className="py-1">
                    {judge.display_name}: {formatMsAsTimecode(match.timestamp_ms)}
                    {match.matched_event_label
                      ? ` → ${match.matched_event_label}`
                      : " · unmatched"}
                    {match.boundary_error_ms !== null
                      ? ` (${match.boundary_error_ms > 0 ? "+" : ""}${match.boundary_error_ms} ms)`
                      : ""}
                  </li>
                ))
              )}
            </ul>
          ) : null}
        </div>
      ))}

      {data.warnings.length > 0 ? (
        <ul className="list-disc px-6 text-xs text-status-notice-boldest">
          {data.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
