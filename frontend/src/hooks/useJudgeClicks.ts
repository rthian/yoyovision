"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";

import { createJudgeClick, deleteJudgeClick } from "@/lib/api-client";
import type { JudgeAccessRead, JudgeClick, JudgeClickCreate } from "@/lib/types";

function patchVideoClicks(
  entry: JudgeAccessRead | undefined,
  entryVideoId: string,
  updater: (clicks: JudgeClick[]) => JudgeClick[]
): JudgeAccessRead | undefined {
  if (!entry) {
    return entry;
  }
  return {
    ...entry,
    videos: entry.videos.map((video) =>
      video.entry_video_id === entryVideoId
        ? { ...video, my_clicks: updater(video.my_clicks ?? []) }
        : video
    ),
  };
}

export function useCreateJudgeClick(token: string, entryVideoId: string) {
  const queryClient = useQueryClient();
  const queryKey = ["judgeAccess", token] as const;

  return useMutation({
    mutationFn: (payload: JudgeClickCreate) =>
      createJudgeClick(token, entryVideoId, payload),
    onSuccess: (newClick) => {
      queryClient.setQueryData<JudgeAccessRead>(queryKey, (entry) =>
        patchVideoClicks(entry, entryVideoId, (clicks) => [...clicks, newClick])
      );
    },
  });
}

export function useDeleteJudgeClick(token: string, entryVideoId: string) {
  const queryClient = useQueryClient();
  const queryKey = ["judgeAccess", token] as const;

  return useMutation({
    mutationFn: (clickId: string) => deleteJudgeClick(token, clickId),
    onSuccess: (_result, clickId) => {
      queryClient.setQueryData<JudgeAccessRead>(queryKey, (entry) =>
        patchVideoClicks(entry, entryVideoId, (clicks) =>
          clicks.filter((click) => click.id !== clickId)
        )
      );
    },
  });
}
