"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getJudgeAccess, submitJudgeFe, upsertJudgeFe } from "@/lib/api-client";
import type { JudgeFreestyleScoreUpsert } from "@/lib/types";

export function useJudgeAccess(token: string | undefined) {
  return useQuery({
    queryKey: ["judgeAccess", token],
    queryFn: () => getJudgeAccess(token!),
    enabled: Boolean(token),
    retry: false,
  });
}

export function useUpsertJudgeFe(token: string, entryVideoId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: JudgeFreestyleScoreUpsert) =>
      upsertJudgeFe(token, entryVideoId, payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["judgeAccess", token] });
    },
  });
}

export function useSubmitJudgeFe(token: string, entryVideoId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: JudgeFreestyleScoreUpsert) =>
      submitJudgeFe(token, entryVideoId, payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["judgeAccess", token] });
    },
  });
}
