"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getAnalysis, getScore, getScoreLineItems, recomputeScore } from "@/lib/api-client";

export const analysisQueryKey = (analysisId: string) => ["analyses", analysisId] as const;
export const scoreQueryKey = (analysisId: string) =>
  ["analyses", analysisId, "score"] as const;
export const scoreLineItemsQueryKey = (analysisId: string) =>
  ["analyses", analysisId, "score", "line-items"] as const;

export function useAnalysisJob(analysisId: string, enabled: boolean) {
  return useQuery({
    queryKey: analysisQueryKey(analysisId),
    queryFn: () => getAnalysis(analysisId),
    enabled,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "pending" || status === "running" ? 3000 : false;
    },
  });
}

export function useScore(analysisId: string, enabled: boolean) {
  return useQuery({
    queryKey: scoreQueryKey(analysisId),
    queryFn: () => getScore(analysisId),
    enabled,
  });
}

export function useScoreLineItems(analysisId: string, enabled: boolean) {
  return useQuery({
    queryKey: scoreLineItemsQueryKey(analysisId),
    queryFn: () => getScoreLineItems(analysisId),
    enabled,
  });
}

export function useRecomputeScore(analysisId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => recomputeScore(analysisId),
    onSuccess: (score) => {
      queryClient.setQueryData(scoreQueryKey(analysisId), score);
      void queryClient.invalidateQueries({ queryKey: scoreLineItemsQueryKey(analysisId) });
    },
  });
}

/** Invalidates score + line-item queries for `analysisId`; every event/deduction/
 * evaluation mutation already recomputes the score server-side (see the API
 * routers' docstrings), so the frontend only needs to refetch, not resend. */
export function useInvalidateScore(analysisId: string) {
  const queryClient = useQueryClient();
  return () => {
    void queryClient.invalidateQueries({ queryKey: scoreQueryKey(analysisId) });
    void queryClient.invalidateQueries({ queryKey: scoreLineItemsQueryKey(analysisId) });
  };
}
