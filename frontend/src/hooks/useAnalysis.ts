"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  getAnalysis,
  getScore,
  getScoreLineItems,
  reopenAnalysis,
  recomputeScore,
  submitAnalysis,
  updateAnalysisRuleset,
  updateRoutineWindow,
} from "@/lib/api-client";

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

export function useUpdateRoutineWindow(analysisId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: { routine_start_ms?: number | null; routine_end_ms?: number | null }) =>
      updateRoutineWindow(analysisId, payload),
    onSuccess: (job) => {
      queryClient.setQueryData(analysisQueryKey(analysisId), job);
      void queryClient.invalidateQueries({ queryKey: scoreQueryKey(analysisId) });
      void queryClient.invalidateQueries({ queryKey: scoreLineItemsQueryKey(analysisId) });
    },
  });
}

export function useSubmitAnalysis(analysisId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => submitAnalysis(analysisId),
    onSuccess: (job) => {
      queryClient.setQueryData(analysisQueryKey(analysisId), job);
    },
  });
}

export function useReopenAnalysis(analysisId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => reopenAnalysis(analysisId),
    onSuccess: (job) => {
      queryClient.setQueryData(analysisQueryKey(analysisId), job);
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


export function useUpdateAnalysisRuleset(analysisId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (rulesetVersion: string) => updateAnalysisRuleset(analysisId, rulesetVersion),
    onSuccess: (job) => {
      queryClient.setQueryData(analysisQueryKey(analysisId), job);
      void queryClient.invalidateQueries({ queryKey: scoreQueryKey(analysisId) });
      void queryClient.invalidateQueries({ queryKey: scoreLineItemsQueryKey(analysisId) });
    },
  });
}
