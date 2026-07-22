"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getEvaluation, upsertEvaluation } from "@/lib/api-client";
import type { FreestyleEvaluationUpsert } from "@/lib/types";

import { useInvalidateScore } from "@/hooks/useAnalysis";

export const evaluationQueryKey = (analysisId: string) =>
  ["analyses", analysisId, "evaluation"] as const;

export function useEvaluation(analysisId: string, enabled: boolean) {
  return useQuery({
    queryKey: evaluationQueryKey(analysisId),
    queryFn: () => getEvaluation(analysisId),
    enabled,
  });
}

export function useUpsertEvaluation(analysisId: string) {
  const queryClient = useQueryClient();
  const invalidateScore = useInvalidateScore(analysisId);
  return useMutation({
    mutationFn: (payload: FreestyleEvaluationUpsert) => upsertEvaluation(analysisId, payload),
    onSuccess: (evaluation) => {
      queryClient.setQueryData(evaluationQueryKey(analysisId), evaluation);
      void invalidateScore();
    },
  });
}
