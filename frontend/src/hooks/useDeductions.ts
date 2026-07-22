"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  confirmDeduction,
  createDeduction,
  deleteDeduction,
  listDeductions,
  rejectDeduction,
  updateDeduction,
} from "@/lib/api-client";
import type { MajorDeductionCreate, MajorDeductionUpdate } from "@/lib/types";

import { useInvalidateScore } from "@/hooks/useAnalysis";

export const deductionsQueryKey = (analysisId: string) =>
  ["analyses", analysisId, "deductions"] as const;

export function useDeductions(analysisId: string, enabled: boolean) {
  return useQuery({
    queryKey: deductionsQueryKey(analysisId),
    queryFn: () => listDeductions(analysisId),
    enabled,
  });
}

function useDeductionsMutationEffects(analysisId: string) {
  const queryClient = useQueryClient();
  const invalidateScore = useInvalidateScore(analysisId);
  return () => {
    void queryClient.invalidateQueries({ queryKey: deductionsQueryKey(analysisId) });
    void invalidateScore();
  };
}

export function useCreateDeduction(analysisId: string) {
  const onSettled = useDeductionsMutationEffects(analysisId);
  return useMutation({
    mutationFn: (payload: MajorDeductionCreate) => createDeduction(analysisId, payload),
    onSuccess: onSettled,
  });
}

export function useUpdateDeduction(analysisId: string) {
  const onSettled = useDeductionsMutationEffects(analysisId);
  return useMutation({
    mutationFn: ({
      deductionId,
      payload,
    }: {
      deductionId: string;
      payload: MajorDeductionUpdate;
    }) => updateDeduction(analysisId, deductionId, payload),
    onSuccess: onSettled,
  });
}

export function useConfirmDeduction(analysisId: string) {
  const onSettled = useDeductionsMutationEffects(analysisId);
  return useMutation({
    mutationFn: (deductionId: string) => confirmDeduction(analysisId, deductionId),
    onSuccess: onSettled,
  });
}

export function useRejectDeduction(analysisId: string) {
  const onSettled = useDeductionsMutationEffects(analysisId);
  return useMutation({
    mutationFn: (deductionId: string) => rejectDeduction(analysisId, deductionId),
    onSuccess: onSettled,
  });
}

export function useDeleteDeduction(analysisId: string) {
  const onSettled = useDeductionsMutationEffects(analysisId);
  return useMutation({
    mutationFn: (deductionId: string) => deleteDeduction(analysisId, deductionId),
    onSuccess: onSettled,
  });
}
