"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  confirmEvent,
  createEvent,
  deleteEvent,
  listEvents,
  rejectEvent,
  updateEvent,
} from "@/lib/api-client";
import type { AnalysisEventCreate, AnalysisEventUpdate } from "@/lib/types";

import { useInvalidateScore } from "@/hooks/useAnalysis";

export const eventsQueryKey = (analysisId: string) => ["analyses", analysisId, "events"] as const;

export function useEvents(analysisId: string, enabled: boolean) {
  return useQuery({
    queryKey: eventsQueryKey(analysisId),
    queryFn: () => listEvents(analysisId),
    enabled,
  });
}

function useEventsMutationEffects(analysisId: string) {
  const queryClient = useQueryClient();
  const invalidateScore = useInvalidateScore(analysisId);
  return () => {
    void queryClient.invalidateQueries({ queryKey: eventsQueryKey(analysisId) });
    void invalidateScore();
  };
}

export function useCreateEvent(analysisId: string) {
  const onSettled = useEventsMutationEffects(analysisId);
  return useMutation({
    mutationFn: (payload: AnalysisEventCreate) => createEvent(analysisId, payload),
    onSuccess: onSettled,
  });
}

export function useUpdateEvent(analysisId: string) {
  const onSettled = useEventsMutationEffects(analysisId);
  return useMutation({
    mutationFn: ({ eventId, payload }: { eventId: string; payload: AnalysisEventUpdate }) =>
      updateEvent(analysisId, eventId, payload),
    onSuccess: onSettled,
  });
}

export function useConfirmEvent(analysisId: string) {
  const onSettled = useEventsMutationEffects(analysisId);
  return useMutation({
    mutationFn: (eventId: string) => confirmEvent(analysisId, eventId),
    onSuccess: onSettled,
  });
}

export function useRejectEvent(analysisId: string) {
  const onSettled = useEventsMutationEffects(analysisId);
  return useMutation({
    mutationFn: (eventId: string) => rejectEvent(analysisId, eventId),
    onSuccess: onSettled,
  });
}

export function useDeleteEvent(analysisId: string) {
  const onSettled = useEventsMutationEffects(analysisId);
  return useMutation({
    mutationFn: (eventId: string) => deleteEvent(analysisId, eventId),
    onSuccess: onSettled,
  });
}
