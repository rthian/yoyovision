"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  createTrainingAnnotation,
  deleteTrainingAnnotation,
  listTrainingAnnotations,
  updateTrainingAnnotation,
} from "@/lib/api-client";
import type {
  TrainingAnnotationCreate,
  TrainingAnnotationUpdate,
} from "@/lib/types";

export const trainingAnnotationsQueryKey = (videoId: string) =>
  ["videos", videoId, "training-annotations"] as const;

export function useTrainingAnnotations(videoId: string, enabled: boolean) {
  return useQuery({
    queryKey: trainingAnnotationsQueryKey(videoId),
    queryFn: () => listTrainingAnnotations(videoId),
    enabled,
  });
}

function useRefreshAnnotations(videoId: string) {
  const queryClient = useQueryClient();
  return () =>
    queryClient.invalidateQueries({ queryKey: trainingAnnotationsQueryKey(videoId) });
}

export function useCreateTrainingAnnotation(videoId: string) {
  const refresh = useRefreshAnnotations(videoId);
  return useMutation({
    mutationFn: (payload: TrainingAnnotationCreate) =>
      createTrainingAnnotation(videoId, payload),
    onSuccess: refresh,
  });
}

export function useUpdateTrainingAnnotation(videoId: string) {
  const refresh = useRefreshAnnotations(videoId);
  return useMutation({
    mutationFn: ({
      annotationId,
      payload,
    }: {
      annotationId: string;
      payload: TrainingAnnotationUpdate;
    }) => updateTrainingAnnotation(videoId, annotationId, payload),
    onSuccess: refresh,
  });
}

export function useDeleteTrainingAnnotation(videoId: string) {
  const refresh = useRefreshAnnotations(videoId);
  return useMutation({
    mutationFn: (annotationId: string) => deleteTrainingAnnotation(videoId, annotationId),
    onSuccess: refresh,
  });
}
