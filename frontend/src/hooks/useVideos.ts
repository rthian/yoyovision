"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  cancelAnalysis,
  deleteAnalysis,
  deleteVideo,
  getVideo,
  listVideoAnalyses,
  listVideos,
  triggerVideoAnalysis,
  uploadVideo,
} from "@/lib/api-client";

export const videosQueryKey = ["videos"] as const;
export const videoQueryKey = (videoId: string) => ["videos", videoId] as const;
export const videoAnalysesQueryKey = (videoId: string) =>
  ["videos", videoId, "analyses"] as const;

export function useVideos(enabled: boolean) {
  return useQuery({
    queryKey: videosQueryKey,
    queryFn: listVideos,
    enabled,
  });
}

export function useVideo(videoId: string, enabled: boolean) {
  return useQuery({
    queryKey: videoQueryKey(videoId),
    queryFn: () => getVideo(videoId),
    enabled,
  });
}

export function useVideoAnalyses(videoId: string, enabled: boolean) {
  return useQuery({
    queryKey: videoAnalysesQueryKey(videoId),
    queryFn: () => listVideoAnalyses(videoId),
    enabled,
    // Analysis jobs move through pending -> running -> completed/failed
    // asynchronously in a worker process; poll while any job is unresolved.
    refetchInterval: (query) => {
      const jobs = query.state.data;
      const hasActiveJob = jobs?.some(
        (job) => job.status === "pending" || job.status === "running"
      );
      return hasActiveJob ? 3000 : false;
    },
  });
}

export function useUploadVideo() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: uploadVideo,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: videosQueryKey });
    },
  });
}

export function useDeleteVideo() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ videoId, hard }: { videoId: string; hard?: boolean }) =>
      deleteVideo(videoId, hard),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: videosQueryKey });
    },
  });
}

export function useTriggerVideoAnalysis(videoId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (options?: { shadow?: boolean }) => triggerVideoAnalysis(videoId, options),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: videoAnalysesQueryKey(videoId) });
    },
  });
}

export function useCancelAnalysis(videoId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (analysisId: string) => cancelAnalysis(analysisId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: videoAnalysesQueryKey(videoId) });
    },
  });
}

export function useDeleteAnalysis(videoId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (analysisId: string) => deleteAnalysis(analysisId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: videoAnalysesQueryKey(videoId) });
    },
  });
}
