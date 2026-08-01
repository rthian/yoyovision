"use client";

import { useEffect, useState } from "react";

import { fetchJudgeVideoBlobUrl } from "@/lib/api-client";

export function useJudgeVideoBlobUrl(
  token: string | undefined,
  entryVideoId: string | undefined,
  enabled: boolean
) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!enabled || !token || !entryVideoId) {
      return;
    }
    let objectUrl: string | null = null;
    let cancelled = false;

    setIsLoading(true);
    setError(null);
    fetchJudgeVideoBlobUrl(token, entryVideoId)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        objectUrl = url;
        setBlobUrl(url);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err : new Error("Failed to load video."));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [token, entryVideoId, enabled]);

  return { blobUrl, error, isLoading };
}
