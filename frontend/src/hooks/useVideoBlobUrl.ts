"use client";

import { useEffect, useState } from "react";

import { fetchVideoBlobUrl } from "@/lib/api-client";

/** Fetches a video's bytes once (via the authenticated `/videos/{id}/stream`
 * endpoint) and exposes a local object URL suitable for a `<video src>`.
 * Revokes the URL on unmount/videoId change to avoid leaking memory across
 * review sessions. */
export function useVideoBlobUrl(videoId: string | undefined, enabled: boolean) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!enabled || !videoId) {
      return;
    }
    let objectUrl: string | null = null;
    let cancelled = false;

    setIsLoading(true);
    setError(null);
    fetchVideoBlobUrl(videoId)
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
  }, [videoId, enabled]);

  return { blobUrl, error, isLoading };
}
