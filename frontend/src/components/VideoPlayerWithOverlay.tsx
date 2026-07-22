"use client";

import { useCallback, useEffect, useRef } from "react";

import type { AnalysisEvent } from "@/lib/types";

interface EvidenceRef {
  frame_ms: number;
  bbox: { x: number; y: number; width: number; height: number } | null;
  keypoint_refs: string[];
  note: string;
}

function activeEventAt(events: AnalysisEvent[], timeMs: number): AnalysisEvent | null {
  return (
    events.find((event) => timeMs >= event.start_ms && timeMs <= event.end_ms) ?? null
  );
}

function closestEvidence(event: AnalysisEvent, timeMs: number): EvidenceRef | null {
  const evidence = event.evidence_json.evidence;
  if (!Array.isArray(evidence) || evidence.length === 0) {
    return null;
  }
  let best: EvidenceRef | null = null;
  let bestDistance = Infinity;
  for (const raw of evidence as EvidenceRef[]) {
    const distance = Math.abs(raw.frame_ms - timeMs);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = raw;
    }
  }
  return best;
}

interface VideoPlayerWithOverlayProps {
  src: string | null;
  events: AnalysisEvent[];
  onTimeUpdateMs?: (ms: number) => void;
  seekToMs?: number | null;
  routineStartMs?: number;
  routineEndMs?: number;
}

export function VideoPlayerWithOverlay({
  src,
  events,
  onTimeUpdateMs,
  seekToMs,
  routineStartMs = 0,
  routineEndMs,
}: VideoPlayerWithOverlayProps): JSX.Element {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const currentMsRef = useRef(0);
  const onTimeUpdateRef = useRef(onTimeUpdateMs);
  const eventsRef = useRef(events);
  const detachVideoListenersRef = useRef<(() => void) | null>(null);
  const routineStartRef = useRef(routineStartMs);
  const routineEndRef = useRef(routineEndMs ?? Number.MAX_SAFE_INTEGER);

  useEffect(() => {
    routineStartRef.current = routineStartMs;
    routineEndRef.current = routineEndMs ?? Number.MAX_SAFE_INTEGER;
  }, [routineStartMs, routineEndMs]);

  useEffect(() => {
    onTimeUpdateRef.current = onTimeUpdateMs;
  }, [onTimeUpdateMs]);

  useEffect(() => {
    eventsRef.current = events;
    drawOverlay(currentMsRef.current);
  }, [events]);

  function drawOverlay(timeMs: number): void {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) {
      return;
    }
    const { width, height } = container.getBoundingClientRect();
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, width, height);

    const active = activeEventAt(eventsRef.current, timeMs);
    if (!active) {
      return;
    }
    const evidence = closestEvidence(active, timeMs);
    if (!evidence?.bbox) {
      return;
    }
    const { x, y, width: bw, height: bh } = evidence.bbox;
    ctx.strokeStyle = "#00b14f";
    ctx.lineWidth = 2;
    ctx.strokeRect(x * width, y * height, bw * width, bh * height);

    ctx.font = "12px Inter, sans-serif";
    const label = `${active.label} (${Math.round(active.confidence * 100)}%)`;
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = "#00b14f";
    ctx.fillRect(x * width, Math.max(0, y * height - 18), textWidth + 8, 18);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x * width + 4, Math.max(12, y * height - 4));
  }

  function publishTime(ms: number, video?: HTMLVideoElement | null): void {
    let nextMs = ms;
    const activeVideo = video ?? videoRef.current;
    const routineEnd = routineEndRef.current;
    if (nextMs < routineStartRef.current) {
      nextMs = routineStartRef.current;
      if (activeVideo && activeVideo.currentTime * 1000 < routineStartRef.current) {
        activeVideo.currentTime = routineStartRef.current / 1000;
      }
    }
    if (routineEnd < Number.MAX_SAFE_INTEGER && nextMs > routineEnd) {
      nextMs = routineEnd;
      if (activeVideo && !activeVideo.paused) {
        activeVideo.pause();
        activeVideo.currentTime = routineEnd / 1000;
      }
    }
    currentMsRef.current = nextMs;
    onTimeUpdateRef.current?.(nextMs);
    drawOverlay(nextMs);
  }

  const attachVideoListeners = useCallback((video: HTMLVideoElement | null): void => {
    detachVideoListenersRef.current?.();
    detachVideoListenersRef.current = null;
    videoRef.current = video;

    if (!video) {
      return;
    }

    let frameId = 0;
    const syncWhilePlaying = (): void => {
      publishTime(Math.round(video.currentTime * 1000), video);
      if (!video.paused && !video.ended) {
        frameId = window.requestAnimationFrame(syncWhilePlaying);
      }
    };

    const onPlay = (): void => {
      if (video.currentTime * 1000 < routineStartRef.current) {
        video.currentTime = routineStartRef.current / 1000;
      }
      window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(syncWhilePlaying);
    };

    const onTimeChanged = (): void => {
      window.cancelAnimationFrame(frameId);
      publishTime(Math.round(video.currentTime * 1000), video);
    };

    video.addEventListener("play", onPlay);
    video.addEventListener("playing", onPlay);
    video.addEventListener("pause", onTimeChanged);
    video.addEventListener("ended", onTimeChanged);
    video.addEventListener("seeked", onTimeChanged);
    video.addEventListener("timeupdate", onTimeChanged);

    if (!video.paused && !video.ended) {
      onPlay();
    } else {
      publishTime(Math.round(video.currentTime * 1000));
    }

    detachVideoListenersRef.current = () => {
      window.cancelAnimationFrame(frameId);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("playing", onPlay);
      video.removeEventListener("pause", onTimeChanged);
      video.removeEventListener("ended", onTimeChanged);
      video.removeEventListener("seeked", onTimeChanged);
      video.removeEventListener("timeupdate", onTimeChanged);
    };
  }, []);

  useEffect(() => {
    return () => {
      detachVideoListenersRef.current?.();
    };
  }, []);

  useEffect(() => {
    if (seekToMs == null || !videoRef.current) {
      return;
    }
    videoRef.current.currentTime = seekToMs / 1000;
    publishTime(seekToMs);
  }, [seekToMs]);

  function handleTimeUpdate(event: React.SyntheticEvent<HTMLVideoElement>): void {
    publishTime(Math.round(event.currentTarget.currentTime * 1000));
  }

  if (!src) {
    return (
      <div className="flex aspect-video items-center justify-center rounded-m bg-content-default text-sm text-white">
        Loading video...
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative aspect-video overflow-hidden rounded-m bg-black">
      <video
        ref={attachVideoListeners}
        src={src}
        controls
        className="h-full w-full"
        onTimeUpdate={handleTimeUpdate}
      />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0" />
    </div>
  );
}
