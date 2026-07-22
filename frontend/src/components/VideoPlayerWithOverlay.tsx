"use client";

import { useEffect, useRef, useState } from "react";

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

/** Picks the evidence ref within `event.evidence_json.evidence` whose
 * `frame_ms` is closest to `timeMs`, per product principle #2 ("Every
 * detected event must include timestamps, confidence, evidence"). */
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
}

/** HTML5 video with a canvas overlay (per the recommended Frontend stack:
 * "HTML5 video" + "Canvas or SVG overlays") drawing the evidence bounding
 * box for whichever event is active at the current playhead position. */
export function VideoPlayerWithOverlay({
  src,
  events,
  onTimeUpdateMs,
  seekToMs,
}: VideoPlayerWithOverlayProps): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentMs, setCurrentMs] = useState(0);

  useEffect(() => {
    if (seekToMs == null || !videoRef.current) {
      return;
    }
    videoRef.current.currentTime = seekToMs / 1000;
  }, [seekToMs]);

  useEffect(() => {
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

    const active = activeEventAt(events, currentMs);
    if (!active) {
      return;
    }
    const evidence = closestEvidence(active, currentMs);
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
  }, [events, currentMs]);

  function handleTimeUpdate(): void {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    const ms = Math.round(video.currentTime * 1000);
    setCurrentMs(ms);
    onTimeUpdateMs?.(ms);
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
        ref={videoRef}
        src={src}
        controls
        className="h-full w-full"
        onTimeUpdate={handleTimeUpdate}
      />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0" />
    </div>
  );
}
