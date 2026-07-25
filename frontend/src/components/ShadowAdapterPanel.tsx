"use client";

import { useState } from "react";

import {
  EMPTY_ADAPTER_FORM,
  formFromPresetId,
  SHADOW_ADAPTER_PRESETS,
  type AdapterFormState,
} from "@/lib/adapterPresets";
import type { PipelineAdapterConfig } from "@/lib/types";

const ADAPTER_OPTIONS = {
  pose: ["", "mock", "mediapipe"],
  hand: ["", "mock", "mediapipe"],
  yoyo: ["", "mock", "pytorch", "onnx"],
  tracker: ["", "mock", "kalman"],
  temporal: ["", "mock", "torch"],
} as const;

export function buildPipelineAdapterConfig(
  form: AdapterFormState
): PipelineAdapterConfig | null {
  const config: PipelineAdapterConfig = {};
  if (form.poseAdapter) config.pose_adapter = form.poseAdapter;
  if (form.handAdapter) config.hand_adapter = form.handAdapter;
  if (form.yoyoAdapter) config.yoyo_adapter = form.yoyoAdapter;
  if (form.trackerAdapter) config.tracker_adapter = form.trackerAdapter;
  if (form.temporalAdapter) config.temporal_event_adapter = form.temporalAdapter;

  const adapter_kwargs: Record<string, Record<string, unknown>> = {};
  if (form.temporalWeights.trim()) {
    adapter_kwargs.temporal_event = { weights_path: form.temporalWeights.trim() };
  }
  if (form.yoyoWeights.trim()) {
    const key = form.yoyoAdapter === "onnx" ? "model_path" : "weights_path";
    adapter_kwargs.yoyo = { [key]: form.yoyoWeights.trim() };
  }
  if (Object.keys(adapter_kwargs).length > 0) {
    config.adapter_kwargs = adapter_kwargs;
  }

  return Object.keys(config).length > 0 ? config : null;
}

export function summarizePipelineAdapterConfig(
  config: PipelineAdapterConfig | null | undefined
): string | null {
  if (!config) return null;
  const parts: string[] = [];
  if (config.pose_adapter) parts.push(`pose=${config.pose_adapter}`);
  if (config.hand_adapter) parts.push(`hand=${config.hand_adapter}`);
  if (config.yoyo_adapter) parts.push(`yoyo=${config.yoyo_adapter}`);
  if (config.tracker_adapter) parts.push(`tracker=${config.tracker_adapter}`);
  if (config.temporal_event_adapter) parts.push(`temporal=${config.temporal_event_adapter}`);
  return parts.length > 0 ? parts.join(", ") : null;
}

interface ShadowAdapterPanelProps {
  enabled: boolean;
  onConfigChange: (config: PipelineAdapterConfig | null) => void;
}

export function ShadowAdapterPanel({
  enabled,
  onConfigChange,
}: ShadowAdapterPanelProps): JSX.Element | null {
  const [open, setOpen] = useState(false);
  const [presetId, setPresetId] = useState("worker-default");
  const [form, setForm] = useState<AdapterFormState>(EMPTY_ADAPTER_FORM);

  function applyForm(next: AdapterFormState) {
    setForm(next);
    onConfigChange(buildPipelineAdapterConfig(next));
  }

  function updateField<K extends keyof AdapterFormState>(key: K, value: AdapterFormState[K]) {
    setPresetId("custom");
    applyForm({ ...form, [key]: value });
  }

  function applyPreset(nextPresetId: string) {
    setPresetId(nextPresetId);
    applyForm(formFromPresetId(nextPresetId));
  }

  if (!enabled) {
    return null;
  }

  const activePreset = SHADOW_ADAPTER_PRESETS.find((preset) => preset.id === presetId);
  const summary = summarizePipelineAdapterConfig(buildPipelineAdapterConfig(form));

  return (
    <div className="rounded-m border border-outline-soft bg-background-alt p-4">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between text-left text-sm font-semibold text-content-default"
      >
        Adapter overrides (optional)
        <span className="text-content-dim">{open ? "Hide" : "Show"}</span>
      </button>
      {open ? (
        <div className="mt-4 flex flex-col gap-3">
          <label className="flex flex-col gap-1 text-sm text-content-dim">
            Profile preset
            <select
              value={presetId}
              onChange={(event) => applyPreset(event.target.value)}
              className="rounded-s border border-outline-default bg-surface-default px-3 py-2 text-content-default"
            >
              {SHADOW_ADAPTER_PRESETS.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.label}
                </option>
              ))}
              {presetId === "custom" ? (
                <option value="custom">Custom (edited)</option>
              ) : null}
            </select>
          </label>
          {activePreset ? (
            <p className="text-xs text-content-dim">{activePreset.description}</p>
          ) : (
            <p className="text-xs text-content-dim">
              Custom overrides. Pick a preset to reset, or edit fields below.
            </p>
          )}
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1 text-sm text-content-dim">
              Pose adapter
              <select
                value={form.poseAdapter}
                onChange={(event) => updateField("poseAdapter", event.target.value)}
                className="rounded-s border border-outline-default bg-surface-default px-3 py-2 text-content-default"
              >
                {ADAPTER_OPTIONS.pose.map((value) => (
                  <option key={value || "default"} value={value}>
                    {value || "Worker default"}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-sm text-content-dim">
              Hand adapter
              <select
                value={form.handAdapter}
                onChange={(event) => updateField("handAdapter", event.target.value)}
                className="rounded-s border border-outline-default bg-surface-default px-3 py-2 text-content-default"
              >
                {ADAPTER_OPTIONS.hand.map((value) => (
                  <option key={value || "default"} value={value}>
                    {value || "Worker default"}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-sm text-content-dim">
              Yo-yo adapter
              <select
                value={form.yoyoAdapter}
                onChange={(event) => updateField("yoyoAdapter", event.target.value)}
                className="rounded-s border border-outline-default bg-surface-default px-3 py-2 text-content-default"
              >
                {ADAPTER_OPTIONS.yoyo.map((value) => (
                  <option key={value || "default"} value={value}>
                    {value || "Worker default"}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-sm text-content-dim">
              Tracker adapter
              <select
                value={form.trackerAdapter}
                onChange={(event) => updateField("trackerAdapter", event.target.value)}
                className="rounded-s border border-outline-default bg-surface-default px-3 py-2 text-content-default"
              >
                {ADAPTER_OPTIONS.tracker.map((value) => (
                  <option key={value || "default"} value={value}>
                    {value || "Worker default"}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-sm text-content-dim sm:col-span-2">
              Temporal event adapter
              <select
                value={form.temporalAdapter}
                onChange={(event) => updateField("temporalAdapter", event.target.value)}
                className="rounded-s border border-outline-default bg-surface-default px-3 py-2 text-content-default"
              >
                {ADAPTER_OPTIONS.temporal.map((value) => (
                  <option key={value || "default"} value={value}>
                    {value || "Worker default"}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-sm text-content-dim sm:col-span-2">
              TCN checkpoint path (container path)
              <input
                type="text"
                value={form.temporalWeights}
                onChange={(event) => updateField("temporalWeights", event.target.value)}
                placeholder="/models/tcn.pt"
                className="rounded-s border border-outline-default bg-surface-default px-3 py-2 text-content-default"
              />
            </label>
            <label className="flex flex-col gap-1 text-sm text-content-dim sm:col-span-2">
              Yo-yo weights path (container path)
              <input
                type="text"
                value={form.yoyoWeights}
                onChange={(event) => updateField("yoyoWeights", event.target.value)}
                placeholder="/models/yoyo.pt"
                className="rounded-s border border-outline-default bg-surface-default px-3 py-2 text-content-default"
              />
            </label>
          </div>
        </div>
      ) : null}
      {summary ? (
        <p role="status" className="mt-3 text-xs text-content-dim">
          Profile: {summary}
        </p>
      ) : null}
    </div>
  );
}
