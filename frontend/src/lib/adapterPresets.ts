import type { PipelineAdapterConfig } from "@/lib/types";

export interface AdapterFormState {
  poseAdapter: string;
  handAdapter: string;
  yoyoAdapter: string;
  trackerAdapter: string;
  temporalAdapter: string;
  temporalWeights: string;
  yoyoWeights: string;
}

export const EMPTY_ADAPTER_FORM: AdapterFormState = {
  poseAdapter: "",
  handAdapter: "",
  yoyoAdapter: "",
  trackerAdapter: "",
  temporalAdapter: "",
  temporalWeights: "",
  yoyoWeights: "",
};

export interface AdapterPreset {
  id: string;
  label: string;
  description: string;
  form: AdapterFormState;
}

/** Built-in shadow-run adapter profiles. Paths assume worker `/models` mount. */
export const SHADOW_ADAPTER_PRESETS: AdapterPreset[] = [
  {
    id: "worker-default",
    label: "Worker default",
    description: "No per-job overrides; use worker environment defaults.",
    form: EMPTY_ADAPTER_FORM,
  },
  {
    id: "mediapipe-perception",
    label: "MediaPipe perception",
    description: "Real pose and hand extraction with Kalman yo-yo tracking.",
    form: {
      poseAdapter: "mediapipe",
      handAdapter: "mediapipe",
      yoyoAdapter: "",
      trackerAdapter: "kalman",
      temporalAdapter: "",
      temporalWeights: "",
      yoyoWeights: "",
    },
  },
  {
    id: "torch-temporal",
    label: "Torch TCN",
    description: "Temporal trick detector with a trained TCN checkpoint.",
    form: {
      poseAdapter: "",
      handAdapter: "",
      yoyoAdapter: "",
      trackerAdapter: "",
      temporalAdapter: "torch",
      temporalWeights: "/models/tcn.pt",
      yoyoWeights: "",
    },
  },
  {
    id: "full-ml-stack",
    label: "Full ML stack",
    description: "MediaPipe + PyTorch yo-yo + Kalman tracker + Torch TCN.",
    form: {
      poseAdapter: "mediapipe",
      handAdapter: "mediapipe",
      yoyoAdapter: "pytorch",
      trackerAdapter: "kalman",
      temporalAdapter: "torch",
      temporalWeights: "/models/tcn.pt",
      yoyoWeights: "/models/yoyo.pt",
    },
  },
];

export function formFromPresetId(presetId: string): AdapterFormState {
  const preset = SHADOW_ADAPTER_PRESETS.find((entry) => entry.id === presetId);
  return preset ? { ...preset.form } : { ...EMPTY_ADAPTER_FORM };
}

export function pipelineConfigToFormState(
  config: PipelineAdapterConfig | null | undefined
): AdapterFormState {
  if (!config) {
    return { ...EMPTY_ADAPTER_FORM };
  }

  const temporalWeights =
    (config.adapter_kwargs?.temporal_event?.weights_path as string | undefined) ?? "";
  const yoyoKwargs = config.adapter_kwargs?.yoyo;
  const yoyoWeights =
    (yoyoKwargs?.weights_path as string | undefined) ??
    (yoyoKwargs?.model_path as string | undefined) ??
    "";

  return {
    poseAdapter: config.pose_adapter ?? "",
    handAdapter: config.hand_adapter ?? "",
    yoyoAdapter: config.yoyo_adapter ?? "",
    trackerAdapter: config.tracker_adapter ?? "",
    temporalAdapter: config.temporal_event_adapter ?? "",
    temporalWeights,
    yoyoWeights,
  };
}
