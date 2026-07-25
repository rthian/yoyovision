import { describe, expect, it } from "vitest";

import {
  formFromPresetId,
  pipelineConfigToFormState,
  SHADOW_ADAPTER_PRESETS,
} from "@/lib/adapterPresets";

describe("adapterPresets", () => {
  it("includes the full ML stack preset with model paths", () => {
    const form = formFromPresetId("full-ml-stack");
    expect(form.poseAdapter).toBe("mediapipe");
    expect(form.temporalWeights).toBe("/models/tcn.pt");
    expect(form.yoyoWeights).toBe("/models/yoyo.pt");
  });

  it("falls back to empty form for unknown preset ids", () => {
    expect(formFromPresetId("missing")).toEqual(
      formFromPresetId("worker-default")
    );
  });

  it("round-trips pipeline config into form state", () => {
    expect(
      pipelineConfigToFormState({
        yoyo_adapter: "onnx",
        adapter_kwargs: {
          yoyo: { model_path: "/models/yoyo.onnx" },
          temporal_event: { weights_path: "/models/tcn.pt" },
        },
        temporal_event_adapter: "torch",
      })
    ).toEqual({
      poseAdapter: "",
      handAdapter: "",
      yoyoAdapter: "onnx",
      trackerAdapter: "",
      temporalAdapter: "torch",
      temporalWeights: "/models/tcn.pt",
      yoyoWeights: "/models/yoyo.onnx",
    });
  });

  it("exposes stable preset ids", () => {
    expect(SHADOW_ADAPTER_PRESETS.map((preset) => preset.id)).toEqual([
      "worker-default",
      "mediapipe-perception",
      "torch-temporal",
      "full-ml-stack",
    ]);
  });
});
