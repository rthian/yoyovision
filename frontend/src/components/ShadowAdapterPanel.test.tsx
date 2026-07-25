import { describe, expect, it } from "vitest";

import {
  buildPipelineAdapterConfig,
  summarizePipelineAdapterConfig,
} from "@/components/ShadowAdapterPanel";

describe("buildPipelineAdapterConfig", () => {
  it("returns null when no overrides are set", () => {
    expect(
      buildPipelineAdapterConfig({
        poseAdapter: "",
        handAdapter: "",
        yoyoAdapter: "",
        trackerAdapter: "",
        temporalAdapter: "",
        temporalWeights: "",
        yoyoWeights: "",
      })
    ).toBeNull();
  });

  it("maps adapter selections and weight paths", () => {
    expect(
      buildPipelineAdapterConfig({
        poseAdapter: "mediapipe",
        handAdapter: "",
        yoyoAdapter: "pytorch",
        trackerAdapter: "kalman",
        temporalAdapter: "torch",
        temporalWeights: "/models/tcn.pt",
        yoyoWeights: "/models/yoyo.pt",
      })
    ).toEqual({
      pose_adapter: "mediapipe",
      yoyo_adapter: "pytorch",
      tracker_adapter: "kalman",
      temporal_event_adapter: "torch",
      adapter_kwargs: {
        temporal_event: { weights_path: "/models/tcn.pt" },
        yoyo: { weights_path: "/models/yoyo.pt" },
      },
    });
  });

  it("uses model_path for onnx yo-yo weights", () => {
    expect(
      buildPipelineAdapterConfig({
        poseAdapter: "",
        handAdapter: "",
        yoyoAdapter: "onnx",
        trackerAdapter: "",
        temporalAdapter: "",
        temporalWeights: "",
        yoyoWeights: "/models/yoyo.onnx",
      })
    ).toEqual({
      yoyo_adapter: "onnx",
      adapter_kwargs: {
        yoyo: { model_path: "/models/yoyo.onnx" },
      },
    });
  });
});

describe("summarizePipelineAdapterConfig", () => {
  it("returns null for empty config", () => {
    expect(summarizePipelineAdapterConfig(null)).toBeNull();
    expect(summarizePipelineAdapterConfig({})).toBeNull();
  });

  it("joins adapter names for display", () => {
    expect(
      summarizePipelineAdapterConfig({
        pose_adapter: "mediapipe",
        temporal_event_adapter: "torch",
      })
    ).toBe("pose=mediapipe, temporal=torch");
  });
});
