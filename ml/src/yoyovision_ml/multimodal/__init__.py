"""Prompt E: RGB, string, and audio fusion.

Everything Prompt B/C built is kinematic -- pose/hand landmarks, yo-yo
bounding boxes/tracks, and the hand<->yoyo distance geometry proxy in
`string_analysis.py`. This package adds three *additional* modalities
(appearance/RGB, pixel-based string segmentation, and audio) as their own
replaceable `Protocol` + registry adapters (see `interfaces.py`,
`adapters_registry.py`), plus a fusion step that merges their output with
the existing kinematic `FeatureSet` into one enriched timeline.

Like Prompt B/C before real weights existed, every adapter here ships only
as a deterministic mock (`adapters_mock.py`) -- there is no real annotated
1A footage with RGB/string-mask/audio ground truth in this repository yet
(see `ml/scripts/generate_sample_dataset.py`). Fusion is therefore
opt-in and *not* the pipeline's default (`run_analysis_pipeline`'s
`feature_fusion_mode` stays `"kinematics_only"` unless a caller explicitly
asks for `"fused"`), exactly like Prompt C's `"torch"` temporal event
detector isn't the default despite being trainable end to end today.
"""

from __future__ import annotations
