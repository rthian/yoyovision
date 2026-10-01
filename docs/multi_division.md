# Multi-division foundation

YoYoVision stores an immutable competition division (`1A`, `2A`, `3A`, `4A`,
or `5A`) on every video, analysis job, judging entry, and exported dataset
record. A judging entry may contain only videos from its own division. This
prevents training data, model output, and contest results from being mixed
silently across disciplines.

## Current support

| Capability | 1A | 2A–5A |
|---|---:|---:|
| Upload and video review | Yes | Yes |
| Manual judging panels | Yes | Yes |
| Dataset export and annotation provenance | Yes | Yes |
| Automated trick analysis and scoring | Prototype | Disabled |

The 1A pipeline is still a research prototype: no real trained yo-yo detector
or real-footage temporal checkpoint ships with the repository. Automated
analysis is blocked for 2A–5A because the existing event ontology, mock model,
and draft ruleset describe 1A. A non-1A upload is stored without creating an
analysis job, and a manual analysis request returns a clear validation error.

## Model rollout contract

Each division needs its own versioned event ontology, training corpus, model
head, evaluation set, and rules adapter. A shared video encoder may be reused,
but a checkpoint must declare the divisions it supports. The API should enable
automated analysis only when a compatible ontology, checkpoint, and ruleset are
all configured.

- **2A:** detect and track two looping yo-yos, hands, loop direction, cadence,
  synchronization, wraps, regenerations, stops, and changes. High frame rate
  and motion blur robustness are essential.
- **3A:** track two independently manipulated string-trick yo-yos and their
  strings, including crossings, mounts, transfers, collisions, and misses.
- **4A:** model an offstring yo-yo, releases, airborne trajectory, catches,
  regenerations, drops, and replacement events.
- **5A:** track the yo-yo and counterweight as separate objects plus the string
  relationship, releases, catches, transfers, tangles, and detachments.

Promote a division only after player-separated testing on unseen competition
footage, comparison against multiple qualified judges, confidence calibration,
and a shadow-mode trial where model output cannot affect official results.
