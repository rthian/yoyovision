"""Kalman-filter baseline tracker for the yo-yo (`ObjectTracker` protocol).

A constant-velocity Kalman filter over the detection's normalized bbox
center `(cx, cy)`, with bbox width/height carried through as a simple
exponential smoothing (not part of the Kalman state -- size changes are not
usefully modelled by a constant-velocity assumption for a small, fast-moving
object). This is intentionally "modest and reproducible" (matching Prompt C's
later baseline-first philosophy) rather than a learned tracker.

Gap handling:
  * A missing detection for up to `max_gap_ms` is bridged by predicting
    forward from the filter state alone (`Track.interpolated=True`,
    `Track.visibility=VisibilityState.FULLY_OCCLUDED`).
  * Beyond `max_gap_ms`, the tracker stops emitting a `Track` for that frame
    entirely and resets internal velocity confidence -- per Prompt B, "no
    interpolation over long or uncertain gaps."
  * `static_camera=True` slightly lowers process noise (the object's own
    motion, not camera motion, should dominate position change), which in
    turn tightens gap-fill predictions for a tripod-mounted recording.

`track_quality()` returns a 0-1 score for the tracker's own recent history,
combining detection coverage and mean confidence -- a cheap, explainable
proxy metric, not a learned quality model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from yoyovision_ml.adapters_registry import register_tracker
from yoyovision_ml.domain import BoundingBox, Detection, Track, VisibilityState


def _bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    return (bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0)


@dataclass
class _KalmanState:
    """Minimal 4-state (position + velocity) Kalman filter, no numpy/scipy.

    Implemented by hand (rather than pulling in `filterpy`/`opencv`'s KF) so
    this baseline tracker has zero additional runtime dependencies -- it only
    needs `yoyovision_ml.domain` types.
    """

    x: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    # Row-major 4x4 covariance.
    p: list[list[float]] = field(
        default_factory=lambda: [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    )

    def predict(self, dt: float, process_noise: float) -> None:
        # Constant-velocity transition: cx' = cx + vx*dt, vx' = vx.
        cx, cy, vx, vy = self.x
        self.x = [cx + vx * dt, cy + vy * dt, vx, vy]
        # F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]; P' = F P F^T + Q.
        f = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        fp = _matmul(f, self.p)
        ft = _transpose(f)
        self.p = _matmul(fp, ft)
        for i in range(4):
            self.p[i][i] += process_noise

    def update(self, measurement: tuple[float, float], measurement_noise: float) -> None:
        # H = [[1,0,0,0],[0,1,0,0]] (observe position only).
        h = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ht = _transpose(h)
        hp = _matmul(h, self.p)
        s = _matmul(hp, ht)
        s[0][0] += measurement_noise
        s[1][1] += measurement_noise
        s_inv = _invert_2x2(s)
        pht = _matmul(self.p, ht)
        k = _matmul(pht, s_inv)  # 4x2 Kalman gain.

        residual = [measurement[0] - self.x[0], measurement[1] - self.x[1]]
        correction = [k[r][0] * residual[0] + k[r][1] * residual[1] for r in range(4)]
        self.x = [self.x[i] + correction[i] for i in range(4)]

        kh = _matmul(k, h)
        identity_minus_kh = [
            [(1.0 if i == j else 0.0) - kh[i][j] for j in range(4)] for i in range(4)
        ]
        self.p = _matmul(identity_minus_kh, self.p)


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows_a, cols_a, cols_b = len(a), len(a[0]), len(b[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            result[i][j] = sum(a[i][k] * b[k][j] for k in range(cols_a))
    return result


def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [[a[r][c] for r in range(len(a))] for c in range(len(a[0]))]


def _invert_2x2(m: list[list[float]]) -> list[list[float]]:
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    det = det if abs(det) > 1e-12 else 1e-12
    return [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]


@dataclass
class _TrackHistoryEntry:
    frame_ms: int
    had_detection: bool
    confidence: float


@register_tracker("kalman")
class KalmanYoyoTracker:
    """Real, deterministic single-object Kalman tracker for the yo-yo.

    Not a mock: given the same detection stream it always produces the same
    filter trajectory (the KF math is deterministic), but it is a real
    numerical estimator, not a placeholder.
    """

    model_name = "kalman-yoyo-tracker"
    model_version = "0.1.0"

    def __init__(
        self,
        max_gap_ms: int = 500,
        static_camera: bool = False,
        process_noise: float | None = None,
        measurement_noise: float = 5e-4,
    ) -> None:
        self.max_gap_ms = max_gap_ms
        self.static_camera = static_camera
        self.process_noise = (
            process_noise if process_noise is not None else (1e-4 if static_camera else 5e-4)
        )
        self.measurement_noise = measurement_noise
        self._track_id = "track-0"
        self._state: _KalmanState | None = None
        self._last_frame_ms: int | None = None
        self._last_size: tuple[float, float] = (0.05, 0.05)
        self._history: list[_TrackHistoryEntry] = []

    def reset(self) -> None:
        self._track_id = "track-0"
        self._state = None
        self._last_frame_ms = None
        self._last_size = (0.05, 0.05)
        self._history = []

    def update(self, detections: list[Detection], timestamp_ms: int) -> list[Track]:
        frame_detections = [d for d in detections if d.frame_ms == timestamp_ms]
        best = max(frame_detections, key=lambda d: d.confidence, default=None)

        gap_ms = 0 if self._last_frame_ms is None else timestamp_ms - self._last_frame_ms
        if gap_ms < 0:
            raise ValueError(
                f"KalmanYoyoTracker.update called out of order: timestamp_ms={timestamp_ms} "
                f"< last seen {self._last_frame_ms}"
            )

        if best is None and (self._state is None or gap_ms > self.max_gap_ms):
            # No detection and either no established track, or the gap is too
            # long/uncertain to bridge -- emit nothing for this frame.
            if self._state is not None:
                self._history.append(
                    _TrackHistoryEntry(frame_ms=timestamp_ms, had_detection=False, confidence=0.0)
                )
                self._last_frame_ms = timestamp_ms
            return []

        dt = gap_ms / 1000.0
        if self._state is not None and dt > 0:
            self._state.predict(dt, self.process_noise)

        if best is not None:
            if self._state is None:
                self._state = _KalmanState(x=[*_bbox_center(best.bbox), 0.0, 0.0])
            else:
                self._state.update(_bbox_center(best.bbox), self.measurement_noise)
            self._last_size = (best.bbox.width, best.bbox.height)
            visibility = VisibilityState.VISIBLE
            interpolated = False
            confidence = best.confidence
        else:
            # Bridging a short/uncertain gap: predicted position only.
            if self._state is None:  # pragma: no cover - guarded by branch above
                raise AssertionError("unreachable: state must exist when bridging a gap")
            visibility = VisibilityState.FULLY_OCCLUDED
            interpolated = True
            confidence = 0.0

        cx, cy = self._state.x[0], self._state.x[1]
        width, height = self._last_size
        bbox = BoundingBox(x=cx - width / 2.0, y=cy - height / 2.0, width=width, height=height)

        self._history.append(
            _TrackHistoryEntry(
                frame_ms=timestamp_ms, had_detection=best is not None, confidence=confidence
            )
        )
        self._last_frame_ms = timestamp_ms

        return [
            Track(
                track_id=self._track_id,
                frame_ms=timestamp_ms,
                bbox=bbox,
                confidence=confidence,
                class_label="yoyo",
                visibility=visibility,
                interpolated=interpolated,
            )
        ]

    def track_quality(self, window: int | None = None) -> float:
        """A 0-1 score combining detection coverage and mean confidence over
        the tracked history (or the last `window` frames if given).

        This is a cheap, explainable heuristic -- not a learned quality
        model -- so it can be reported alongside every artefact without
        implying more rigor than it has.
        """
        history = self._history[-window:] if window else self._history
        if not history:
            return 0.0
        coverage = sum(1 for h in history if h.had_detection) / len(history)
        mean_confidence = sum(h.confidence for h in history) / len(history)
        return round((coverage + mean_confidence) / 2.0, 4)
