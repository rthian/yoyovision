"""Config-driven adapter registry.

Product principle #5: "Model adapters must be replaceable. Do not tightly
couple the product to one detector, tracker or model vendor." Calling code
(the workers pipeline) resolves adapters by name through this registry
instead of importing concrete classes directly, so swapping a mock adapter
for a real one is a configuration change, not a code change.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_POSE_ESTIMATORS: dict[str, Callable[[], object]] = {}
_HAND_ESTIMATORS: dict[str, Callable[[], object]] = {}
_YOYO_DETECTORS: dict[str, Callable[[], object]] = {}
_TRACKERS: dict[str, Callable[[], object]] = {}
_TEMPORAL_EVENT_DETECTORS: dict[str, Callable[[], object]] = {}
_STORAGE_BACKENDS: dict[str, Callable[..., object]] = {}
#: Prompt E (RGB/string/audio fusion) adapter kinds.
_RGB_ENCODERS: dict[str, Callable[[], object]] = {}
_STRING_SEGMENTERS: dict[str, Callable[[], object]] = {}
_AUDIO_ANALYZERS: dict[str, Callable[[], object]] = {}


class AdapterNotRegisteredError(KeyError):
    """Raised when a requested adapter name has no registered factory."""


def _register(registry: dict[str, Callable[..., object]], name: str) -> Callable[[T], T]:
    def decorator(factory: T) -> T:
        registry[name] = factory  # type: ignore[assignment]
        return factory

    return decorator


def register_pose_estimator(name: str) -> Callable[[T], T]:
    return _register(_POSE_ESTIMATORS, name)


def register_hand_estimator(name: str) -> Callable[[T], T]:
    return _register(_HAND_ESTIMATORS, name)


def register_yoyo_detector(name: str) -> Callable[[T], T]:
    return _register(_YOYO_DETECTORS, name)


def register_tracker(name: str) -> Callable[[T], T]:
    return _register(_TRACKERS, name)


def register_temporal_event_detector(name: str) -> Callable[[T], T]:
    return _register(_TEMPORAL_EVENT_DETECTORS, name)


def register_storage_backend(name: str) -> Callable[[T], T]:
    return _register(_STORAGE_BACKENDS, name)


def register_rgb_encoder(name: str) -> Callable[[T], T]:
    return _register(_RGB_ENCODERS, name)


def register_string_segmenter(name: str) -> Callable[[T], T]:
    return _register(_STRING_SEGMENTERS, name)


def register_audio_analyzer(name: str) -> Callable[[T], T]:
    return _register(_AUDIO_ANALYZERS, name)


def _resolve(
    registry: dict[str, Callable[..., object]], name: str, kind: str, **kwargs: object
) -> object:
    try:
        factory = registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "<none registered>"
        raise AdapterNotRegisteredError(
            f"No {kind} adapter named '{name}' is registered. Available: {available}"
        ) from exc
    return factory(**kwargs)


def create_pose_estimator(name: str, **kwargs: object) -> object:
    return _resolve(_POSE_ESTIMATORS, name, "pose estimator", **kwargs)


def create_hand_estimator(name: str, **kwargs: object) -> object:
    return _resolve(_HAND_ESTIMATORS, name, "hand estimator", **kwargs)


def create_yoyo_detector(name: str, **kwargs: object) -> object:
    return _resolve(_YOYO_DETECTORS, name, "yo-yo detector", **kwargs)


def create_tracker(name: str, **kwargs: object) -> object:
    return _resolve(_TRACKERS, name, "tracker", **kwargs)


def create_temporal_event_detector(name: str, **kwargs: object) -> object:
    return _resolve(_TEMPORAL_EVENT_DETECTORS, name, "temporal event detector", **kwargs)


def create_storage_backend(name: str, **kwargs: object) -> object:
    try:
        factory = _STORAGE_BACKENDS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_STORAGE_BACKENDS)) or "<none registered>"
        raise AdapterNotRegisteredError(
            f"No storage backend named '{name}' is registered. Available: {available}"
        ) from exc
    return factory(**kwargs)


def create_rgb_encoder(name: str, **kwargs: object) -> object:
    return _resolve(_RGB_ENCODERS, name, "RGB encoder", **kwargs)


def create_string_segmenter(name: str, **kwargs: object) -> object:
    return _resolve(_STRING_SEGMENTERS, name, "string segmenter", **kwargs)


def create_audio_analyzer(name: str, **kwargs: object) -> object:
    return _resolve(_AUDIO_ANALYZERS, name, "audio analyzer", **kwargs)
