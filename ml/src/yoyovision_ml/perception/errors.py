"""Exceptions shared by every perception adapter (Prompt B).

Kept in their own module (no other project imports) so both `adapters_mock`
-style always-available modules and the optional-dependency adapters in this
package can raise/catch them without import-order concerns.
"""

from __future__ import annotations


class MissingOptionalDependencyError(RuntimeError):
    """Raised when an adapter needing an optional package (mediapipe, torch,
    onnxruntime, opencv) is used but that package is not installed.

    Distinct from `ModelWeightsNotConfiguredError`: this is about the Python
    *package* being absent, not about model weights being unconfigured.
    """

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"The '{package}' package is required for this adapter but is not "
            f"installed. Install it with: pip install 'yoyovision-ml[{extra}]'"
        )


class ModelWeightsNotConfiguredError(RuntimeError):
    """Raised at adapter construction time when real inference is requested
    but no model weights/checkpoint path has been configured.

    Per Prompt B: "Use real inference only when model files are explicitly
    configured. Fail clearly when expected weights are unavailable." This
    must never silently fall back to mock/random output.
    """

    def __init__(self, adapter_name: str, config_hint: str) -> None:
        self.adapter_name = adapter_name
        super().__init__(
            f"No model weights configured for '{adapter_name}'. {config_hint} "
            "Refusing to run un-configured 'real' inference (it would silently "
            "produce meaningless output). Use the 'mock' adapter for tests."
        )
