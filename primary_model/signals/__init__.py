"""Signal-construction adapters."""

from .variant1 import (
    build_primary_signal_variant1,
    build_variant1_indicators,
    composite_score,
    expanding_zscore,
    score_to_signal,
)

__all__ = [
    "build_primary_signal_variant1",
    "build_variant1_indicators",
    "composite_score",
    "expanding_zscore",
    "score_to_signal",
]
