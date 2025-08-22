"""Constants for tree-related modules (split criteria, histograms).

This module centralizes string literals and defaults used across
`split_criteria.py` and `histogram.py` to avoid typos and ease maintenance.
"""


class StatKey:
    """Dictionary keys for aggregate statistics."""

    COUNT = "count"
    SUM_Y = "sum_y"
    SUM_Y2 = "sum_y2"


class HistogramType:
    """Supported histogram type names."""

    ADAPTIVE = "adaptive"
    EQUIWIDTH = "equiwidth"
    MEAN = "mean"
    AVERAGE = "average"


class SplitCriterion:
    """Aliases for split-criterion selector names."""

    # Canonical names
    VR_ABS = "vr_abs"
    VR_REL = "vr_rel"

    # Accepted aliases
    ABS_ALIASES = (VR_ABS, "vr", "reduction_abs")
    REL_ALIASES = (VR_REL, "vr_relative", "reduction_rel")


__all__ = [
    "StatKey",
    "HistogramType",
    "SplitCriterion",
]
