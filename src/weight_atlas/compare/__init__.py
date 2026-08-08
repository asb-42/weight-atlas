"""Comparison/delta layer: compare two scanned models quantitatively and cartographically."""

from weight_atlas.compare.align import (
    AlignResult,
    align,
    check_compatibility,
)
from weight_atlas.compare.delta import (
    ChannelDelta,
    CompareSummary,
    compute_compare_summary,
    hotspot_ranking,
)

__all__ = [
    "AlignResult",
    "ChannelDelta",
    "CompareSummary",
    "align",
    "check_compatibility",
    "compute_compare_summary",
    "hotspot_ranking",
]
