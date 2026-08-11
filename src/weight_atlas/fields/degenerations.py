"""Per-channel degeneration detection: valid fraction, normalized Std, range compression, min cells.

Detects degenerate channels where:
- Std < eps (nearly constant values across all slots/layers)
- valid_fraction < 50% (too many NaN/missing cells)
- range_compression: extreme outliers compress >95% of values into <5% of raw range (v2.1)
- min_cells: too few cells for robust percentile statistics (v2.1)

Produces warnings that flow into:
- CLI stderr output
- warnings block in fingerprint.json
- UI banner on detail page
"""

from __future__ import annotations

import sys
from typing import TextIO
from dataclasses import dataclass, field

import numpy as np

# Thresholds
_EPS = 1e-6  # normalized Std below this → degenerate
_MIN_VALID_FRACTION = 0.5  # less than 50% valid → degenerate
_RANGE_COMPRESSION_THRESHOLD = 0.05  # (q_hi - q_lo) / (raw_max - raw_min) below this → warning
_MIN_CELLS_FOR_ROBUST_STATS = 50  # fewer cells → rank-based stats unreliable


@dataclass
class ChannelDiagnostics:
    """Diagnostics for a single channel field."""
    channel: str
    valid_fraction: float  # fraction of cells that are finite
    normalized_std: float  # std / mean of finite values (0 if mean is 0)
    is_degenerate: bool
    reason: str  # explanation if degenerate
    range_compression: float | None = None  # v2.1: ratio of robust range to raw range
    n_valid: int = 0  # number of finite cells


@dataclass
class DegenerationReport:
    """Full degeneration report for all channels."""
    channels: dict[str, ChannelDiagnostics] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_degenerations(self) -> bool:
        return any(c.is_degenerate for c in self.channels.values())


def diagnose_channel(channel: str, field: np.ndarray) -> ChannelDiagnostics:
    """Compute diagnostics for a single channel field."""
    total_cells = field.size
    if total_cells == 0:
        return ChannelDiagnostics(
            channel=channel,
            valid_fraction=0.0,
            normalized_std=0.0,
            is_degenerate=True,
            reason="empty field",
        )

    finite_mask = np.isfinite(field)
    n_valid = int(finite_mask.sum())
    valid_fraction = n_valid / total_cells

    if n_valid == 0:
        return ChannelDiagnostics(
            channel=channel,
            valid_fraction=0.0,
            normalized_std=0.0,
            is_degenerate=True,
            reason="no finite values",
        )

    vals = field[finite_mask]
    std = float(np.std(vals))
    mean = float(np.mean(np.abs(vals)))

    # Normalized std: std / mean (coefficient of variation)
    # If mean is 0, check if std is also 0
    normalized_std = std / mean if mean > 0 else 0.0 if std == 0 else std

    # v2.1: range_compression check (post-scaling guard)
    range_compression = _compute_range_compression(vals)

    is_degenerate = False
    reasons: list[str] = []

    if normalized_std < _EPS:
        is_degenerate = True
        reasons.append(f"normalized_std={normalized_std:.2e} < {_EPS}")

    if valid_fraction < _MIN_VALID_FRACTION:
        is_degenerate = True
        reasons.append(f"valid_fraction={valid_fraction:.1%} < {_MIN_VALID_FRACTION:.0%}")

    # v2.1: min_cells guard — too few cells for robust percentile statistics
    if n_valid < _MIN_CELLS_FOR_ROBUST_STATS:
        is_degenerate = True
        reasons.append(f"n_valid={n_valid} < {_MIN_CELLS_FOR_ROBUST_STATS} (too few cells for robust stats)")

    return ChannelDiagnostics(
        channel=channel,
        valid_fraction=round(valid_fraction, 4),
        normalized_std=round(normalized_std, 6),
        is_degenerate=is_degenerate,
        reason="; ".join(reasons) if reasons else "",
        range_compression=range_compression,
        n_valid=n_valid,
    )


def _compute_range_compression(vals: np.ndarray) -> float | None:
    """Compute range compression ratio: (q99 - q1) / (max - min).

    Returns None if the range is zero (constant field).
    Low values indicate extreme outliers that compress the robust range.
    """
    raw_min = float(np.min(vals))
    raw_max = float(np.max(vals))
    raw_range = raw_max - raw_min
    if raw_range <= 0:
        return None
    q_lo = float(np.quantile(vals, 0.01))
    q_hi = float(np.quantile(vals, 0.99))
    return (q_hi - q_lo) / raw_range


def diagnose_fields(
    fields: dict[str, np.ndarray],
    *,
    file: TextIO | None = None,
) -> DegenerationReport:
    """Run diagnostics on all channel fields.

    Args:
        fields: dict mapping channel name to 2D numpy array
        file: where to print warnings (default: stderr)

    Returns:
        DegenerationReport with all diagnostics and warnings
    """
    if file is None:
        file = sys.stderr

    report = DegenerationReport()
    for channel, data in sorted(fields.items()):
        diag = diagnose_channel(channel, data)
        report.channels[channel] = diag
        if diag.is_degenerate:
            warning = (
                f"DEGENERATE CHANNEL '{channel}': {diag.reason} "
                f"(valid_fraction={diag.valid_fraction}, "
                f"normalized_std={diag.normalized_std:.2e})"
            )
            report.warnings.append(warning)
            print(warning, file=file)
        # v2.1: range_compression warning (even if not degenerate)
        if diag.range_compression is not None and diag.range_compression < _RANGE_COMPRESSION_THRESHOLD:
            warning = (
                f"RANGE COMPRESSION '{channel}': extreme outlier detected; "
                f"99% of values compressed into {diag.range_compression:.1%} of raw range "
                f"(threshold: {_RANGE_COMPRESSION_THRESHOLD:.0%})"
            )
            report.warnings.append(warning)
            print(warning, file=file)

    return report


def check_constant_field(field: np.ndarray) -> bool:
    """Check if a field is constant (all same value or all NaN).

    Returns True if the field is degenerate (constant or all-NaN).
    """
    finite_mask = np.isfinite(field)
    if not finite_mask.any():
        return True
    vals = field[finite_mask]
    return bool(np.all(vals == vals[0]))
