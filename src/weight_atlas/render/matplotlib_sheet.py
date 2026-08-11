"""Matplotlib sheet renderer: hillshade, hypsometric tint, contours."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 – must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap

from weight_atlas.core.registry import register_renderer
from weight_atlas.core.types import AtlasSpec, Field2D

# Fixed hypsometric palette: green → brown → white
_HYPSO_COLORS = ['#2166ac', '#4393c3', '#92c5de', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
_HYPSO = LinearSegmentedColormap.from_list("hypsometric", _HYPSO_COLORS, N=256)

# Fixed PNG metadata per spec: Software + Creation Time (epoch zero for determinism)
_PNG_METADATA = {
    "Software": "weight-atlas",
    "Creation Time": "1970-01-01T00:00:00Z",
}

# Thresholds for degenerate channel detection
_EPS = 1e-6
_MIN_VALID_FRACTION = 0.1


@register_renderer("sheet")
class MatplotlibSheet:
    """Renders a field as a topographic sheet: hillshade + hypsometric tint + contours."""

    renderer_id = "sheet"

    def render(self, field: Field2D, spec: AtlasSpec, out: Path, *, scatter_path: Path | None = None) -> list[Path]:
        out.mkdir(parents=True, exist_ok=True)
        sheet = spec.sheet
        dpi = int(sheet["dpi"])
        azdeg = float(sheet["light_azdeg"])
        altdeg = float(sheet["light_altdeg"])
        contour_levels = int(sheet.get("contour_levels", 12))

        data = field.data
        n_rows, n_cols = data.shape
        figsize = (max(6, n_cols * 0.5), max(4, n_rows * 0.4))

        # Check for degenerate channel
        is_degenerate, degen_reason = _check_degenerate(data)

        # Per-row normalization (spec v2 knob)
        if sheet.get("per_row_normalize", False):
            data = _per_row_normalize(data)

        # Direct colormap (no hillshade) for better contrast with rank-scaled data
        normed = filled_norm(data)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(normed, cmap=_HYPSO, vmin=0, vmax=1, origin="upper", extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5))

        # Scatter overlay for embedding visualization
        if scatter_path is not None and scatter_path.exists():
            scatter_coords = np.load(scatter_path)
            if scatter_coords.size > 0:
                # Normalize scatter to data grid
                x = scatter_coords[:, 0]
                y = scatter_coords[:, 1]
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                if x_max > x_min:
                    x_norm = (x - x_min) / (x_max - x_min) * (n_cols - 1)
                else:
                    x_norm = np.full_like(x, n_cols / 2)
                if y_max > y_min:
                    y_norm = (y - y_min) / (y_max - y_min) * (n_rows - 1)
                else:
                    y_norm = np.full_like(y, n_rows / 2)
                ax.scatter(x_norm, y_norm, s=0.3, alpha=0.3, c="white", linewidths=0)

        # Contour overlay on the scaled height field (v2.1: fixed [0,1] range).
        # robust_scale guarantees a well-distributed [0,1] range, so fixed
        # percentile levels are globally comparable without per-model recomputation.
        finite = np.isfinite(data)
        if finite.any():
            levels = np.linspace(0.02, 0.98, contour_levels)
            ax.contour(
                data,
                levels=levels,
                colors="black",
                alpha=0.4,
                linewidths=0.6,
                origin="upper",
                extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5),
            )

        ax.set_xlabel("slot")
        ax.set_ylabel("layer")
        # Handle upsampled data: col_labels may have fewer entries than n_cols
        if len(field.col_labels) == n_cols:
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(field.col_labels, rotation=90, fontsize=6)
        elif len(field.col_labels) > 0:
            # Upsampled: show ticks at original positions
            n_labels = len(field.col_labels)
            tick_positions = [i * n_cols // n_labels for i in range(n_labels)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(field.col_labels, rotation=90, fontsize=6)
        if len(field.row_labels) == n_rows:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(field.row_labels, fontsize=6)
        elif len(field.row_labels) > 0:
            n_labels = len(field.row_labels)
            tick_positions = [i * n_rows // n_labels for i in range(n_labels)]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(field.row_labels, fontsize=6)
        ax.set_title(f"{field.channel} – raw")

        # Colorbar/legend showing actual data range
        if finite.any():
            vals = data[finite]
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            # Create a scalar mappable for the colorbar
            sm = plt.cm.ScalarMappable(cmap=_HYPSO, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"Value range: {vmin:.2f} – {vmax:.2f}", fontsize=8)
            cbar.ax.tick_params(labelsize=6)

        # Degenerate channel banner on PNG
        if is_degenerate:
            fig.text(
                0.5, 0.01,
                f"Channel degenerate — check spec calibration: {degen_reason}",
                ha="center", va="bottom",
                fontsize=8, color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
            )

        raw_path = out / f"{field.channel}_raw.png"
        fig.savefig(raw_path, dpi=dpi, bbox_inches="tight", metadata=_PNG_METADATA)
        plt.close(fig)
        return [raw_path]


def _check_degenerate(data: np.ndarray) -> tuple[bool, str]:
    """Check if a channel field is degenerate (constant or near-constant).

    Returns (is_degenerate, reason_string).
    """
    total_cells = data.size
    if total_cells == 0:
        return True, "empty field"

    finite_mask = np.isfinite(data)
    n_valid = int(finite_mask.sum())
    valid_fraction = n_valid / total_cells

    if n_valid == 0:
        return True, "no finite values"

    vals = data[finite_mask]
    std = float(np.std(vals))
    mean = float(np.mean(np.abs(vals)))
    normalized_std = std / mean if mean > 0 else (0.0 if std == 0 else std)

    reasons: list[str] = []
    if normalized_std < _EPS:
        reasons.append(f"normalized_std={normalized_std:.2e} < {_EPS}")
    if valid_fraction < _MIN_VALID_FRACTION:
        reasons.append(f"valid_fraction={valid_fraction:.1%} < {_MIN_VALID_FRACTION:.0%}")

    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def _per_row_normalize(data: np.ndarray) -> np.ndarray:
    """Normalize each row independently to [0, 1].

    Helps when norms dominate — reveals within-row structure.
    NaN values are preserved.
    """
    out = data.copy()
    for i in range(out.shape[0]):
        row = out[i, :]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        vals = row[finite]
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax > vmin:
            row[finite] = (vals - vmin) / (vmax - vmin)
        else:
            row[finite] = 0.0
    return out


def filled_norm(data: np.ndarray, quantile_clip: float = 0.02) -> np.ndarray:
    """Normalize a field to [0,1] using quantile clipping, ignoring NaN.

    Uses q02-q98 quantile range for normalization to be robust against outliers.
    Values outside the quantile range are clipped.
    """
    out = data.copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    vals = out[finite]
    vmin = float(np.quantile(vals, quantile_clip))
    vmax = float(np.quantile(vals, 1 - quantile_clip))
    if vmax > vmin:
        out[finite] = np.clip((out[finite] - vmin) / (vmax - vmin), 0, 1)
    else:
        out[finite] = 0.0
    return out
