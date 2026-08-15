"""Matplotlib sheet renderer: direct colormap + contours.

Renders a field as a topographic-style sheet. Uses a direct diverging
colormap (no hillshade) for contrast on rank-scaled data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 – must be set before pyplot import

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from weight_atlas.core.registry import register_renderer
from weight_atlas.core.types import AtlasSpec, Field2D
from weight_atlas.fields.normalize import depth_landmark_labels, project_normalized_depth

# Diverging blue-white-red palette (RdBu family) for rank-scaled [0,1] fields.
# Replaces the original green→brown→white hypsometric terrain scale, which
# washed out uniformly distributed rank data (commit 6067626).
_HYPSO_COLORS = ['#2166ac', '#4393c3', '#92c5de', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
_HYPSO = LinearSegmentedColormap.from_list("sheet_diverging", _HYPSO_COLORS, N=256)

# Fixed PNG metadata per spec: Software + Creation Time (epoch zero for determinism)
_PNG_METADATA = {
    "Software": "weight-atlas",
    "Creation Time": "1970-01-01T00:00:00Z",
}

# Thresholds for degenerate channel detection
_EPS = 1e-6
_MIN_VALID_FRACTION = 0.1

# Cap on the total number of rendered pixels (RGBA = 4 bytes/px). Fields like
# large MoE expert panels (Kimi K3: 736x7168 after upsampling) would otherwise
# produce a figure sized in inches proportional to the field — 7168 x 0.5 in at
# 150 dpi is 537,600 px wide, a ~95 GB RGBA buffer that OOM-kills the worker.
# The raster is scaled to this pixel budget preserving the aspect ratio: small
# fields are upscaled for quality, and huge wide panels keep useful resolution
# on their short axis instead of being crushed into a thin strip.
_MAX_RENDER_PIXELS = 12_000_000


@register_renderer("sheet")
class MatplotlibSheet:
    """Renders a field as a topographic-style sheet: direct colormap + contours."""

    renderer_id = "sheet"

    def render(self, field: Field2D, spec: AtlasSpec, out: Path, *, scatter_path: Path | None = None) -> list[Path]:
        out.mkdir(parents=True, exist_ok=True)
        sheet = spec.sheet
        dpi = int(sheet["dpi"])
        contour_levels = int(sheet.get("contour_levels", 12))

        data = field.data
        n_rows, n_cols = data.shape

        # Ebene 2: normalized-depth projection (spec v2.4 knob). Re-map rows onto
        # fixed relative-depth landmarks (0%..100%) so models with different layer
        # counts become comparable and NaN "perforation" holes are interpolated
        # over. The interpolation mask marks every estimated cell so the sheet can
        # shade it and stay honest about where values are estimates.
        row_labels = field.row_labels
        col_labels = field.col_labels
        interp_mask: np.ndarray | None = None
        normalized_depth = False
        if sheet.get("normalized_depth", False) and n_rows > 1:
            n_landmarks = int(sheet.get("normalized_depth_landmarks", 21))
            data, interp_mask = project_normalized_depth(data, n_landmarks)
            n_rows, n_cols = data.shape
            row_labels = depth_landmark_labels(n_landmarks)
            normalized_depth = True

        # Ebene 3: drop all-NaN columns (spec v2.4 knob, default off). Slot
        # families absent from a model leave white columns; compressing them
        # makes the sheet readable. Display-only — TIFF fields keep full width.
        # Smooth fields are upsampled, so one slot spans ``block`` data columns;
        # drop whole slots, then imshow's extent re-spaces the survivors evenly.
        dropped_cols: list[str] = []
        if sheet.get("drop_empty_cols", False) and n_cols > 1 and col_labels:
            n_labels = min(len(col_labels), n_cols)
            block = n_cols // n_labels if n_labels and n_cols % n_labels == 0 else 1
            valid_slots = [
                bool(np.isfinite(data[:, i * block : (i + 1) * block]).any())
                for i in range(n_labels)
            ]
            if not all(valid_slots):
                keep_mask = np.repeat(valid_slots, block)
                dropped_cols = [
                    lbl for lbl, keep in zip(col_labels[:n_labels], valid_slots, strict=True) if not keep
                ]
                data = data[:, keep_mask]
                n_cols = data.shape[1]
                col_labels = [
                    lbl for lbl, keep in zip(col_labels[:n_labels], valid_slots, strict=True) if keep
                ]
                if interp_mask is not None:
                    interp_mask = interp_mask[:, keep_mask]

        # Bound the raster: keep the field's aspect ratio and scale to a fixed
        # pixel budget so the dpi-scaled RGBA buffer stays small even for huge
        # panels (see ``_MAX_RENDER_PIXELS``). Without this, rendering a
        # 736x7168 expert panel allocates ~95 GB and is OOM-killed; a
        # long-edge-only cap crushes it to 4096x420 (uninspectable layers).
        scale = math.sqrt(_MAX_RENDER_PIXELS / max(1, n_rows * n_cols))
        px_h = max(2, int(round(n_rows * scale)))
        px_w = max(2, int(round(n_cols * scale)))
        figsize = (max(2.0, px_w / dpi), max(2.0, px_h / dpi))

        # Check for degenerate channel
        is_degenerate, degen_reason = _check_degenerate(data)

        # Per-row normalization (spec v2 knob)
        if sheet.get("per_row_normalize", False):
            data = _per_row_normalize(data)

        # Direct colormap (no hillshade) for better contrast with rank-scaled data
        normed = filled_norm(data)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(normed, cmap=_HYPSO, vmin=0, vmax=1, origin="upper", extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5))

        # Subtle gray veil over interpolated (estimated) cells so the normalized
        # depth projection stays honest about which regions are estimates.
        if interp_mask is not None and interp_mask.any():
            shade = np.where(interp_mask & np.isfinite(data), 1.0, np.nan)
            ax.imshow(
                shade,
                cmap="Greys", vmin=0, vmax=1, origin="upper",
                extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5),
                alpha=0.25, zorder=2,
            )

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

        # Axis labels
        ax.set_xlabel("slot", fontsize=10, fontweight="bold")
        ax.set_ylabel("layer", fontsize=10, fontweight="bold")

        # Handle upsampled data: col_labels may have fewer entries than n_cols
        if len(col_labels) == n_cols:
            step = max(1, n_cols // 20)
            ax.set_xticks(range(0, n_cols, step))
            ax.set_xticklabels([col_labels[i] for i in range(0, n_cols, step)],
                             rotation=90, fontsize=7, ha="center")
        elif len(col_labels) > 0:
            n_labels = len(col_labels)
            tick_positions = [i * n_cols // n_labels for i in range(n_labels)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(col_labels, rotation=90, fontsize=7, ha="center")

        if len(row_labels) == n_rows:
            step = max(1, n_rows // 20)
            ax.set_yticks(range(0, n_rows, step))
            ax.set_yticklabels([row_labels[i] for i in range(0, n_rows, step)],
                             fontsize=7)
        elif len(row_labels) > 0:
            n_labels = len(row_labels)
            tick_positions = [i * n_rows // n_labels for i in range(n_labels)]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(row_labels, fontsize=7)

        # Title: model · channel · transformation
        display_channel = field.channel
        ch_spec = spec.channels.get(field.channel, {})
        if not ch_spec and field.channel.startswith("vision_"):
            # Vision sheets use the vision slot taxonomy and its own channels.
            display_channel = f"vision:{field.channel[len('vision_'):]}"
            ch_spec = spec.vision_channels.get(field.channel[len('vision_'):], {})
        if not ch_spec and field.channel.startswith("expert_"):
            # MoE expert panels use the cheap expert_channels statistics.
            panel_slot, base = field.channel[len("expert_"):].split("_", 1)
            display_channel = f"expert:{panel_slot}:{base}"
            ch_spec = (spec.expert_channels or spec.channels).get(base, {})
        transform_parts = []
        if ch_spec.get("pre"):
            transform_parts.append(ch_spec["pre"])
        if ch_spec.get("scale", {}).get("type"):
            transform_parts.append(ch_spec["scale"]["type"])
        transform_str = " → ".join(transform_parts) if transform_parts else "raw"
        title = f"{display_channel}: {transform_str}"
        if field.model_name:
            title = f"{field.model_name}: {title}"
        if normalized_depth:
            title = f"{title} · normalized-depth (interp shaded)"
        if dropped_cols:
            title = f"{title} · columns dropped: {', '.join(dropped_cols)}"
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Colorbar with real percentile values mapped onto the display scale.
        # The image shows filled_norm(data) (q02–q98 min-max to [0,1]); place
        # p10/p50/p90 of the underlying data at their actual display positions
        # so the scale reads in real values, not bare normalized ranks.
        if finite.any():
            vals = data[finite]
            p10 = float(np.quantile(vals, 0.1))
            p50 = float(np.quantile(vals, 0.5))
            p90 = float(np.quantile(vals, 0.9))
            vmin = float(np.quantile(vals, 0.02))
            vmax = float(np.quantile(vals, 0.98))
            sm = plt.cm.ScalarMappable(cmap=_HYPSO, norm=Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)
            if vmax > vmin:
                positions = [float(np.clip((p - vmin) / (vmax - vmin), 0.0, 1.0)) for p in (p10, p50, p90)]
                cbar.set_ticks(positions)
                cbar.set_ticklabels([f"{v:.3g}" for v in (p10, p50, p90)])
                cbar.set_label("p10 / p50 / p90", fontsize=8)
            else:
                cbar.set_label(f"constant ({p50:.3g})", fontsize=8)

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
