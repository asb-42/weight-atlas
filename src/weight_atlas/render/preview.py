"""Preview renderer: float32 TIFF → 8-bit PNG with auto-levels + gamma correction.

This renderer converts raw TIFF fields into viewable PNGs without requiring
ImageJ or other scientific image viewers. It applies auto-levels (quantile-based
histogram stretching) and gamma correction for visual inspection.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 – must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np

from weight_atlas.core.registry import register_renderer
from weight_atlas.core.types import AtlasSpec, Field2D

# Fixed PNG metadata per spec
_PNG_METADATA = {
    "Software": "weight-atlas",
    "Creation Time": "1970-01-01T00:00:00Z",
}

# Pixel budget for the rendered raster (mirrors the sheet renderer's cap).
# An expert panel is e.g. 4096×384 (8× upsample) — at the natural figsize
# (~2048×153 in) and dpi=150 that becomes a ~7 GPx RGBA buffer (~28 GB) and
# OOM-kills the worker. Cap the raster so huge panels keep a useful
# resolution while the allocation stays small.
_PREVIEW_DPI = 150
_MAX_RENDER_PIXELS = 12_000_000


@register_renderer("preview")
class PreviewRenderer:
    """Renders a float32 TIFF field as an 8-bit PNG with auto-levels + gamma.

    Uses quantile-based auto-levels (1%–99%) and gamma=2.2 for perceptual
    correction. Output is suitable for quick visual inspection in any image viewer.
    """

    renderer_id = "preview"

    def render(self, field: Field2D, spec: AtlasSpec, out: Path, **kwargs: object) -> list[Path]:
        out.mkdir(parents=True, exist_ok=True)

        data = field.data
        n_rows, n_cols = data.shape

        # Bound the raster to a fixed pixel budget (see ``_MAX_RENDER_PIXELS``).
        # The natural preview figsize scales with the field's dimensions, so a
        # wide expert panel would otherwise allocate a huge RGBA buffer.
        scale = math.sqrt(_MAX_RENDER_PIXELS / max(1, n_rows * n_cols))
        px_h = max(2, int(round(n_rows * scale)))
        px_w = max(2, int(round(n_cols * scale)))
        figsize = (max(6.0, px_w / _PREVIEW_DPI), max(4.0, px_h / _PREVIEW_DPI))

        # Auto-levels: quantile-based histogram stretching
        normalized = _auto_levels(data, lo=0.01, hi=0.99)

        # Gamma correction (gamma=2.2 for perceptual brightening)
        gamma = 2.2
        normalized = np.power(normalized, 1.0 / gamma)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(normalized, cmap="gray", origin="upper",
                  extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5))

        ax.set_xlabel("slot")
        ax.set_ylabel("layer")

        # Slot labels: dense when labels match the column count, otherwise
        # spread the (fewer) slot names across the upsampled columns.
        if field.col_labels:
            if len(field.col_labels) == n_cols:
                step = max(1, n_cols // 20)
                ax.set_xticks(range(0, n_cols, step))
                ax.set_xticklabels([field.col_labels[i] for i in range(0, n_cols, step)],
                                   rotation=90, fontsize=6, ha="center")
            else:
                n_labels = len(field.col_labels)
                positions = [i * n_cols // n_labels for i in range(n_labels)]
                ax.set_xticks(positions)
                ax.set_xticklabels(field.col_labels, rotation=90, fontsize=6, ha="center")

        if field.row_labels:
            if len(field.row_labels) == n_rows:
                step = max(1, n_rows // 20)
                ax.set_yticks(range(0, n_rows, step))
                ax.set_yticklabels([field.row_labels[i] for i in range(0, n_rows, step)], fontsize=6)
            else:
                n_labels = len(field.row_labels)
                positions = [i * n_rows // n_labels for i in range(n_labels)]
                ax.set_yticks(positions)
                ax.set_yticklabels(field.row_labels, fontsize=6)

        title = f"{field.channel} — preview (auto-levels, γ=2.2)"
        if field.model_name:
            title = f"{field.model_name}: {title}"
        ax.set_title(title)

        raw_path = out / f"preview_{field.channel}.png"
        fig.savefig(raw_path, dpi=_PREVIEW_DPI, bbox_inches="tight", metadata=_PNG_METADATA)
        plt.close(fig)
        return [raw_path]


def _auto_levels(data: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    """Apply auto-levels: quantile-based histogram stretching to [0, 1].

    Values outside the quantile range are clipped. NaN is preserved.
    """
    out = data.copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    vals = out[finite]
    vmin = float(np.quantile(vals, lo))
    vmax = float(np.quantile(vals, hi))
    if vmax > vmin:
        out[finite] = np.clip((out[finite] - vmin) / (vmax - vmin), 0, 1)
    else:
        out[finite] = 0.0
    return out
