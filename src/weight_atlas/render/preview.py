"""Preview renderer: float32 TIFF → 8-bit PNG with auto-levels + gamma correction.

This renderer converts raw TIFF fields into viewable PNGs without requiring
ImageJ or other scientific image viewers. It applies auto-levels (quantile-based
histogram stretching) and gamma correction for visual inspection.
"""

from __future__ import annotations

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
        figsize = (max(6, n_cols * 0.5), max(4, n_rows * 0.4))

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
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(field.col_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(field.row_labels, fontsize=6)
        ax.set_title(f"{field.channel} — preview (auto-levels, γ=2.2)")

        raw_path = out / f"preview_{field.channel}.png"
        fig.savefig(raw_path, dpi=150, bbox_inches="tight", metadata=_PNG_METADATA)
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
