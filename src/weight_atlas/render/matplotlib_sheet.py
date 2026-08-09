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
_HYPSO_COLORS = ["#3b6e2a", "#7a5f3c", "#c8b48a", "#f5f1e3", "#ffffff"]
_HYPSO = LinearSegmentedColormap.from_list("hypsometric", _HYPSO_COLORS, N=256)

# Fixed PNG metadata per spec: Software + Creation Time (epoch zero for determinism)
_PNG_METADATA = {
    "Software": "weight-atlas",
    "Creation Time": "1970-01-01T00:00:00Z",
}


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

        # Hillshade from the normalised field.
        ls = LightSource(azdeg=azdeg, altdeg=altdeg)
        normed = filled_norm(data)
        hs = ls.shade(normed, cmap=_HYPSO, vert_exag=1.0)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(hs, origin="upper", extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5))

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

        # Contour overlay on the scaled height field (q02-q98 quantile range).
        finite = np.isfinite(data)
        if finite.any():
            vals = data[finite]
            q02 = float(np.quantile(vals, 0.02))
            q98 = float(np.quantile(vals, 0.98))
            if q98 > q02:
                levels = np.linspace(q02, q98, contour_levels)
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
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(field.col_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(field.row_labels, fontsize=6)
        ax.set_title(f"{field.channel} – raw")

        raw_path = out / f"{field.channel}_raw.png"
        fig.savefig(raw_path, dpi=dpi, bbox_inches="tight", metadata=_PNG_METADATA)
        plt.close(fig)
        return [raw_path]


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
