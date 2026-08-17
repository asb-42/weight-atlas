"""Delta sheet renderer: diverging colormap visualization of field deltas."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 – must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm, TwoSlopeNorm

from weight_atlas.core.registry import register_renderer
from weight_atlas.core.types import AtlasSpec

# Fixed PNG metadata per spec (same as matplotlib_sheet)
_PNG_METADATA = {
    "Software": "weight-atlas",
    "Creation Time": "1970-01-01T00:00:00Z",
}


def _noise_floor_mask(delta: np.ndarray, noise_floor_dir: Path, channel: str) -> np.ndarray:
    """Per-cell mask of |delta| at or below the calibration |delta|.

    Reads the calibration compare job's ``field_delta_<channel>_raw.tif`` and
    marks every cell where the current |delta| <= calibration |delta|. NaN is
    never a value: both sides must be finite for the cell to be veiled.
    """
    calib_path = noise_floor_dir / f"field_delta_{channel}_raw.tif"
    if not calib_path.exists():
        return np.zeros(delta.shape, dtype=bool)
    from weight_atlas.fields.tif_io import read_tif

    calib = read_tif(calib_path)
    if calib.shape != delta.shape:
        return np.zeros(delta.shape, dtype=bool)
    floor = np.abs(calib)
    cur = np.abs(delta)
    both = np.isfinite(floor) & np.isfinite(cur)
    mask = np.zeros(delta.shape, dtype=bool)
    mask[both] = cur[both] <= floor[both]
    return mask


def _get_diverging_clip(spec: AtlasSpec) -> float:
    """Get diverging_clip quantile from spec, default 0.98."""
    compare_spec = getattr(spec, "compare", None)
    if compare_spec and isinstance(compare_spec, dict):
        return float(compare_spec.get("diverging_clip", 0.98))
    return 0.98


def _compute_vmax(data: np.ndarray, diverging_clip: float) -> float:
    """Symmetric limit for the diverging colormap.

    Uses the ``diverging_clip`` quantile of |Δ|, capped at a robust spread
    (median + 4.4826·MAD, ≈3σ) so a few extreme outliers cannot flatten the
    bulk of the values toward white.
    """
    finite = np.isfinite(data)
    if not finite.any():
        return 1.0
    abs_vals = np.abs(data[finite])
    vmax = float(np.quantile(abs_vals, diverging_clip))
    if abs_vals.size >= 20:
        median = float(np.median(abs_vals))
        mad = float(np.median(np.abs(abs_vals - median)))
        robust = median + 4.4826 * mad
        if robust > 0 and vmax > robust:
            return robust
    return vmax


@register_renderer("delta")
class DeltaSheet:
    """Renders a delta field as a diverging colormap sheet.

    Uses a blue-white-red diverging colormap centered at zero.
    """

    renderer_id = "delta"

    def render(
        self,
        delta: np.ndarray,
        spec: AtlasSpec,
        out: Path,
        *,
        channel: str = "height",
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        mode: str = "strict",
        model_a: str = "",
        model_b: str = "",
        render_profile: bool = True,
        noise_floor_dir: Path | None = None,
    ) -> list[Path]:
        """Render a delta field as a diverging colormap.

        Args:
            delta: 2D array of delta values (B - A)
            spec: atlas specification
            out: output directory
            channel: channel name for title/filename
            row_labels: labels for rows (layers)
            col_labels: labels for columns (slots)
            mode: alignment mode for title
            model_a, model_b: model display names for the title
            render_profile: if True, also render 1×L profile strip
            noise_floor_dir: optional directory of a calibration compare job;
                cells with |delta| at or below the calibration |delta| get a
                grey noise-floor veil (alpha=0.25) on the sheets
        """
        out.mkdir(parents=True, exist_ok=True)
        produced: list[Path] = []

        # M9 noise-floor veil: composed at the raw field level, before column
        # dropping, so the floor mask tracks the original grid.
        noise_floor_mask: np.ndarray | None = None
        if noise_floor_dir is not None:
            noise_floor_mask = _noise_floor_mask(delta, noise_floor_dir, channel)

        # Drop all-NaN columns (slots missing in one or both models) so the
        # sheet compresses horizontally. ``kept_cols`` stores the mapping of
        # original column indices that survive, so the caller can translate
        # back (e.g. for the hotspot slot names).
        valid_cols = ~np.isnan(delta).all(axis=0)
        self.kept_cols = [int(i) for i in np.where(valid_cols)[0]]
        # Column labels must track the real field width. A caller may pass
        # spec.slots from a newer spec than the scan used; truncate to the
        # actual column count so zip(..., strict=True) cannot raise.
        if col_labels is not None and len(col_labels) != delta.shape[1]:
            col_labels = list(col_labels[:delta.shape[1]])
        if not valid_cols.any():
            # All columns NaN: keep as-is so the renderer still emits a sheet.
            data = delta
        elif valid_cols.all():
            data = delta
        else:
            data = delta[:, valid_cols]
            if col_labels is not None:
                col_labels = [label for label, keep in zip(col_labels, valid_cols, strict=True) if keep]
            if noise_floor_mask is not None:
                # Column-dropped sheet: align the veil mask to the kept columns.
                noise_floor_mask = noise_floor_mask[:, valid_cols]

        # Compute symmetric limits using diverging_clip quantile
        diverging_clip = _get_diverging_clip(spec)
        vmax = _compute_vmax(data, diverging_clip)

        # Render main delta sheet
        sheet_path = self._render_sheet(data, spec, out, channel, row_labels, col_labels, mode, vmax, model_a, model_b, noise_floor_mask)
        produced.append(sheet_path)

        # Render profile strip if requested
        if render_profile:
            profile_path = self._render_profile(data, spec, out, channel, mode, model_a, model_b)
            produced.append(profile_path)

        return produced

    def _render_sheet(
        self,
        data: np.ndarray,
        spec: AtlasSpec,
        out: Path,
        channel: str,
        row_labels: list[str] | None,
        col_labels: list[str] | None,
        mode: str,
        vmax: float,
        model_a: str = "",
        model_b: str = "",
        noise_floor_mask: np.ndarray | None = None,
    ) -> Path:
        """Render the main delta sheet."""
        dpi = int(spec.sheet["dpi"])
        n_rows, n_cols = data.shape
        figsize = (max(6, n_cols * 0.5), max(4, n_rows * 0.4))

        fig, ax = plt.subplots(figsize=figsize)

        # TwoSlopeNorm: blue (negative) → white (zero) → red (positive)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if vmax > 0 else CenteredNorm()

        im = ax.imshow(
            data,
            cmap="RdBu_r",
            norm=norm,
            origin="upper",
            extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5),
            aspect="auto",
        )

        # M9 noise-floor veil: grey shading over cells at/below the calibration
        # |delta|, composed on top of the delta image (never stacked with any
        # other veil — the noise-floor mask is the only veil here).
        if noise_floor_mask is not None and noise_floor_mask.any():
            shade = np.where(noise_floor_mask & np.isfinite(data), 1.0, np.nan)
            ax.imshow(
                shade,
                cmap="Greys", vmin=0, vmax=1, origin="upper",
                extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5),
                alpha=0.25, zorder=2,
            )

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Δ (B − A)")

        # Labels
        ax.set_xlabel("slot")
        ax.set_ylabel("layer")

        if col_labels:
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(col_labels, rotation=90, fontsize=6)
        else:
            ax.set_xticks(range(n_cols))

        if row_labels:
            # Show every nth label if many rows
            step = max(1, len(row_labels) // 20)
            ticks = list(range(0, len(row_labels), step))
            ax.set_yticks(ticks)
            ax.set_yticklabels([row_labels[i] for i in ticks], fontsize=6)
        else:
            ax.set_yticks(range(n_rows))

        title = f"Δ {channel} – {mode}"
        if model_a and model_b:
            title += f" ({model_a} vs {model_b})"
        if noise_floor_mask is not None and noise_floor_mask.any():
            title += " · noise-floor veiled"
        ax.set_title(title)

        delta_path = out / f"delta_sheet_{channel}.png"
        fig.savefig(delta_path, dpi=dpi, bbox_inches="tight", metadata=_PNG_METADATA)
        plt.close(fig)
        return delta_path

    def _render_profile(
        self,
        data: np.ndarray,
        spec: AtlasSpec,
        out: Path,
        channel: str,
        mode: str,
        model_a: str = "",
        model_b: str = "",
    ) -> Path:
        """Render a 1×L profile strip (per-layer relative L2) — the 'ablitation bar'."""
        dpi = int(spec.sheet["dpi"])
        n_rows, n_cols = data.shape

        # Compute per-row relative L2 (relative to row norm)
        profile = np.full(n_rows, np.nan, dtype=np.float64)
        for i in range(n_rows):
            row = data[i, :]
            finite_row = row[np.isfinite(row)]
            if finite_row.size > 0:
                profile[i] = float(np.linalg.norm(finite_row) / np.sqrt(finite_row.size))

        fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.5), 1.0))

        # Scale the strip to the profile's own values, NOT the sheet's
        # cell-level vmax: a channel whose bulk is unchanged (many ~zero
        # cells) collapses the sheet vmax to a tiny value, and reusing it here
        # would saturate every bar to the top of the "hot" colormap (white).
        profile_vmax = _compute_vmax(profile.reshape(-1, 1), _get_diverging_clip(spec))

        # Plot as 1×L strip using hot colormap
        profile_2d = profile.reshape(1, -1)
        norm = TwoSlopeNorm(vmin=0, vcenter=profile_vmax / 2, vmax=profile_vmax) if profile_vmax > 0 else CenteredNorm()

        ax.imshow(
            profile_2d,
            cmap="hot",
            norm=norm,
            origin="upper",
            extent=(-0.5, n_cols - 0.5, -0.5, 0.5),
            aspect="auto",
        )

        ax.set_xlabel("slot")
        ax.set_yticks([])
        title = f"Δ profile {channel} – {mode}"
        if model_a and model_b:
            title += f" ({model_a} vs {model_b})"
        ax.set_title(title)

        profile_path = out / f"delta_profile_{channel}.png"
        fig.savefig(profile_path, dpi=dpi, bbox_inches="tight", metadata=_PNG_METADATA)
        plt.close(fig)
        return profile_path
