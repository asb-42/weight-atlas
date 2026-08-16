"""Paired sheet renderers (M9): quant-impact + edit-signature sheets.

Two registered renderers share the fixed-anchor display rule (bypass
``filled_norm``/``per_row_normalize`` so absolute anchors survive display):

- ``ImpactSheet`` (``"impact"``): SQNR sheets + qtype map + profile, using the
  ``qimpact.db_range`` fixed dB anchors and ``qimpact.colormap``.
- ``EditSheet`` (``"edit"``): rel-L2 sheet with fixed *log* anchors
  (``edit.rel_l2_log_range``), a delta-stable-rank sheet
  (``edit.rank_log_range``) and the per-layer median rel-L2 profile strip
  (the abliteration bar). PNG metadata stays byte-deterministic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 – must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from weight_atlas.core.registry import register_renderer
from weight_atlas.core.types import AtlasSpec
from weight_atlas.fields.scaling import fixed_anchor
from weight_atlas.fields.tif_io import read_tif

# Fixed PNG metadata per spec (byte-deterministic).
_PNG_METADATA = {
    "Software": "weight-atlas",
    "Creation Time": "1970-01-01T00:00:00Z",
}

# Bound the sheet raster to a fixed pixel budget so the dpi-scaled RGBA buffer
# stays small even for huge smooth fields (same guard as the scan sheet
# renderer). Without this, a 256x536 smooth field would allocate a
# ~40k x 15k px figure and take tens of seconds to draw.
_MAX_RENDER_PIXELS = 12_000_000


def _figsize(n_rows: int, n_cols: int, dpi: float) -> tuple[float, float]:
    """Cap the figure size so n_rows*n_cols <= the pixel budget."""
    scale = math.sqrt(_MAX_RENDER_PIXELS / max(1, n_rows * n_cols))
    px_h = max(2, int(round(n_rows * scale)))
    px_w = max(2, int(round(n_cols * scale)))
    return (max(6.0, px_w / dpi), max(4.0, px_h / dpi))


def _paired_cmap(spec: AtlasSpec, block: dict, default: str) -> LinearSegmentedColormap:
    """Build the preset colormap (magma_r default) from the spec block."""
    name = block.get("colormap", default)
    try:
        base = plt.get_cmap(name)
    except ValueError:
        base = plt.get_cmap(default)
    colors = [base(i) for i in np.linspace(0, 1, 256)]
    return LinearSegmentedColormap.from_list(f"paired_{name}", colors, N=256)


def _db_range(spec: AtlasSpec) -> tuple[float, float]:
    """Fixed display anchor range from ``qimpact.db_range`` (default 5–60 dB)."""
    db = (spec.qimpact or {}).get("db_range", [5, 60])
    return float(db[0]), float(db[1])


def _log_anchor_range(block: dict, key: str, default: list[float]) -> tuple[float, float]:
    """Fixed log anchor range from a spec block (e.g. ``edit.rel_l2_log_range``)."""
    rng = block.get(key, default)
    return float(rng[0]), float(rng[1])


def _load_smooth(out_dir: Path, name: str) -> np.ndarray:
    """Load the smooth TIFF for a paired field, falling back to raw."""
    smooth = out_dir / f"{name}_smooth.tif"
    if smooth.exists():
        return read_tif(smooth)
    return read_tif(out_dir / f"{name}_raw.tif")


def _render_paired_sheet(
    data: np.ndarray,
    spec: AtlasSpec,
    out: Path,
    *,
    title: str,
    cmap: LinearSegmentedColormap,
    filename: str,
    display: np.ndarray,
    cbar_label: str,
    tag: str,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
) -> Path:
    """Render a fixed-anchor paired field as a sheet."""
    dpi = int(spec.sheet["dpi"])
    n_rows, n_cols = data.shape
    figsize = _figsize(n_rows, n_cols, dpi)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=1, origin="upper", aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xlabel("slot")
    ax.set_ylabel("layer")

    if col_labels:
        step = max(1, n_cols // 20)
        ticks = [i for i in range(0, n_cols, step) if i < len(col_labels)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([col_labels[i] for i in ticks], rotation=90, fontsize=6)
    if row_labels:
        step = max(1, n_rows // 20)
        ticks = [i for i in range(0, n_rows, step) if i < len(row_labels)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([row_labels[i] for i in ticks], fontsize=6)

    ax.set_title(f"{title} · {tag}", fontsize=12, fontweight="bold")

    path = out / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata=_PNG_METADATA)
    plt.close(fig)
    return path


def _render_profile_strip(
    data: np.ndarray,
    spec: AtlasSpec,
    out: Path,
    *,
    title: str,
    filename: str,
    tag: str,
    log: bool = False,
) -> Path:
    """Render a 1×L per-layer median rel-L2 profile strip (hot colormap)."""
    dpi = int(spec.sheet["dpi"])
    n_rows, n_cols = data.shape

    profile = np.full(n_rows, np.nan, dtype=np.float64)
    for i in range(n_rows):
        row = data[i, :]
        finite = row[np.isfinite(row)]
        if finite.size > 0:
            profile[i] = float(np.median(np.abs(finite)))
    if log:
        with np.errstate(divide="ignore"):
            profile = np.log10(profile)

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.5), 1.0))
    finite = np.isfinite(profile)
    if log and finite.any():
        vmin = float(np.nanmin(profile))
        vmax = float(np.nanmax(profile))
    else:
        vmin, vmax = 0.0, float(np.nanmax(profile)) if finite.any() else 1.0
    if not vmax > vmin:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(
        profile.reshape(1, -1),
        cmap="hot",
        norm=norm,
        origin="upper",
        extent=(-0.5, n_cols - 0.5, -0.5, 0.5),
        aspect="auto",
    )
    ax.set_xlabel("layer")
    ax.set_yticks([])
    ax.set_title(f"{title} profile · {tag}", fontsize=10, fontweight="bold")
    path = out / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata=_PNG_METADATA)
    plt.close(fig)
    return path


def _render_qtype_map(
    out_dir: Path,
    spec: AtlasSpec,
    out: Path,
    *,
    codes: dict[str, int],
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
) -> Path:
    """Render a discrete quantization-type map with a legend."""
    dpi = int(spec.sheet["dpi"])
    raw_path = out_dir / "field_qtype_raw.tif"
    if not raw_path.exists():
        return out / "qtype_map.png"  # placeholder; not produced
    grid = read_tif(raw_path)
    n_rows, n_cols = grid.shape

    # Build a qualitative colormap keyed by type code.
    n_types = max(len(codes), 1)
    base = plt.get_cmap("tab20")
    colors = [base(i % 20) for i in range(n_types)]
    while len(colors) < 2:  # matplotlib needs >= 2 mapping points
        colors.append(colors[-1])
    cmap = LinearSegmentedColormap.from_list("qtype", colors, N=max(n_types, 2))

    fig, ax = plt.subplots(figsize=_figsize(n_rows, n_cols, dpi))
    im = ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=n_types - 0.5, origin="upper", aspect="auto")
    ax.set_xlabel("slot")
    ax.set_ylabel("layer")
    if col_labels:
        step = max(1, n_cols // 20)
        ticks = [i for i in range(0, n_cols, step) if i < len(col_labels)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([col_labels[i] for i in ticks], rotation=90, fontsize=6)
    if row_labels:
        step = max(1, n_rows // 20)
        ticks = [i for i in range(0, n_rows, step) if i < len(row_labels)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([row_labels[i] for i in ticks], fontsize=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [t for t, _ in sorted(codes.items(), key=lambda kv: kv[1])]
    cbar.set_ticks(range(n_types))
    cbar.set_ticklabels(labels, fontsize=6)
    cbar.set_label("quantization type (non-reference)", fontsize=8)
    ax.set_title("quantization type map · q-impact", fontsize=12, fontweight="bold")

    path = out / "qtype_map.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata=_PNG_METADATA)
    plt.close(fig)
    return path


@register_renderer("impact")
class ImpactSheet:
    """Renders M9 quantization-impact sheets from a paired quant output dir."""

    renderer_id = "impact"

    def render(self, out_dir: Path, spec: AtlasSpec, out: Path) -> list[Path]:
        """Render impact PNGs from a paired quant-run output directory."""
        out.mkdir(parents=True, exist_ok=True)
        produced: list[Path] = []
        cmap = _paired_cmap(spec, spec.qimpact or {}, "magma_r")

        sqnr = _load_smooth(out_dir, "field_impact_sqnr_db")
        rel_l2 = _load_smooth(out_dir, "field_impact_rel_l2")

        row_labels = [str(i) for i in range(sqnr.shape[0])]
        col_labels = list(spec.slots)

        vmin, vmax = _db_range(spec)
        produced.append(
            _render_paired_sheet(
                sqnr, spec, out,
                title="SQNR",
                cmap=cmap,
                filename="impact_sqnr_db.png",
                display=fixed_anchor(sqnr, vmin, vmax),
                cbar_label=f"SQNR (dB, fixed [{vmin:.0f}, {vmax:.0f}])",
                tag="q-impact",
                row_labels=row_labels,
                col_labels=col_labels,
            )
        )
        produced.append(
            _render_paired_sheet(
                rel_l2, spec, out,
                title="rel-L2",
                cmap=cmap,
                filename="impact_rel_l2.png",
                display=fixed_anchor(rel_l2, vmin, vmax),
                cbar_label=f"rel-L2 (dB, fixed [{vmin:.0f}, {vmax:.0f}])",
                tag="q-impact",
                row_labels=row_labels,
                col_labels=col_labels,
            )
        )

        produced.append(
            _render_profile_strip(
                rel_l2, spec, out,
                title="rel-L2",
                filename="impact_profile.png",
                tag="q-impact",
            )
        )

        codes_path = out_dir / "qtype_map.json"
        codes: dict[str, int] = {}
        if codes_path.exists():
            codes = {str(k): int(v) for k, v in json.loads(codes_path.read_text()).items()}
        produced.append(
            _render_qtype_map(
                out_dir, spec, out,
                codes=codes,
                row_labels=row_labels,
                col_labels=col_labels,
            )
        )

        return produced


@register_renderer("edit")
class EditSheet:
    """Renders M9 edit-signature sheets from a paired edit output dir.

    Uses fixed *log* anchors (rel-L2 and delta-stable-rank), so the absolute
    ranges survive display exactly like the quant preset's fixed dB anchors.
    """

    renderer_id = "edit"

    def render(self, out_dir: Path, spec: AtlasSpec, out: Path) -> list[Path]:
        """Render edit PNGs from a paired edit-run output directory."""
        out.mkdir(parents=True, exist_ok=True)
        produced: list[Path] = []
        block = spec.edit or {}
        cmap = _paired_cmap(spec, block, "magma_r")

        rel_l2 = _load_smooth(out_dir, "field_edit_rel_l2")
        rank = _load_smooth(out_dir, "field_edit_delta_stable_rank")

        row_labels = [str(i) for i in range(rel_l2.shape[0])]
        col_labels = list(spec.slots)

        vmin, vmax = _log_anchor_range(block, "rel_l2_log_range", [-4.0, -0.5])
        with np.errstate(divide="ignore"):
            log_rel = np.log10(rel_l2)
        produced.append(
            _render_paired_sheet(
                rel_l2, spec, out,
                title="rel-L2",
                cmap=cmap,
                filename="edit_rel_l2.png",
                display=fixed_anchor(log_rel, vmin, vmax),
                cbar_label=f"rel-L2 (log10, fixed [{vmin:.0f}, {vmax:.0f}])",
                tag="edit",
                row_labels=row_labels,
                col_labels=col_labels,
            )
        )

        rmin, rmax = _log_anchor_range(block, "rank_log_range", [-1.0, 3.0])
        with np.errstate(divide="ignore"):
            log_rank = np.log10(rank)
        produced.append(
            _render_paired_sheet(
                rank, spec, out,
                title="Δ stable rank",
                cmap=cmap,
                filename="edit_rank.png",
                display=fixed_anchor(log_rank, rmin, rmax),
                cbar_label=f"Δ stable rank (log10, fixed [{rmin:.0f}, {rmax:.0f}])",
                tag="edit",
                row_labels=row_labels,
                col_labels=col_labels,
            )
        )

        produced.append(
            _render_profile_strip(
                rel_l2, spec, out,
                title="rel-L2",
                filename="edit_profile.png",
                tag="edit",
                log=True,
            )
        )

        return produced
