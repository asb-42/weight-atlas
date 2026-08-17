"""SDF mosaic mesh builder (per-slot mini-SDF objects).

The SDF mode renders a *mosaic* of small 3D fractal objects: one mini-SDF
(Menger sponge or Mandelbulb) per slot cell of the (layers × slots) raster,
each parameterised by its own slot's statistics. The per-cell meshes are
extracted with ``surface_nets`` and merged into a single triangle mesh that
the ``render_sdf.py`` Blender script renders like a sculpture garden.

Determinism: the SDF evaluation and iso-surface extraction are pure NumPy
(fixed lattice, no RNG) → byte-identical mesh for identical inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from weight_atlas.render.fractal.sdf import sdf_volume
from weight_atlas.render.fractal.surface_nets import surface_nets

# Sampling extent per family (larger than the fractal's bounding box so the
# iso-surface stays inside the sampled volume and the mesh is watertight).
_EXTENTS: dict[str, float] = {
    "menger": 1.35,
    "mandelbulb": 2.0,
}
# Object fills this fraction of its cell (gaps keep the mosaic readable).
_FILL = 0.8
# Maximum number of mini-SDF objects in one mosaic. Larger rasters (e.g. MoE
# expert panels with one column per expert) are decimated deterministically so
# the mesh stays buildable and renderable in bounded time/memory.
_DEFAULT_MAX_CELLS = 1024


def _extent_for(family: str) -> float:
    return _EXTENTS.get(family, 1.35)


def _cell_strides(n_rows: int, n_cols: int, max_cells: int) -> tuple[int, int]:
    """Deterministic (row, col) sampling strides keeping the cell count bounded.

    Returns ``(1, 1)`` when ``n_rows * n_cols <= max_cells``. Otherwise picks
    aspect-preserving strides so ``ceil(n_rows / rs) * ceil(n_cols / cs)``
    stays at or below ``max_cells``. Deterministic for a given input shape.
    """
    total = n_rows * n_cols
    if total <= max_cells:
        return 1, 1
    ratio = n_cols / max(n_rows, 1)
    row_count = max(1, math.ceil(math.sqrt(max_cells / ratio)))
    col_count = max(1, math.ceil(math.sqrt(max_cells * ratio)))
    while row_count * col_count > max_cells:
        if row_count >= col_count:
            row_count -= 1
        else:
            col_count -= 1
        row_count = max(1, row_count)
        col_count = max(1, col_count)
    return max(1, math.ceil(n_rows / row_count)), max(1, math.ceil(n_cols / col_count))


def build_sdf_mosaic(
    n_rows: int,
    n_cols: int,
    slots: Sequence[str],
    sdf_params: dict[str, dict[str, float]],
    family: str,
    grid: int,
    cell_h: int,
    cell_w: int,
    max_cells: int = _DEFAULT_MAX_CELLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the per-slot SDF mosaic mesh.

    Returns ``(verts, faces, tint)`` where ``verts`` is an (N, 3) float64
    array of mesh coordinates, ``faces`` an (M, 3) int64 triangle array, and
    ``tint`` an (N,) float64 per-vertex colour channel in [0, 1] encoding the
    slot column (so each slot's objects read as a distinct band). The mosaic
    footprint is normalised to fit [-1, 1]² in x/y (same frame as the terrain
    renders) with object proportions preserved. Deterministic.

    When the raster exceeds ``max_cells`` cells (e.g. MoE expert panels with
    one column per expert) the raster is decimated with deterministic
    aspect-preserving strides — the sampled objects keep their true (row,
    col) positions and tints, only their count is bounded.
    """
    cell_size = float(min(cell_h, cell_w))
    extent = _extent_for(family)
    row_stride, col_stride = _cell_strides(n_rows, n_cols, max_cells)

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_tint: list[np.ndarray] = []
    offset = 0

    for i in range(0, n_rows, row_stride):
        for j in range(0, n_cols, col_stride):
            params = sdf_params.get(slots[j], {})
            vol = sdf_volume(family, params, grid, extent=extent)
            v, f = surface_nets(vol)
            if len(v) == 0 or len(f) == 0:
                continue

            # Normalise the object into a unit cube so every mini-SDF fills
            # its cell regardless of family or parameters.
            v = v.astype(np.float64)
            span_v = float(np.max(v.max(axis=0) - v.min(axis=0)))
            v = (v - v.min(axis=0)) / max(span_v, 1e-9)
            v = (v - 0.5) * (_FILL * cell_size)
            v[:, 2] += 0.5 * _FILL * cell_size

            # Place the cell: row i along y, column j along x (same raster
            # orientation as the fBm field).
            v[:, 0] += (j + 0.5) * cell_w
            v[:, 1] += (i + 0.5) * cell_h

            tint = np.full(v.shape[0], (j + 0.5) / n_cols, dtype=np.float64)
            all_verts.append(v)
            all_faces.append(f.astype(np.int64) + offset)
            all_tint.append(tint)
            offset += len(v)

    if not all_verts:
        raise ValueError("SDF mosaic produced no geometry (check fractal.sdf grid/family)")

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    tint = np.concatenate(all_tint, axis=0)

    # Normalise the mosaic footprint into the [-1, 1]² render frame, keeping
    # aspect ratio and object proportions (uniform scale, then centre x/y).
    span = max(float(verts[:, 0].max() - verts[:, 0].min()),
               float(verts[:, 1].max() - verts[:, 1].min()))
    if span <= 1e-9:
        raise ValueError("SDF mosaic has no lateral extent")
    verts = (verts - verts.mean(axis=0)) / span * 2.0

    return verts, faces, tint
