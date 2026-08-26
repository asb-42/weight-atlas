"""SDF mosaic mesh builder (per-slot mini-SDF objects).

The SDF mode renders a *mosaic* of small 3D fractal objects: one mini-SDF
(Menger sponge or Mandelbulb) per slot cell of the (layers × slots) raster,
each parameterised by its own slot's statistics. The per-cell meshes are
extracted with ``surface_nets`` and merged into a single triangle mesh that
the ``render_sdf.py`` Blender script renders like a standing sculpture garden.

Per-cell character: each object is scaled and yaw-rotated by a deterministic
hash of its (row, col) lattice point (splitmix64, seedable), so the mosaic
reads as varied standing sculptures rather than a symmetric grid. Tint
encodes the slot's real statistics (normalised across slots) when a
``slot_tint`` map is supplied, falling back to the slot column index.

Determinism: the SDF evaluation, iso-surface extraction, and per-cell
variation are pure (fixed lattice, splitmix64 hash, no RNG) → byte-identical
mesh for identical inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from weight_atlas.render.fractal.fbm import _hash_lattice
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
# Per-cell size factor bounds (deterministic, keyed on the cell lattice point).
_SIZE_LO, _SIZE_HI = 0.6, 1.4
# Target vertical relief: the tallest object reaches this fraction of the
# [-1, 1]² render frame *before* the render script's z_scale exaggeration.
# With the default blender z_scale (0.3) the tallest object stands ~0.3 units
# tall in the frame — clearly visible, comparable to the fBm terrain's relief.
_DEFAULT_RELIEF = 1.0


def _extent_for(family: str) -> float:
    return _EXTENTS.get(family, 1.35)


def _cell_strides(n_rows: int, n_cols: int, max_cells: int) -> tuple[int, int]:
    """Deterministic (row, col) sampling strides keeping the cell count bounded.

    Returns ``(1, 1)`` when ``n_rows * n_cols <= max_cells``. Otherwise picks
    aspect-preserving strides so ``ceil(n_rows / rs) * ceil(n_cols / cs)``
    stays at or below ``max_cells``. Deterministic for a given input shape.
    """
    total = n_rows * n_cols
    # A non-positive budget can never be satisfied by the loop below (counts
    # floor at 1) — it would spin the worker forever. Clamp to at least one.
    max_cells = max(1, int(max_cells))
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
    seed: int = 0,
    variation: bool = True,
    relief: float = _DEFAULT_RELIEF,
    slot_tint: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the per-slot SDF mosaic mesh.

    Returns ``(verts, faces, tint)`` where ``verts`` is an (N, 3) float64
    array of mesh coordinates, ``faces`` an (M, 3) int64 triangle array, and
    ``tint`` an (N,) float64 per-vertex colour channel in [0, 1]. The mosaic
    footprint is normalised to fit [-1, 1]² in x/y (same frame as the terrain
    renders); objects stand upright with real vertical relief (the tallest
    reaches ``relief`` in the frame), so the mosaic reads as a sculpture
    garden rather than a flat grid. Deterministic.

    Per-cell character:
    - ``variation`` (default True) scales each object by a deterministic
      factor in [0.6, 1.4] and yaw-rotates it around its own axis, both keyed
      on the cell's (row, col) lattice point and the base ``seed`` — breaks
      the symmetric-grid look while staying reproducible.
    - ``slot_tint`` maps slot → colour value in [0, 1] (e.g. a normalised
      statistic); when omitted the tint falls back to the slot column index
      so each column reads as a band.

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

            # Deterministic per-cell variation: scale factor + yaw rotation,
            # both derived from the cell's (row, col) lattice hash.
            size = 1.0
            yaw = 0.0
            if variation:
                h = _hash_lattice(
                    np.array([i], dtype=np.int64), np.array([j], dtype=np.int64), seed
                )[0]
                size = _SIZE_LO + float(h) * (_SIZE_HI - _SIZE_LO)
                h2 = _hash_lattice(
                    np.array([i], dtype=np.int64), np.array([j], dtype=np.int64), seed + 1
                )[0]
                yaw = float(h2) * 2.0 * math.pi

            v = (v - 0.5) * (_FILL * cell_size * size)
            v[:, 2] += 0.5 * _FILL * cell_size * size

            # Yaw around the object's own vertical axis (after centring, so
            # rotation keeps the object centred in its cell).
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            x, y = v[:, 0].copy(), v[:, 1].copy()
            v[:, 0] = x * cos_yaw - y * sin_yaw
            v[:, 1] = x * sin_yaw + y * cos_yaw

            # Place the cell: row i along y, column j along x (same raster
            # orientation as the fBm field).
            v[:, 0] += (j + 0.5) * cell_w
            v[:, 1] += (i + 0.5) * cell_h

            if slot_tint is not None:
                value = slot_tint.get(slots[j])
                tint_val = float(value) if value is not None and np.isfinite(value) else 0.5
                tint = np.full(v.shape[0], min(max(tint_val, 0.0), 1.0), dtype=np.float64)
            else:
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

    # Normalise the mosaic footprint into the [-1, 1]² render frame: x/y keep
    # aspect and centre, while z keeps real vertical relief (objects stand).
    span = max(float(verts[:, 0].max() - verts[:, 0].min()),
               float(verts[:, 1].max() - verts[:, 1].min()))
    if span <= 1e-9:
        raise ValueError("SDF mosaic has no lateral extent")
    verts[:, 0] = (verts[:, 0] - verts[:, 0].mean()) / span * 2.0
    verts[:, 1] = (verts[:, 1] - verts[:, 1].mean()) / span * 2.0
    z_max = float(verts[:, 2].max())
    if z_max > 1e-9:
        verts[:, 2] = verts[:, 2] / z_max * relief

    return verts, faces, tint
