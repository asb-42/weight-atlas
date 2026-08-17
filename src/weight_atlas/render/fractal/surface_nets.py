"""Deterministic iso-surface extraction (naive Surface Nets) in pure NumPy.

Turns a sampled signed-distance field into a watertight triangle mesh without
external lookup tables or libraries. For a closed surface strictly inside the
sampled volume, every surface-crossing grid edge yields exactly one quad, so
the result is a closed 2-manifold — byte-identical for identical inputs.

Winding: each quad is oriented so its normal points from the "inside" corner
of the crossed edge toward the "outside" corner, which yields globally
outward-facing normals on a closed surface. Verified in tests (sphere normals
point away from the centroid).
"""

from __future__ import annotations

import numpy as np


def surface_nets(volume: np.ndarray, iso: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Extract a watertight triangle mesh from a sampled scalar field.

    ``volume`` is a float64 ``(nx, ny, nz)`` array of signed distances; the
    iso-surface is ``value == iso``. Returns ``(verts, faces)`` with verts in
    the same coordinate space as the sampling lattice (each sample at integer
    index → unit-spaced positions) and faces as 1-indexed-free triangle index
    triples into ``verts``.
    """
    vol = np.asarray(volume, dtype=np.float64)
    nx, ny, nz = vol.shape
    inside = vol < iso

    ncx = nx - 1
    ncy = ny - 1
    ncz = nz - 1

    # ---- Vertices: one per cell whose 8 corners straddle the iso ----
    cell_idx = np.full((ncx, ncy, ncz), -1, dtype=np.int64)
    verts: list[tuple[float, float, float]] = []
    for cx in range(ncx):
        for cy in range(ncy):
            for cz in range(ncz):
                corners = inside[cx : cx + 2, cy : cy + 2, cz : cz + 2]
                if corners.all() or not corners.any():
                    continue
                idx = len(verts)
                cell_idx[cx, cy, cz] = idx
                verts.append(_cell_vertex(vol, inside, cx, cy, cz, iso))

    # ---- Faces: one quad per surface-crossing edge ----
    faces: list[tuple[int, int, int]] = []

    def _quad_cells(
        i: int, j: int, k: int
    ) -> tuple[int, int, int, int] | None:
        """4 cells around the x-edge at (i,j,k); None if any out of bounds."""
        if j < 1 or k < 1 or j >= ncy or k >= ncz or i < 0 or i >= ncx:
            return None
        return (
            int(cell_idx[i, j - 1, k - 1]),
            int(cell_idx[i, j, k - 1]),
            int(cell_idx[i, j, k]),
            int(cell_idx[i, j - 1, k]),
        )

    def _quad_cells_y(
        i: int, j: int, k: int
    ) -> tuple[int, int, int, int] | None:
        """4 cells around the y-edge at (i,j,k)."""
        if i < 1 or k < 1 or i >= ncx or k >= ncz or j < 0 or j >= ncy:
            return None
        return (
            int(cell_idx[i - 1, j, k - 1]),
            int(cell_idx[i, j, k - 1]),
            int(cell_idx[i, j, k]),
            int(cell_idx[i - 1, j, k]),
        )

    def _quad_cells_z(
        i: int, j: int, k: int
    ) -> tuple[int, int, int, int] | None:
        """4 cells around the z-edge at (i,j,k)."""
        if i < 1 or j < 1 or i >= ncx or j >= ncy or k < 0 or k >= ncz:
            return None
        return (
            int(cell_idx[i - 1, j - 1, k]),
            int(cell_idx[i, j - 1, k]),
            int(cell_idx[i, j, k]),
            int(cell_idx[i - 1, j, k]),
        )

    def _emit(quad: tuple[int, int, int, int]) -> None:
        a, b, c, d = quad
        if min(a, b, c, d) < 0:
            return
        faces.append((a, b, c))
        faces.append((a, c, d))

    # x-edges: (i,j,k)-(i+1,j,k)
    for i in range(ncx):
        for j in range(1, ny):
            for k in range(1, nz):
                if inside[i, j, k] == inside[i + 1, j, k]:
                    continue
                quad = _quad_cells(i, j, k)
                if quad is None:
                    continue
                # Direct order → +x normal (outward when corner (i,j,k) inside).
                if not inside[i, j, k]:
                    a, b, c, d = quad
                    quad = (a, d, c, b)
                _emit(quad)

    # y-edges: (i,j,k)-(i,j+1,k)
    for i in range(1, nx):
        for j in range(ncy):
            for k in range(1, nz):
                if inside[i, j, k] == inside[i, j + 1, k]:
                    continue
                quad = _quad_cells_y(i, j, k)
                if quad is None:
                    continue
                # Reversed order → +y normal (outward when corner (i,j,k) inside).
                if inside[i, j, k]:
                    a, b, c, d = quad
                    quad = (a, d, c, b)
                _emit(quad)

    # z-edges: (i,j,k)-(i,j,k+1)
    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(ncz):
                if inside[i, j, k] == inside[i, j, k + 1]:
                    continue
                quad = _quad_cells_z(i, j, k)
                if quad is None:
                    continue
                # Direct order → +z normal (outward when corner (i,j,k) inside).
                if not inside[i, j, k]:
                    a, b, c, d = quad
                    quad = (a, d, c, b)
                _emit(quad)

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _cell_vertex(
    vol: np.ndarray,
    inside: np.ndarray,
    cx: int,
    cy: int,
    cz: int,
    iso: float,
) -> tuple[float, float, float]:
    """Vertex position for a straddling cell: average of its edge crossings.

    A cell with mixed corner signs has at least one crossed edge; averaging the
    linear-interpolated crossing points of all its crossed edges places the
    vertex on the iso-surface (smoother than the plain cell centre).
    Deterministic.
    """
    pts: list[tuple[float, float, float]] = []
    # 12 edges of the cell.
    edges = [
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (0, 0, 1)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 0, 0), (1, 0, 1)),
        ((0, 1, 0), (1, 1, 0)),
        ((0, 1, 0), (0, 1, 1)),
        ((0, 0, 1), (1, 0, 1)),
        ((0, 0, 1), (0, 1, 1)),
        ((1, 1, 0), (1, 1, 1)),
        ((1, 0, 1), (1, 1, 1)),
        ((0, 1, 1), (1, 1, 1)),
    ]
    for (e0, e1) in edges:
        p0 = (cx + e0[0], cy + e0[1], cz + e0[2])
        p1 = (cx + e1[0], cy + e1[1], cz + e1[2])
        v0 = float(vol[p0])
        v1 = float(vol[p1])
        if (v0 < iso) == (v1 < iso):
            continue
        denom = v1 - v0
        t = 0.5 if abs(denom) < 1e-15 else (iso - v0) / denom
        t = float(np.clip(t, 0.0, 1.0))
        pts.append(
            (
                p0[0] + t * (p1[0] - p0[0]),
                p0[1] + t * (p1[1] - p0[1]),
                p0[2] + t * (p1[2] - p0[2]),
            )
        )
    if not pts:
        # Fallback (should be unreachable for straddling cells).
        return (cx + 0.5, cy + 0.5, cz + 0.5)
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
        sum(p[2] for p in pts) / len(pts),
    )
