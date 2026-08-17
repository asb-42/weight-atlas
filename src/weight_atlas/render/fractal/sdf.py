"""Deterministic Signed Distance Field fractals in pure NumPy.

Two families for the SDF mode of the fractal renderer:

- Menger sponge (fold-based SDF, ``menger_sdf``)
- Mandelbulb (spherical-pow distance estimator, ``mandelbulb_sdf``)

Both are evaluated on a fixed 3D lattice and then turned into a watertight
triangle mesh by :mod:`surface_nets`. Determinism contract: pure arithmetic,
fixed iteration counts, no RNG, no timestamps — identical inputs produce
byte-identical fields, meshes, PNGs and OBJs.
"""

from __future__ import annotations

import numpy as np


def sd_box(p: np.ndarray, half: float = 1.0) -> np.ndarray:
    """Signed distance to an axis-aligned box (used by the Menger sponge)."""
    q = np.abs(p) - half
    outside = np.sqrt(np.sum(np.maximum(q, 0.0) ** 2, axis=-1))
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return np.asarray(outside + inside, dtype=np.float64)


def menger_sdf(points: np.ndarray, iterations: int, scale: float = 3.0) -> np.ndarray:
    """Signed distance to a Menger sponge (classic box-fold).

    ``points`` is a (…, 3) float64 array. ``iterations`` sets the recursion
    depth, ``scale`` the fold scale (3.0 = classic Menger). Deterministic.
    """
    d = sd_box(points, 1.0)
    s = 1.0
    for _ in range(max(1, int(iterations))):
        a = np.mod(points * s, 2.0) - 1.0
        s = s * scale
        r = np.abs(1.0 - scale * np.abs(a))
        da = np.maximum(r[..., 0], r[..., 1])
        db = np.maximum(r[..., 1], r[..., 2])
        dc = np.maximum(r[..., 2], r[..., 0])
        c = (np.minimum(da, np.minimum(db, dc)) - 1.0) / s
        d = np.maximum(d, c)
    return np.asarray(d, dtype=np.float64)


def mandelbulb_sdf(points: np.ndarray, power: float, iterations: int) -> np.ndarray:
    """Distance estimate of a Mandelbulb (spherical pow, classical DE).

    ``points`` is a (…, 3) float64 array. ``power`` is the exponent,
    ``iterations`` the bail-out count. Points that leave the bail-out radius
    freeze at their last DE (masked iteration), so the computation stays
    finite regardless of ``power``. Fully vectorised, element-wise
    independent → deterministic.
    """
    power = float(power)
    n_iter = max(1, int(iterations))
    bailout = 2.0
    z = np.asarray(points, dtype=np.float64)
    dr = np.ones(z.shape[:-1], dtype=np.float64)
    r = np.zeros(z.shape[:-1], dtype=np.float64)
    zz = np.zeros_like(z)
    alive = np.ones(z.shape[:-1], dtype=bool)
    result = np.zeros(z.shape[:-1], dtype=np.float64)

    for _ in range(n_iter):
        r = np.sqrt(np.sum(z * z, axis=-1))
        cur = alive & (r < bailout)
        if not cur.any():
            break
        rs = np.where(cur, np.maximum(r, 1e-12), 1.0)
        theta = np.arccos(np.clip(np.where(cur, z[..., 2], 0.0) / rs, -1.0, 1.0))
        phi = np.arctan2(np.where(cur, z[..., 1], 0.0), np.where(cur, z[..., 0], 1.0))
        zr = np.power(rs, power - 1.0)
        dr = np.where(cur, zr * power * dr + 1.0, dr)
        theta = theta * power
        phi = phi * power
        zr_full = np.power(rs, power)
        zz[..., 0] = np.where(cur, zr_full * np.sin(theta) * np.cos(phi), 0.0)
        zz[..., 1] = np.where(cur, zr_full * np.sin(theta) * np.sin(phi), 0.0)
        zz[..., 2] = np.where(cur, zr_full * np.cos(theta), 0.0)
        z = zz + points
        result = np.where(cur, 0.5 * np.log(np.maximum(rs, 1e-12)) * rs / np.maximum(dr, 1e-12), result)
        alive = cur

    # Finalise any still-alive points (inside the set) with their last DE.
    r_safe = np.maximum(r, 1e-12)
    result = np.where(alive & (r >= 1e-12), 0.5 * np.log(r_safe) * r / np.maximum(dr, 1e-12), result)
    result = np.where(alive & (r < 1e-12), -0.5, result)
    return np.asarray(result, dtype=np.float64)


_SDF_FAMILIES = ("menger", "mandelbulb")


def sdf_volume(family: str, params: dict, grid: int, extent: float = 1.35) -> np.ndarray:
    """Evaluate an SDF family on a ``(grid+1)³`` lattice in ``[-extent, extent]³``.

    ``params`` carries the family's parameters (``iterations`` and either
    ``scale`` for menger or ``power`` for mandelbulb). ``extent`` is larger
    than the fractal's bounding box (Menger: 1.0; Mandelbulb: ~1.0 for
    moderate power), so the iso-surface stays strictly inside the sampled
    volume and the extracted mesh is watertight. Returns a float64 array of
    shape ``(grid+1, grid+1, grid+1)`` with signed distances. Deterministic.
    """
    n = int(grid) + 1
    axis = np.linspace(-extent, extent, n, dtype=np.float64)
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.stack((xx, yy, zz), axis=-1)
    iterations = int(params.get("iterations", 3))
    if family == "menger":
        scale = float(params.get("scale", 3.0))
        return menger_sdf(coords, iterations, scale)
    if family == "mandelbulb":
        power = float(params.get("power", 6.0))
        return mandelbulb_sdf(coords, power, iterations)
    raise ValueError(f"unknown SDF family: {family!r} (expected {_SDF_FAMILIES})")
