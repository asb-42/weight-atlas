"""Deterministic fractional Brownian motion (fBm) in pure NumPy.

Used by the fractal terrain renderer: tensor statistics are mapped to fBm
parameters (octaves, persistence, lacunarity, base frequency) per slot, and
the resulting height field replaces the plain heightmap with genuinely
fractal, self-similar geometry derived from the real model data.

Determinism contract: value noise uses a fixed integer-lattice hash (no RNG,
no timestamps), so identical inputs produce byte-identical arrays.
"""

from __future__ import annotations

import numpy as np

_K1 = 0x9E3779B97F4A7C15
_K2 = 0xBF58476D1CE4E5B9
_UINT64_MAX = float(0xFFFFFFFFFFFFFFFF)


def _hash_lattice(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic splitmix64-style hash of an integer lattice point.

    ``ix``/``iy`` are int64 arrays of lattice indices, ``seed`` a fixed
    integer. Returns a float64 array in [0, 1). Pure integer arithmetic (all
    in uint64), so the result is stable across platforms and Blender's
    bundled Python.
    """
    seed64 = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    x = (ix.astype(np.uint64) * np.uint64(_K1)) + (iy.astype(np.uint64) * np.uint64(_K2))
    z = x + seed64
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_K2)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_K1)
    z = z ^ (z >> np.uint64(31))
    return (z.astype(np.float64) / _UINT64_MAX)


def value_noise(coords: np.ndarray, seed: int) -> np.ndarray:
    """Bilinear-interpolated value noise on a unit lattice.

    ``coords`` is a (…, 2) float64 array of lattice coordinates. Returns
    noise in [0, 1) with smoothstep interpolation between lattice values.
    Deterministic for a fixed ``seed``.
    """
    x = coords[..., 0]
    y = coords[..., 1]
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = x - ix
    fy = y - iy
    u = fx * fx * (3.0 - 2.0 * fx)
    v = fy * fy * (3.0 - 2.0 * fy)

    a = _hash_lattice(ix, iy, seed)
    b = _hash_lattice(ix + 1, iy, seed)
    c = _hash_lattice(ix, iy + 1, seed)
    d = _hash_lattice(ix + 1, iy + 1, seed)

    xblend = a * (1.0 - u) + b * u
    xbr = c * (1.0 - u) + d * u
    out = xblend * (1.0 - v) + xbr * v
    return np.asarray(out, dtype=np.float64)


def fbm(
    coords: np.ndarray,
    octaves: int,
    persistence: float,
    lacunarity: float,
    base_freq: float,
    seed: int,
) -> np.ndarray:
    """Fractional Brownian motion: sum of octaves of value noise.

    ``coords`` is a (…, 2) float64 array of lattice coordinates. Each octave
    doubles frequency (scaled by ``lacunarity``) and halves amplitude (scaled
    by ``persistence``). Returns a float64 array (normalised to roughly
    [0, 1) — exact range depends on persistence). Deterministic.
    """
    total = np.zeros(coords.shape[:-1], dtype=np.float64)
    amp = 1.0
    freq = max(base_freq, 1e-9)
    norm = 0.0
    for o in range(octaves):
        total = total + amp * value_noise(coords * freq, seed + o * 131)
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / norm if norm > 0.0 else total


def terrain_field(
    n_rows: int,
    n_cols: int,
    params: dict[str, float],
    seed: int,
    *,
    resolution: int = 128,
) -> np.ndarray:
    """Generate an fBm height field shaped for the Blender grid.

    ``n_rows``/``n_cols`` mirror the raster (layers × slots) so the fractal
    terrain keeps the same layout proportions. ``params`` carries the fBm
    octaves/persistence/lacunarity/base_freq for the whole field. The field
    is upsampled from a low-res lattice to ``resolution``×``resolution`` via
    the deterministic noise function (fBm evaluated on a denser grid). NaN is
    never introduced; the result is finite in [0,1).
    """
    rows = np.linspace(0.0, n_rows, resolution, dtype=np.float64)
    cols = np.linspace(0.0, n_cols, resolution, dtype=np.float64)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    coords = np.stack((xx, yy), axis=-1)
    h = fbm(
        coords,
        octaves=int(params.get("octaves", 4)),
        persistence=float(params.get("persistence", 0.5)),
        lacunarity=float(params.get("lacunarity", 2.0)),
        base_freq=float(params.get("base_freq", 1.0)),
        seed=int(seed),
    )
    lo = float(h.min())
    hi = float(h.max())
    h = (h - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(h)
    return h


def slot_fractal_field(
    n_rows: int,
    n_cols: int,
    slot_params: dict[str, dict[str, float]],
    slots: list[str],
    *,
    cell_h: int = 8,
    cell_w: int = 8,
) -> np.ndarray:
    """Build a per-slot fractal mosaic field (layers × slots cells).

    Each slot column is its own fBm strip evaluated with that slot's params
    (octaves/persistence/lacunarity/base_freq/seed from the real tensor
    statistics), so the geometry inherits its character from the data instead
    of being a heightmap with a fractal texture on top. The field has shape
    ``(n_rows * cell_h, n_cols * cell_w)``; the Blender pipeline's bilinear
    resample to the render grid upsamples it to the final resolution.
    Deterministic: pure NumPy, fixed seeds, no RNG.
    """
    rows_out = n_rows * cell_h
    cols_out = n_cols * cell_w
    out = np.zeros((rows_out, cols_out), dtype=np.float64)
    y = np.linspace(0.0, n_rows, rows_out, dtype=np.float64)
    for j, slot in enumerate(slots):
        p = slot_params.get(slot, {})
        x = np.linspace(float(j), float(j + 1), cell_w, dtype=np.float64)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        coords = np.stack((xx, yy), axis=-1)
        band = fbm(
            coords,
            octaves=max(1, int(p.get("octaves", 4))),
            persistence=float(p.get("persistence", 0.5)),
            lacunarity=float(p.get("lacunarity", 2.0)),
            base_freq=float(p.get("base_freq", 1.0)),
            seed=int(p.get("seed", 0)),
        )
        lo = float(band.min())
        hi = float(band.max())
        if hi - lo > 1e-12:
            band = (band - lo) / (hi - lo)
        out[:, j * cell_w:(j + 1) * cell_w] = band
    return out
