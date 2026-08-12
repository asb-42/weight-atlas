"""Float32 TIFF I/O – byte-deterministic via tifffile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def write_tif(path: Path, field: np.ndarray) -> None:
    """Write a 2D float32 field to TIFF. Deterministic: no metadata."""
    # Ensure C-contiguous float32.
    arr = np.ascontiguousarray(field.astype(np.float32))
    tifffile.imwrite(path, arr, metadata=None)


def read_tif(path: Path) -> np.ndarray:
    """Read a float32 TIFF back to a 2D array."""
    return tifffile.imread(path).astype(np.float64)
