"""UMAP projection of token embeddings (optional, requires umap-learn)."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    seed: int = 0,
    init: np.ndarray | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute UMAP projection.

    Args:
        embeddings: (V, D) array of token embeddings
        n_components: Number of output dimensions
        seed: Random seed
        init: Initial embedding (e.g., from PCA)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter

    Returns:
        Tuple of (projected, metadata)
        - projected: (V, n_components) projected coordinates
        - metadata: dict with umap version and parameters
    """
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "umap-learn is required for UMAP projection. "
            "Install with: pip install -e '.[umap]'"
        ) from None

    reducer = umap.UMAP(
        n_components=n_components,
        random_state=seed,
        init=init if init is not None else "random",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )

    projected = reducer.fit_transform(embeddings)

    metadata = {
        "method": "umap",
        "version": umap.__version__,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "seed": seed,
    }

    return projected, metadata
