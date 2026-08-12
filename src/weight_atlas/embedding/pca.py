"""PCA projection of token embeddings (deterministic, dependency-free).

Uses randomized SVD with seeded RNG for reproducibility.
"""

from __future__ import annotations

import numpy as np


def compute_pca(
    embeddings: np.ndarray,
    n_components: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA on embeddings using randomized SVD.

    Args:
        embeddings: (V, D) array of token embeddings
        n_components: Number of PCA components to compute
        seed: Random seed for deterministic SVD

    Returns:
        Tuple of (components, explained_variance, mean)
        - components: (n_components, D) principal axes
        - explained_variance: (n_components,) variance explained
        - mean: (D,) mean of input
    """
    # Center the embeddings
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # Randomized SVD
    rng = np.random.default_rng(seed)
    n_samples = centered.shape[0]
    n_features = centered.shape[1]

    # Oversampling for better approximation
    n_oversamples = min(10, n_features - n_components)
    n_random = n_components + n_oversamples

    # Generate random projection matrix
    omega = rng.standard_normal((n_features, n_random)).astype(centered.dtype)

    # Project and compute QR
    y_proj = centered @ omega
    q_matrix, _ = np.linalg.qr(y_proj)

    # Project data onto Q
    b_matrix = q_matrix.T @ centered

    # SVD of smaller matrix
    u, s_values, vt = np.linalg.svd(b_matrix, full_matrices=False)

    # Get components in original space
    components = vt[:n_components]

    # Fix sign convention: largest |loading| should be positive
    for i in range(n_components):
        col = components[i]
        max_idx = np.argmax(np.abs(col))
        if col[max_idx] < 0:
            components[i] = -components[i]

    # Explained variance
    explained_variance = (s_values[:n_components] ** 2) / (n_samples - 1)

    return components, explained_variance, mean


def project_with_pca(
    embeddings: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Project embeddings using pre-computed PCA.

    Args:
        embeddings: (V, D) array of token embeddings
        components: (n_components, D) principal axes
        mean: (D,) mean used for centering

    Returns:
        (V, n_components) projected coordinates
    """
    centered = embeddings - mean
    result: np.ndarray = centered @ components.T
    return result


def embedding_to_density(
    coords: np.ndarray,
    grid_size: int = 256,
    subsample: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Convert 2D coordinates to density field via histogram.

    Args:
        coords: (N, 2) array of 2D coordinates (first 2 PCA components)
        grid_size: Size of the output grid
        subsample: If set, subsample to this many points (deterministic)
        seed: Random seed for subsampling

    Returns:
        (grid_size, grid_size) density field
    """
    x = coords[:, 0]
    y = coords[:, 1]

    # Subsample if needed
    if subsample is not None and len(x) > subsample:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(x), size=subsample, replace=False)
        x = x[indices]
        y = y[indices]

    # Quantile normalization to [0, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else np.full_like(x, 0.5)

    y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else np.full_like(y, 0.5)

    # Convert to bin indices
    x_bins_int = np.clip((x_norm * (grid_size - 1)).astype(np.int64), 0, grid_size - 1)
    y_bins_int = np.clip((y_norm * (grid_size - 1)).astype(np.int64), 0, grid_size - 1)
    x_bins = x_bins_int.astype(np.intp)
    y_bins = y_bins_int.astype(np.intp)

    # Compute histogram
    density = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(len(x)):
        density[y_bins[i], x_bins[i]] += 1

    return density
