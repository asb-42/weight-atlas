"""Embedding projection: PCA (core) and UMAP (optional) for token embeddings."""

from weight_atlas.embedding.pca import compute_pca, project_with_pca

__all__ = ["compute_pca", "project_with_pca"]
