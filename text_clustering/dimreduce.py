"""
dimreduce.py — Dimensionality reduction for the SEAL-Clust pipeline.

Provides t-SNE (and optionally PCA) wrappers so downstream clustering
operates in a lower-dimensional, noise-reduced space.

Why t-SNE?
----------
t-SNE preserves local neighbourhood structure — documents that are
semantically close in the high-dimensional embedding space stay close
after projection.  This helps microcluster algorithms (K-Medoids, GMM)
form tighter, more coherent clusters.

Functions
---------
reduce_tsne(embeddings, n_components, perplexity, random_state, ...)
    Apply t-SNE and return the reduced embedding matrix.

reduce_pca(embeddings, n_components, random_state)
    Apply PCA and return the reduced embedding matrix.
    (Useful as a fast pre-step before t-SNE on very large datasets.)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def reduce_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float | str = "auto",
    n_iter: int = 1000,
    random_state: int = 42,
    init: str = "pca",
    metric: str = "cosine",
) -> np.ndarray:
    """Apply t-SNE dimensionality reduction.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_samples, dim)``.
    n_components : int
        Target dimensionality (typically 2 or 3 for visualisation,
        or higher, e.g. 50, for downstream clustering).
    perplexity : float
        Roughly the expected number of nearest neighbours.
        Must be < n_samples.  Rule of thumb: 5–50.
    learning_rate : float | str
        ``"auto"`` lets sklearn choose (recommended for most cases).
    n_iter : int
        Number of optimisation iterations.
    random_state : int
        Seed for reproducibility.
    init : str
        ``"pca"`` for deterministic initialisation (recommended) or ``"random"``.
    metric : str
        Distance metric in the original space.  ``"cosine"`` works well for
        sentence embeddings.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, n_components)`` — the reduced embeddings.
    """
    from sklearn.manifold import TSNE

    n_samples = embeddings.shape[0]

    # Clamp perplexity: must be < n_samples
    if perplexity >= n_samples:
        clamped = max(5.0, n_samples / 4.0)
        logger.warning(
            "perplexity=%.1f >= n_samples=%d — clamping to %.1f",
            perplexity, n_samples, clamped,
        )
        perplexity = clamped

    logger.info(
        "Running t-SNE: n_components=%d, perplexity=%.1f, metric=%s, "
        "max_iter=%d, n_samples=%d, original_dim=%d",
        n_components, perplexity, metric, n_iter, n_samples, embeddings.shape[1],
    )

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=random_state,
        init=init,
        metric=metric,
    )

    reduced = tsne.fit_transform(embeddings)
    logger.info(
        "t-SNE complete — output shape: %s, KL divergence: %.4f",
        reduced.shape, tsne.kl_divergence_,
    )
    return reduced


def reduce_pca(
    embeddings: np.ndarray,
    n_components: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Apply PCA dimensionality reduction.

    Useful as a fast pre-processing step before t-SNE when the original
    dimension is very high (e.g. 768 or 1024).

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_samples, dim)``.
    n_components : int
        Number of principal components to keep.
    random_state : int
        Seed for the randomised SVD solver.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, n_components)``.
    """
    from sklearn.decomposition import PCA

    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])

    logger.info(
        "Running PCA: n_components=%d, n_samples=%d, original_dim=%d",
        n_components, embeddings.shape[0], embeddings.shape[1],
    )

    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    explained = sum(pca.explained_variance_ratio_) * 100
    logger.info(
        "PCA complete — output shape: %s, explained variance: %.1f%%",
        reduced.shape, explained,
    )
    return reduced
