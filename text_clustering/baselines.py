"""
baselines.py — Non-LLM baseline clustering pipelines.

Pure embedding-based clustering for benchmarking against the hybrid and
SEAL-Clust pipelines.  No LLM calls are made — clustering relies entirely
on geometric structure in the embedding space.

Pipelines
---------
1. **KMeans Baseline** — Embedding → L2-normalise → KMeans → evaluate
2. **GMM Baseline** — Embedding → L2-normalise → GMM → evaluate

Both pipelines:
  - Compute (or load cached) embeddings
  - Optionally apply PCA dimensionality reduction
  - Cluster directly on the embedding space
  - Assign documents to clusters using hard assignments
  - Evaluate using ACC / NMI / ARI (no label names — just cluster IDs)

Functions
---------
run_kmeans_baseline(embeddings, k, random_state)
    Fit KMeans and return (labels, inertia, silhouette).

run_gmm_baseline(embeddings, k, covariance_type, random_state)
    Fit GMM and return (labels, bic, silhouette).

auto_select_k_kmeans(embeddings, k_min, k_max, random_state)
    Try multiple k values with KMeans and pick the best by silhouette.

auto_select_k_gmm(embeddings, k_min, k_max, covariance_type, random_state)
    Try multiple k values with GMM and pick the best by BIC.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KMeans Baseline
# ---------------------------------------------------------------------------

def run_kmeans_baseline(
    embeddings: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> tuple[np.ndarray, float, float]:
    """Fit KMeans on L2-normalised embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_samples, dim)``.
    k : int
        Number of clusters.
    random_state : int
    n_init : int
        Number of KMeans initialisations.
    max_iter : int
        Maximum iterations per KMeans run.

    Returns
    -------
    labels : np.ndarray   shape ``(n_samples,)``
    inertia : float       Sum of squared distances to cluster centres.
    sil_score : float     Silhouette coefficient (−1 to 1).
    """
    n_samples = embeddings.shape[0]
    k = min(k, n_samples - 1)

    emb_norm = normalize(embeddings, norm="l2")

    logger.info(
        "KMeans baseline: k=%d, n_samples=%d, dim=%d",
        k, n_samples, emb_norm.shape[1],
    )

    km = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = km.fit_predict(emb_norm)
    inertia = float(km.inertia_)

    n_unique = len(set(labels))
    if n_unique < 2:
        sil = -1.0
    else:
        sil = float(silhouette_score(
            emb_norm, labels,
            metric="euclidean",
            sample_size=min(5000, n_samples),
        ))

    logger.info(
        "KMeans baseline: inertia=%.2f, silhouette=%.4f", inertia, sil,
    )
    return labels, inertia, sil


def auto_select_k_kmeans(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 50,
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    """Try multiple k values with KMeans and select the best by silhouette.

    Parameters
    ----------
    embeddings : np.ndarray
    k_min, k_max : int
        Inclusive range for k.
    random_state : int

    Returns
    -------
    best_k : int
    scores : dict[int, float]
        ``{k: silhouette_score}``.
    """
    emb_norm = normalize(embeddings, norm="l2")
    n_samples = emb_norm.shape[0]
    k_max = min(k_max, n_samples - 1)

    logger.info(
        "KMeans auto-select k ∈ [%d, %d], n_samples=%d",
        k_min, k_max, n_samples,
    )

    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=k, random_state=random_state, n_init=10, max_iter=300,
        )
        labels = km.fit_predict(emb_norm)
        n_unique = len(set(labels))
        if n_unique < 2:
            continue
        sil = float(silhouette_score(
            emb_norm, labels, metric="euclidean",
            sample_size=min(5000, n_samples),
        ))
        scores[k] = sil
        logger.info("  k=%d  silhouette=%.4f", k, sil)

    if not scores:
        logger.warning("No valid k — defaulting to k=%d", k_min)
        return k_min, {}

    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    logger.info("Best k=%d (silhouette=%.4f)", best_k, scores[best_k])
    return best_k, scores


# ---------------------------------------------------------------------------
# GMM Baseline
# ---------------------------------------------------------------------------

def run_gmm_baseline(
    embeddings: np.ndarray,
    k: int,
    covariance_type: str = "tied",
    random_state: int = 42,
    n_init: int = 3,
    max_iter: int = 300,
) -> tuple[np.ndarray, float, float]:
    """Fit GMM on L2-normalised embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_samples, dim)``.
    k : int
        Number of Gaussian components.
    covariance_type : str
        ``"full"`` | ``"tied"`` | ``"diag"`` | ``"spherical"``.
    random_state : int
    n_init : int
    max_iter : int

    Returns
    -------
    labels : np.ndarray   shape ``(n_samples,)``
    bic : float           Bayesian Information Criterion.
    sil_score : float     Silhouette coefficient.
    """
    n_samples = embeddings.shape[0]
    k = min(k, n_samples - 1)

    emb_norm = normalize(embeddings, norm="l2")

    logger.info(
        "GMM baseline: k=%d, cov=%s, n_samples=%d, dim=%d",
        k, covariance_type, n_samples, emb_norm.shape[1],
    )

    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
    )
    gmm.fit(emb_norm)

    labels = gmm.predict(emb_norm)
    bic = float(gmm.bic(emb_norm))

    n_unique = len(set(labels))
    if n_unique < 2:
        sil = -1.0
    else:
        sil = float(silhouette_score(
            emb_norm, labels,
            metric="euclidean",
            sample_size=min(5000, n_samples),
        ))

    logger.info("GMM baseline: BIC=%.2f, silhouette=%.4f", bic, sil)
    return labels, bic, sil


def auto_select_k_gmm(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 50,
    covariance_type: str = "tied",
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    """Try multiple k values with GMM and select the best by BIC (lower is better).

    Parameters
    ----------
    embeddings : np.ndarray
    k_min, k_max : int
    covariance_type : str
    random_state : int

    Returns
    -------
    best_k : int
    scores : dict[int, float]
        ``{k: bic_score}``.
    """
    emb_norm = normalize(embeddings, norm="l2")
    n_samples = emb_norm.shape[0]
    k_max = min(k_max, n_samples - 1)

    logger.info(
        "GMM auto-select k ∈ [%d, %d], cov=%s, n_samples=%d",
        k_min, k_max, covariance_type, n_samples,
    )

    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                max_iter=300,
                n_init=3,
                random_state=random_state,
            )
            gmm.fit(emb_norm)
            bic = float(gmm.bic(emb_norm))
            scores[k] = bic
            logger.info("  k=%d  BIC=%.2f", k, bic)
        except Exception as e:
            logger.warning("  k=%d  failed: %s", k, e)
            continue

    if not scores:
        logger.warning("No valid k — defaulting to k=%d", k_min)
        return k_min, {}

    best_k = min(scores, key=scores.get)  # type: ignore[arg-type]
    logger.info("Best k=%d (BIC=%.2f)", best_k, scores[best_k])
    return best_k, scores
