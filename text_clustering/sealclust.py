"""
sealclust.py — SEAL-Clust core: auto-k via Elbow method + clustering helpers.

This module implements the *automatic determination of k* step in the
SEAL-Clust framework.  It uses the **Elbow method** on K-Medoids inertia
(sum of distances to medoids) to find the best k, with a manual fallback.

The Elbow method
----------------
For each candidate k we fit K-Medoids and record the inertia (within-cluster
sum of distances).  The "elbow" — the point of maximum curvature in the
inertia-vs-k curve — is detected using the *Kneedle* algorithm (second
derivative / geometric approach).

If the automatic detection fails or the user wants to override, a manual k
can be supplied via ``--sealclust_k``.

Functions
---------
elbow_select_k(embeddings, k_range, ...)
    Try each k, fit K-Medoids, record inertia, find the elbow.

find_elbow(k_values, inertias)
    Geometric Kneedle algorithm to detect the elbow point.

run_sealclust_clustering(embeddings, k, ...)
    Convenience wrapper: run K-Medoids with the chosen k.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Elbow detection (Kneedle algorithm)
# ---------------------------------------------------------------------------

def find_elbow(k_values: list[int], inertias: list[float]) -> int:
    """Detect the elbow in an inertia-vs-k curve.

    Uses the geometric "maximum distance to the line" approach:
    draw a straight line from the first point (k_min, inertia_max) to the
    last point (k_max, inertia_min) and pick the k whose inertia is
    farthest above that line.

    Parameters
    ----------
    k_values : list[int]
        Candidate k values (sorted ascending).
    inertias : list[float]
        Corresponding inertia for each k.

    Returns
    -------
    int
        The k at the elbow.
    """
    if len(k_values) < 3:
        logger.warning("Need ≥ 3 candidate k values for elbow detection; returning first k")
        return k_values[0]

    # Normalise to [0, 1] for numerical stability
    k_arr = np.array(k_values, dtype=float)
    inert_arr = np.array(inertias, dtype=float)

    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min() + 1e-12)
    inert_norm = (inert_arr - inert_arr.min()) / (inert_arr.max() - inert_arr.min() + 1e-12)

    # Line from first to last point
    p1 = np.array([k_norm[0], inert_norm[0]])
    p2 = np.array([k_norm[-1], inert_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        logger.warning("Flat inertia curve — returning middle k")
        return k_values[len(k_values) // 2]

    # Distance of each point from the line
    distances = []
    for i in range(len(k_values)):
        point = np.array([k_norm[i], inert_norm[i]])
        # Signed distance from point to line (we want the one furthest above)
        d = abs(np.cross(line_vec, p1 - point)) / line_len
        distances.append(d)

    best_idx = int(np.argmax(distances))
    elbow_k = k_values[best_idx]

    logger.info(
        "Elbow detected at k=%d (distance=%.4f from baseline)",
        elbow_k, distances[best_idx],
    )
    return elbow_k


# ---------------------------------------------------------------------------
# Elbow-based k selection with K-Medoids
# ---------------------------------------------------------------------------

def elbow_select_k(
    embeddings: np.ndarray,
    k_range: tuple[int, int],
    step: int = 5,
    random_state: int = 42,
    max_iter: int = 300,
) -> tuple[int, dict[int, float]]:
    """Try candidate k values with K-Medoids and find the elbow.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_samples, dim)`` — already dimensionality-reduced.
    k_range : tuple[int, int]
        ``(k_min, k_max)`` inclusive.
    step : int
        Step between candidate k values.  E.g. step=5 → try 10, 15, 20, …
    random_state : int
        Seed for K-Medoids.
    max_iter : int
        Max iterations per K-Medoids fit.

    Returns
    -------
    best_k : int
        The k at the elbow point.
    inertias : dict[int, float]
        ``{k: inertia}`` for every k tried.
    """
    from sklearn_extra.cluster import KMedoids

    k_min, k_max = k_range
    n_samples = embeddings.shape[0]

    # Build candidate list
    candidates = list(range(k_min, k_max + 1, step))
    # Always include k_max if step doesn't land on it
    if candidates[-1] != k_max:
        candidates.append(k_max)

    # Clamp to n_samples
    candidates = [k for k in candidates if k < n_samples]
    if not candidates:
        logger.warning("All candidates >= n_samples=%d — using k=%d", n_samples, n_samples // 2)
        return n_samples // 2, {}

    logger.info(
        "Elbow search: trying %d candidates in [%d, %d] step=%d",
        len(candidates), candidates[0], candidates[-1], step,
    )

    inertias: dict[int, float] = {}
    for k in candidates:
        km = KMedoids(
            n_clusters=k,
            metric="cosine",
            method="alternate",
            init="k-medoids++",
            max_iter=max_iter,
            random_state=random_state,
        )
        km.fit(embeddings)
        inertias[k] = float(km.inertia_)
        logger.info("  k=%d  inertia=%.4f", k, km.inertia_)

    # Find elbow
    k_vals = sorted(inertias.keys())
    inert_vals = [inertias[k] for k in k_vals]
    best_k = find_elbow(k_vals, inert_vals)

    logger.info("Elbow method selected k=%d", best_k)
    return best_k, inertias


# ---------------------------------------------------------------------------
# K-Medoids clustering wrapper (for SEAL-Clust pipeline)
# ---------------------------------------------------------------------------

def run_sealclust_clustering(
    embeddings: np.ndarray,
    k: int,
    random_state: int = 42,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Run K-Medoids on (optionally reduced) embeddings.

    This is a thin wrapper around ``kmedoids.run_kmedoids`` that exists so
    the SEAL-Clust pipeline has its own entry point and metadata namespace.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_samples, dim)`` — raw or dimensionality-reduced.
    k : int
        Number of clusters / medoids.
    random_state : int
    max_iter : int

    Returns
    -------
    cluster_labels : np.ndarray  shape ``(n_samples,)``
    medoid_indices : np.ndarray  shape ``(k,)``
    """
    from text_clustering.kmedoids import run_kmedoids

    return run_kmedoids(embeddings, k=k, random_state=random_state, max_iter=max_iter)


def get_prototypes(
    documents: list[dict],
    medoid_indices: np.ndarray,
) -> list[dict]:
    """Extract prototype documents at medoid positions.

    Delegates to ``kmedoids.get_medoid_documents`` but provides a
    SEAL-Clust-specific name.
    """
    from text_clustering.kmedoids import get_medoid_documents

    return get_medoid_documents(documents, medoid_indices)


def propagate_labels(
    medoid_labels: dict[int, str],
    cluster_assignments: np.ndarray,
    n_documents: int,
) -> list[str]:
    """Propagate prototype labels to every document.

    Delegates to ``kmedoids.propagate_labels``.
    """
    from text_clustering.kmedoids import propagate_labels as _kmedoids_propagate

    return _kmedoids_propagate(medoid_labels, cluster_assignments, n_documents)
