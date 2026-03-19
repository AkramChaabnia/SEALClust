"""
_kmedoids_impl.py — Pure-sklearn/scipy K-Medoids implementation.

Drop-in replacement for ``sklearn_extra.cluster.KMedoids`` that works
with NumPy ≥ 2.0.  Only the subset of the API actually used by this
project is implemented (``alternate`` method, cosine metric, k-medoids++
initialisation).

The ``scikit-learn-extra`` package (v0.3.0) was compiled against
NumPy 1.x and crashes on NumPy 2.x due to ABI incompatibility.
This module removes that dependency entirely.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class KMedoids:
    """K-Medoids clustering (PAM ``alternate`` algorithm).

    Parameters
    ----------
    n_clusters : int
        Number of clusters (K).
    metric : str
        Distance metric (passed to ``scipy.spatial.distance.cdist``).
    method : str
        Only ``"alternate"`` is supported (the default in sklearn-extra).
    init : str
        ``"k-medoids++"`` (D² weighted seeding) or ``"random"``.
    max_iter : int
        Maximum number of swap iterations.
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        metric: str = "cosine",
        method: str = "alternate",
        init: str = "k-medoids++",
        max_iter: int = 300,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state

        # Fitted attributes
        self.labels_: np.ndarray | None = None
        self.medoid_indices_: np.ndarray | None = None
        self.inertia_: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMedoids":
        """Fit K-Medoids on *X*."""
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        if self.n_clusters >= n_samples:
            # Degenerate case: every point is its own medoid
            self.medoid_indices_ = np.arange(n_samples)
            self.labels_ = np.arange(n_samples)
            self.inertia_ = 0.0
            return self

        # Pre-compute full distance matrix (fine for the dataset sizes
        # this project deals with — a few thousand samples at most).
        D = cdist(X, X, metric=self.metric)

        # Initialise medoids
        if self.init == "k-medoids++":
            medoids = self._kmedoids_pp_init(D, rng)
        else:
            medoids = rng.choice(n_samples, self.n_clusters, replace=False)

        # Alternate (BUILD + SWAP) iterations
        for _ in range(self.max_iter):
            # Assign each point to nearest medoid
            labels = np.argmin(D[:, medoids], axis=1)

            # Update medoids: for each cluster pick the member with the
            # smallest total distance to all other members.
            new_medoids = medoids.copy()
            for c in range(self.n_clusters):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                # Sum of distances from each member to every other member
                intra = D[np.ix_(members, members)].sum(axis=1)
                new_medoids[c] = members[np.argmin(intra)]

            if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
                break
            medoids = new_medoids

        # Final assignment
        labels = np.argmin(D[:, medoids], axis=1)
        inertia = sum(
            D[i, medoids[labels[i]]] for i in range(n_samples)
        )

        self.medoid_indices_ = medoids
        self.labels_ = labels
        self.inertia_ = float(inertia)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _kmedoids_pp_init(
        self, D: np.ndarray, rng: np.random.RandomState,
    ) -> np.ndarray:
        """K-Medoids++ initialisation (D²-weighted sampling)."""
        n_samples = D.shape[0]
        medoids = np.empty(self.n_clusters, dtype=int)

        # First medoid: uniform random
        medoids[0] = rng.randint(n_samples)

        for i in range(1, self.n_clusters):
            # Distance from each point to its nearest existing medoid
            min_dist = D[:, medoids[:i]].min(axis=1)
            # D²-weighted probability
            probs = min_dist ** 2
            total = probs.sum()
            if total == 0:
                # All remaining points are co-located with a medoid
                remaining = np.setdiff1d(np.arange(n_samples), medoids[:i])
                medoids[i] = rng.choice(remaining)
            else:
                probs /= total
                medoids[i] = rng.choice(n_samples, p=probs)

        return medoids
