"""
gmm.py — Gaussian Mixture Model clustering and label propagation utilities.

This module provides an alternative to K-Medoids for the document compression
stage.  Given pre-computed embeddings it fits a GMM and selects *representative*
documents (closest to each component mean) for LLM classification.  Labels are
then propagated back to every document using the posterior probability
assignments.

Key advantages over K-Medoids
-----------------------------
* **Soft assignments** — each document has a probability vector over clusters,
  allowing a confidence-weighted propagation strategy.
* **Automatic model selection** — when a range of k is given, the module fits
  several models and picks the best k by BIC (or silhouette).
* **Covariance flexibility** — supports full / tied / diag / spherical.

Functions
---------
run_gmm(embeddings, k, ...)
    Fit a single GMM and return (labels, probabilities, means).

auto_select_k(embeddings, k_range, ...)
    Fit GMMs over a range of k values and return the best k by BIC.

get_representative_documents(documents, embeddings, means, labels, k)
    For each cluster, pick the document whose embedding is closest to the
    component mean (the "representative").

propagate_labels(representative_labels, posterior_probs, n_documents)
    Assign every document the label of its highest-probability cluster.

propagate_labels_soft(representative_labels, posterior_probs, threshold)
    Like propagate_labels but marks low-confidence documents as Unsuccessful.

build_cluster_map(labels)
    Return {cluster_id: [member_indices]} mapping.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core GMM fitting
# ---------------------------------------------------------------------------

def run_gmm(
    embeddings: np.ndarray,
    k: int,
    covariance_type: str = "tied",
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a Gaussian Mixture Model.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_samples, dim)``.  Will be L2-normalised internally so that
        Euclidean distances in the normalised space approximate cosine distance.
    k : int
        Number of Gaussian components.
    covariance_type : str
        ``"full"`` | ``"tied"`` | ``"diag"`` | ``"spherical"``.
    random_state : int
        Reproducibility seed.
    max_iter : int
        EM iteration limit.
    n_init : int
        Number of EM initialisations (best is kept).

    Returns
    -------
    labels : np.ndarray          shape ``(n,)``  hard cluster assignment
    probs  : np.ndarray          shape ``(n, k)`` posterior probabilities
    means  : np.ndarray          shape ``(k, dim)`` component means
    """
    n_samples = embeddings.shape[0]
    if k >= n_samples:
        logger.warning("k=%d >= n_samples=%d — clamping to %d", k, n_samples, n_samples)
        k = n_samples

    # L2-normalise so Euclidean ≈ cosine
    emb_norm = normalize(embeddings, norm="l2")

    logger.info(
        "Fitting GMM: k=%d, cov=%s, n_init=%d, n_samples=%d, dim=%d",
        k, covariance_type, n_init, n_samples, emb_norm.shape[1],
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
    probs = gmm.predict_proba(emb_norm)
    means = gmm.means_

    logger.info("GMM converged — BIC=%.2f, %d components", gmm.bic(emb_norm), k)
    return labels, probs, means


# ---------------------------------------------------------------------------
# Automatic k selection
# ---------------------------------------------------------------------------

def auto_select_k(
    embeddings: np.ndarray,
    k_range: tuple[int, int],
    criterion: Literal["bic", "silhouette"] = "bic",
    covariance_type: str = "tied",
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 3,
) -> tuple[int, dict[int, float]]:
    """Try multiple k values and return the best one.

    Parameters
    ----------
    embeddings : np.ndarray
    k_range : tuple[int, int]
        ``(k_min, k_max)`` inclusive.
    criterion : str
        ``"bic"`` (lower is better) or ``"silhouette"`` (higher is better).

    Returns
    -------
    best_k : int
    scores : dict[int, float]
        ``{k: score}`` for every k tried.
    """
    emb_norm = normalize(embeddings, norm="l2")
    k_min, k_max = k_range
    scores: dict[int, float] = {}

    logger.info("Auto-selecting k in [%d, %d] by %s", k_min, k_max, criterion)

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        gmm.fit(emb_norm)

        if criterion == "bic":
            score = gmm.bic(emb_norm)
        else:
            labels = gmm.predict(emb_norm)
            if len(set(labels)) < 2:
                score = -1.0
            else:
                score = silhouette_score(
                    emb_norm, labels,
                    metric="euclidean",
                    sample_size=min(5000, len(labels)),
                )

        scores[k] = score
        logger.info("  k=%d  %s=%.2f", k, criterion, score)

    if criterion == "bic":
        best_k = min(scores, key=scores.get)  # type: ignore[arg-type]
    else:
        best_k = max(scores, key=scores.get)  # type: ignore[arg-type]

    logger.info("Best k=%d (%s=%.2f)", best_k, criterion, scores[best_k])
    return best_k, scores


# ---------------------------------------------------------------------------
# Representative document extraction
# ---------------------------------------------------------------------------

def get_representative_documents(
    documents: list[dict],
    embeddings: np.ndarray,
    means: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[list[dict], np.ndarray]:
    """For each cluster, pick the document closest to the component mean.

    Returns
    -------
    representatives : list[dict]
        Subset of *documents* (one per cluster that has members), sorted by index.
    representative_indices : np.ndarray
        Original indices into *documents*.
    """
    emb_norm = normalize(embeddings, norm="l2")
    rep_indices: list[int] = []

    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        # Distance from each member to the component mean
        dists = np.linalg.norm(emb_norm[members] - means[c], axis=1)
        best_local = np.argmin(dists)
        rep_indices.append(int(members[best_local]))

    rep_indices_sorted = sorted(rep_indices)
    representatives = [documents[i] for i in rep_indices_sorted]
    logger.info(
        "Extracted %d representative documents from %d total (%d clusters)",
        len(representatives), len(documents), k,
    )
    return representatives, np.array(rep_indices_sorted)


# ---------------------------------------------------------------------------
# Label propagation
# ---------------------------------------------------------------------------

def build_cluster_map(labels: np.ndarray) -> dict[int, list[int]]:
    """Return ``{cluster_id: [doc_idx, ...]}``.  Works for any integer labels."""
    cluster_map: dict[int, list[int]] = {}
    for idx, cid in enumerate(labels):
        cluster_map.setdefault(int(cid), []).append(idx)
    return cluster_map


def propagate_labels(
    representative_labels: dict[int, str],
    labels: np.ndarray,
    n_documents: int,
) -> list[str]:
    """Hard propagation — every document gets its cluster's representative label.

    Parameters
    ----------
    representative_labels : dict[int, str]
        ``{representative_doc_index: predicted_label}``.
    labels : np.ndarray
        Hard cluster assignments for every document.
    n_documents : int
        Total documents.
    """
    # Build cluster_id → label  via representative membership
    cluster_to_label: dict[int, str] = {}
    for rep_idx, lbl in representative_labels.items():
        cid = int(labels[rep_idx])
        cluster_to_label[cid] = lbl

    no_label_count = 0
    result: list[str] = []
    for i in range(n_documents):
        cid = int(labels[i])
        label = cluster_to_label.get(cid, "Unsuccessful")
        if label == "Unsuccessful":
            no_label_count += 1
        result.append(label)

    if no_label_count:
        logger.warning(
            "%d/%d documents received no label (cluster had no representative label)",
            no_label_count, n_documents,
        )
    return result


def propagate_labels_soft(
    representative_labels: dict[int, str],
    labels: np.ndarray,
    probs: np.ndarray,
    n_documents: int,
    confidence_threshold: float = 0.4,
) -> list[str]:
    """Soft propagation — uses posterior probabilities.

    Documents whose max posterior probability is below *confidence_threshold*
    get the label ``"Unsuccessful"`` (low confidence).
    """
    cluster_to_label: dict[int, str] = {}
    for rep_idx, lbl in representative_labels.items():
        cid = int(labels[rep_idx])
        cluster_to_label[cid] = lbl

    result: list[str] = []
    low_conf = 0
    for i in range(n_documents):
        max_prob = float(probs[i].max())
        cid = int(labels[i])
        label = cluster_to_label.get(cid, "Unsuccessful")
        if max_prob < confidence_threshold:
            label = "Unsuccessful"
            low_conf += 1
        result.append(label)

    if low_conf:
        logger.info(
            "%d/%d documents below confidence threshold %.2f → Unsuccessful",
            low_conf, n_documents, confidence_threshold,
        )
    return result
