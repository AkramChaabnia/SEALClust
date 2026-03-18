"""
kmedoids.py — K-Medoids clustering and label propagation utilities.

This module implements the document compression stage: given pre-computed
embeddings it selects representative *medoid* documents (real data points,
not synthetic centroids) and provides helpers to propagate labels from
medoids back to every cluster member.

Functions
---------
run_kmedoids(embeddings, k)
    Fit K-Medoids and return (cluster_labels, medoid_indices).

get_medoid_documents(documents, medoid_indices)
    Extract the subset of documents at the medoid positions.

propagate_labels(medoid_labels, cluster_assignments, n_documents)
    Map medoid-level labels to every document via cluster membership.

build_cluster_map(cluster_assignments, medoid_indices)
    Return {cluster_id: [member_indices]} mapping.

Usage
-----
    from text_clustering.kmedoids import run_kmedoids, get_medoid_documents

    cluster_labels, medoid_idx = run_kmedoids(embeddings, k=100)
    medoid_docs = get_medoid_documents(all_docs, medoid_idx)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def run_kmedoids(
    embeddings: np.ndarray,
    k: int,
    random_state: int = 42,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit K-Medoids on the embedding matrix.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_samples, dim)``.
    k : int
        Number of clusters (medoids to select).
    random_state : int
        Reproducibility seed.
    max_iter : int
        Maximum optimisation iterations.

    Returns
    -------
    cluster_labels : np.ndarray
        Shape ``(n_samples,)`` — cluster id for each document.
    medoid_indices : np.ndarray
        Shape ``(k,)`` — indices into the original document list that are medoids.
    """
    from sklearn_extra.cluster import KMedoids

    # Clamp k to the number of samples (avoids crash on tiny datasets)
    n_samples = embeddings.shape[0]
    if k >= n_samples:
        logger.warning(
            "Requested k=%d >= n_samples=%d — clamping k to %d",
            k, n_samples, n_samples,
        )
        k = n_samples

    logger.info("Running K-Medoids: k=%d, n_samples=%d, dim=%d", k, n_samples, embeddings.shape[1])
    km = KMedoids(
        n_clusters=k,
        metric="cosine",
        method="alternate",
        init="k-medoids++",
        max_iter=max_iter,
        random_state=random_state,
    )
    km.fit(embeddings)

    cluster_labels: np.ndarray = km.labels_
    medoid_indices: np.ndarray = km.medoid_indices_

    logger.info("K-Medoids converged — %d medoids selected", len(medoid_indices))
    return cluster_labels, medoid_indices


def get_medoid_documents(documents: list[dict], medoid_indices: np.ndarray) -> list[dict]:
    """Return the documents at the given medoid positions.

    Parameters
    ----------
    documents : list[dict]
        Full dataset (each element has at least ``"input"`` and ``"label"`` keys).
    medoid_indices : np.ndarray
        Integer indices into *documents*.

    Returns
    -------
    list[dict]
        Subset of *documents* corresponding to the medoids, in index order.
    """
    medoids = [documents[int(i)] for i in sorted(medoid_indices)]
    logger.info("Extracted %d medoid documents from %d total", len(medoids), len(documents))
    return medoids


def build_cluster_map(
    cluster_assignments: np.ndarray,
    medoid_indices: np.ndarray,
) -> dict[int, list[int]]:
    """Build a mapping from cluster id to the list of member document indices.

    Parameters
    ----------
    cluster_assignments : np.ndarray
        Shape ``(n_samples,)`` — cluster id for every document.
    medoid_indices : np.ndarray
        Shape ``(k,)`` — indices of medoid documents.

    Returns
    -------
    dict[int, list[int]]
        ``{cluster_id: [doc_idx_0, doc_idx_1, …]}``.
    """
    cluster_map: dict[int, list[int]] = {}
    for doc_idx, cluster_id in enumerate(cluster_assignments):
        cluster_id = int(cluster_id)
        cluster_map.setdefault(cluster_id, []).append(doc_idx)
    return cluster_map


def propagate_labels(
    medoid_labels: dict[int, str],
    cluster_assignments: np.ndarray,
    n_documents: int,
) -> list[str]:
    """Propagate labels from medoids to every cluster member.

    Parameters
    ----------
    medoid_labels : dict[int, str]
        ``{medoid_document_index: predicted_label}``.
        Keys are the original (global) document indices of the medoids.
    cluster_assignments : np.ndarray
        Shape ``(n_documents,)`` — cluster id for every document.
    n_documents : int
        Total number of documents (sanity check).

    Returns
    -------
    list[str]
        A label for each document, in the original dataset order.
        Documents whose cluster has no medoid label receive ``"Unsuccessful"``.
    """
    # Build reverse map: medoid_index → cluster_id
    # (each medoid belongs to exactly one cluster; its cluster_assignment == its cluster)
    medoid_to_cluster: dict[int, int] = {}
    for med_idx in medoid_labels:
        medoid_to_cluster[med_idx] = int(cluster_assignments[med_idx])

    # cluster_id → label
    cluster_to_label: dict[int, str] = {}
    for med_idx, label in medoid_labels.items():
        cid = int(cluster_assignments[med_idx])
        cluster_to_label[cid] = label

    # Propagate
    all_labels: list[str] = []
    unlabelled = 0
    for doc_idx in range(n_documents):
        cid = int(cluster_assignments[doc_idx])
        label = cluster_to_label.get(cid, "Unsuccessful")
        if label == "Unsuccessful":
            unlabelled += 1
        all_labels.append(label)

    if unlabelled:
        logger.warning(
            "%d/%d documents received no label (cluster had no medoid label)",
            unlabelled, n_documents,
        )
    else:
        logger.info("All %d documents labelled via propagation", n_documents)

    return all_labels
