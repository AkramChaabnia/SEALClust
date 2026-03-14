"""
sealclust.py — SEAL-Clust core algorithms.

This module implements the core computational steps of the SEAL-Clust
(Scalable Efficient Autonomous LLM Clustering) framework:

  1. **Overclustering** — K-Medoids with a large K₀ to create micro-clusters
  2. **Elbow-based k selection** — legacy method (Kneedle algorithm)
  3. **BIC-based K* estimation** — run GMM on representative embeddings
     and select the K that minimises the Bayesian Information Criterion
  4. **Label discovery** — send representative texts to the LLM in batches
  5. **Label consolidation** — merge candidate labels to exactly K* via LLM
  6. **Prototype extraction** and **label propagation**

Functions
---------
run_sealclust_clustering(embeddings, k, ...)
    Run K-Medoids overclustering.

estimate_k_star_bic(representative_embeddings, k_min, k_max, ...)
    GMM + BIC on representative embeddings → optimal K*.

discover_labels(representative_texts, client, chunk_size)
    LLM label discovery from representative documents only.

consolidate_labels(candidate_labels, k_star, client)
    LLM label consolidation to exactly K* labels.

elbow_select_k(embeddings, k_range, ...)
    Legacy: Elbow method auto-k selection.

find_elbow(k_values, inertias)
    Legacy: geometric Kneedle algorithm.

get_prototypes(documents, medoid_indices)
    Extract prototype documents at medoid positions.

propagate_labels(medoid_labels, cluster_assignments, n_documents)
    Map prototype labels to all documents via cluster membership.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BIC-based K* estimation (Stage 6)
# ---------------------------------------------------------------------------

def estimate_k_star_bic(
    representative_embeddings: np.ndarray,
    k_min: int = 5,
    k_max: int = 50,
    covariance_type: str = "tied",
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 3,
) -> tuple[int, dict[int, float]]:
    """Estimate optimal K* using GMM + BIC on representative embeddings.

    This implements Stage 6 of SEALClust: run GMM for each candidate K on
    the K₀ representative embeddings and select the K that minimises BIC.

    Parameters
    ----------
    representative_embeddings : np.ndarray
        Shape ``(K₀, dim)`` — embeddings of representative documents only.
    k_min : int
        Minimum candidate K.
    k_max : int
        Maximum candidate K.  Will be clamped to K₀ - 1.
    covariance_type : str
        GMM covariance type: ``"full"`` | ``"tied"`` | ``"diag"`` | ``"spherical"``.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    k_star : int
        The K that minimises BIC.
    bic_scores : dict[int, float]
        ``{k: bic_score}`` for every K tried.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import normalize

    n_reps = representative_embeddings.shape[0]
    # Clamp k_max to n_reps - 1  (GMM needs n_components < n_samples)
    k_max = min(k_max, n_reps - 1)
    if k_min > k_max:
        k_min = max(2, k_max // 2)

    # L2-normalise so Euclidean ≈ cosine
    emb_norm = normalize(representative_embeddings, norm="l2")

    logger.info(
        "Stage 6: Estimating K* via GMM+BIC on %d representative embeddings (dim=%d)",
        n_reps, emb_norm.shape[1],
    )
    logger.info("  Candidate range: K ∈ [%d, %d], covariance_type=%s", k_min, k_max, covariance_type)

    bic_scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                max_iter=max_iter,
                n_init=n_init,
                random_state=random_state,
            )
            gmm.fit(emb_norm)
            bic = gmm.bic(emb_norm)
            bic_scores[k] = float(bic)
            logger.info("  K=%d  BIC=%.2f", k, bic)
        except Exception as e:
            logger.warning("  K=%d  GMM failed: %s", k, e)
            continue

    if not bic_scores:
        logger.warning("All GMM fits failed — defaulting K*=%d", k_min)
        return k_min, {}

    k_star = min(bic_scores, key=bic_scores.get)  # type: ignore[arg-type]
    logger.info("Stage 6: K* = %d (BIC=%.2f)", k_star, bic_scores[k_star])
    return k_star, bic_scores


# ---------------------------------------------------------------------------
# LLM Label Discovery (Stage 5) — on representatives only
# ---------------------------------------------------------------------------

def discover_labels(
    representative_texts: list[str],
    client,
    chunk_size: int = 30,
) -> list[str]:
    """Send representative documents to the LLM in batches to discover labels.

    Unlike the original label_generation which sends ALL documents, this sends
    only the K₀ representative texts — dramatically reducing LLM calls.

    Parameters
    ----------
    representative_texts : list[str]
        Texts of the K₀ representative documents.
    client : OpenAI client
        Initialised LLM client.
    chunk_size : int
        Number of representative texts per LLM call.

    Returns
    -------
    list[str]
        All unique candidate labels discovered.
    """
    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_discover_labels

    all_labels: list[str] = []
    n_chunks = (len(representative_texts) + chunk_size - 1) // chunk_size

    logger.info(
        "Stage 5: Discovering labels from %d representatives in %d chunks (chunk_size=%d)",
        len(representative_texts), n_chunks, chunk_size,
    )

    for i in range(0, len(representative_texts), chunk_size):
        chunk = representative_texts[i : i + chunk_size]
        prompt = prompt_discover_labels(chunk)
        raw = chat(prompt, client, max_tokens=4096)
        if raw is None:
            logger.warning("  Chunk %d: LLM returned None — skipping", i // chunk_size + 1)
            continue

        try:
            parsed = eval(raw)  # noqa: S307
        except Exception:
            logger.warning("  Chunk %d: could not parse LLM response — skipping", i // chunk_size + 1)
            continue

        # Handle both {"labels": [...]} and flat list [...]
        if isinstance(parsed, dict):
            for val in parsed.values():
                if isinstance(val, list):
                    for label in val:
                        if isinstance(label, str) and label not in all_labels:
                            all_labels.append(label)
        elif isinstance(parsed, list):
            for label in parsed:
                if isinstance(label, str) and label not in all_labels:
                    all_labels.append(label)

        logger.info("  Chunk %d/%d — labels so far: %d", i // chunk_size + 1, n_chunks, len(all_labels))

    logger.info("Stage 5: Discovered %d unique candidate labels", len(all_labels))
    return all_labels


# ---------------------------------------------------------------------------
# LLM Label Consolidation (Stage 7) — merge to exactly K*
# ---------------------------------------------------------------------------

def consolidate_labels(
    candidate_labels: list[str],
    k_star: int,
    client,
) -> list[str]:
    """Merge candidate labels into exactly K* final labels using the LLM.

    Parameters
    ----------
    candidate_labels : list[str]
        All candidate labels from Stage 5 (may be hundreds).
    k_star : int
        The statistically optimal number of clusters from Stage 6.
    client : OpenAI client

    Returns
    -------
    list[str]
        Exactly K* merged labels (or best effort).
    """
    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_consolidate_labels

    logger.info(
        "Stage 7: Consolidating %d candidate labels into exactly K*=%d labels",
        len(candidate_labels), k_star,
    )

    prompt = prompt_consolidate_labels(candidate_labels, k_star)
    raw = chat(prompt, client, max_tokens=4096)

    if raw is None:
        logger.error("Stage 7: LLM returned None — returning unmerged labels")
        return candidate_labels

    try:
        parsed = eval(raw)  # noqa: S307
    except Exception:
        logger.error("Stage 7: could not parse LLM response — returning unmerged labels")
        return candidate_labels

    # Extract labels from response
    final_labels: list[str] = []
    if isinstance(parsed, dict):
        for val in parsed.values():
            if isinstance(val, list):
                final_labels.extend(val)
    elif isinstance(parsed, list):
        final_labels = [x for x in parsed if isinstance(x, str)]

    if not final_labels:
        logger.error("Stage 7: parsed empty label list — returning unmerged labels")
        return candidate_labels

    logger.info(
        "Stage 7: Consolidated %d → %d labels (target was %d)",
        len(candidate_labels), len(final_labels), k_star,
    )

    # If the LLM didn't produce exactly K*, try a second pass
    if len(final_labels) != k_star and abs(len(final_labels) - k_star) > 2:
        logger.info("Stage 7: Second consolidation pass (got %d, want %d)", len(final_labels), k_star)
        prompt2 = prompt_consolidate_labels(final_labels, k_star)
        raw2 = chat(prompt2, client, max_tokens=4096)
        if raw2:
            try:
                parsed2 = eval(raw2)  # noqa: S307
                labels2: list[str] = []
                if isinstance(parsed2, dict):
                    for val in parsed2.values():
                        if isinstance(val, list):
                            labels2.extend(val)
                elif isinstance(parsed2, list):
                    labels2 = [x for x in parsed2 if isinstance(x, str)]
                if labels2:
                    logger.info("Stage 7: Second pass produced %d labels", len(labels2))
                    final_labels = labels2
            except Exception:
                pass

    return final_labels


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
