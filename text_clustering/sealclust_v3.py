"""
sealclust_v3.py — SEAL-Clust v3 core algorithms.

Improvements over v2:
  - **Multiple clustering backends**: K-Medoids (default), GMM, or KMeans
  - **One-word label constraint**: labels are single general words
  - **Iterative consolidation**: handles large label sets via chunked merging
  - **Batched representative classification**: reduces LLM calls in Stage 8
  - **Flexible representative selection**: medoid (K-Medoids), closest-to-centroid
    (GMM/KMeans)

Stages
------
  1. Document Embedding
  2. Optional Dimensionality Reduction (PCA)
  3. Overclustering (K-Medoids / GMM / KMeans with large K₀)
  4. Representative Selection (one per cluster)
  5. Label Discovery (LLM — one-word general labels)
  6. K* Estimation (optional — manual or silhouette/BIC/ensemble)
  7. Label Consolidation (LLM — merge to exactly K*)
  8. Representative Classification (LLM — classify K₀ reps into K* labels)
  9. Label Propagation (assign each document its representative's label)

Functions
---------
run_overclustering(embeddings, k0, method, ...)
    Stage 3 — supports kmedoids, gmm, kmeans.

select_representatives(documents, embeddings, labels, method, ...)
    Stage 4 — extract one representative per cluster.

discover_labels_v3(representative_texts, client, chunk_size, ...)
    Stage 5 — one-word label discovery via LLM.

consolidate_labels_v3(candidate_labels, k_star, client)
    Stage 7 — iterative chunked merge to exactly K*.

classify_representatives_v3(representative_texts, labels, client, batch_size, ...)
    Stage 8 — batched LLM classification of representatives.

propagate_labels_v3(rep_labels, cluster_assignments, n_documents)
    Stage 9 — map representative labels to all documents.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re

import numpy as np

logger = logging.getLogger(__name__)


# ── Checkpoint helpers ────────────────────────────────────────────────────

def _save_checkpoint(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


def _remove_checkpoint(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Stage 3 — Overclustering (multi-method)
# ---------------------------------------------------------------------------

def run_overclustering(
    embeddings: np.ndarray,
    k0: int,
    method: str = "kmedoids",
    random_state: int = 42,
    max_iter: int = 300,
    covariance_type: str = "tied",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Stage 3: Overclustering with K₀ micro-clusters.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_samples, dim)`` — optionally reduced.
    k0 : int
        Number of overclusters.
    method : str
        ``"kmedoids"`` (default), ``"gmm"``, or ``"kmeans"``.
    random_state : int
    max_iter : int
    covariance_type : str
        GMM covariance type (only used when method="gmm").

    Returns
    -------
    cluster_labels : np.ndarray   shape ``(n_samples,)``
    extra : np.ndarray
        For kmedoids: medoid_indices shape ``(k0,)``
        For gmm: component_means shape ``(k0, dim)``
        For kmeans: centroids shape ``(k0, dim)``
    method_used : str
    """
    n_samples = embeddings.shape[0]
    k0 = min(k0, n_samples - 1)

    if method == "kmedoids":
        from text_clustering.kmedoids import run_kmedoids
        labels, medoid_indices = run_kmedoids(
            embeddings, k=k0, random_state=random_state, max_iter=max_iter,
        )
        logger.info("Stage 3: K-Medoids overclustering → %d clusters", k0)
        return labels, medoid_indices, "kmedoids"

    elif method == "gmm":
        from text_clustering.gmm import run_gmm
        labels, _probs, means = run_gmm(
            embeddings, k=k0, covariance_type=covariance_type,
            random_state=random_state, max_iter=max_iter,
        )
        logger.info("Stage 3: GMM overclustering → %d components", k0)
        return labels, means, "gmm"

    elif method == "kmeans":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        emb_norm = normalize(embeddings, norm="l2")
        km = KMeans(
            n_clusters=k0, random_state=random_state,
            max_iter=max_iter, n_init=3,
        )
        labels = km.fit_predict(emb_norm)
        logger.info("Stage 3: KMeans overclustering → %d clusters", k0)
        return labels, km.cluster_centers_, "kmeans"

    else:
        raise ValueError(f"Unknown clustering method: {method!r}")


# ---------------------------------------------------------------------------
# Stage 4 — Representative Selection
# ---------------------------------------------------------------------------

def select_representatives(
    documents: list[dict],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    clustering_method: str,
    extra: np.ndarray,
) -> tuple[list[dict], list[int]]:
    """Stage 4: Extract one representative document per cluster.

    - **K-Medoids**: representatives are the medoid documents (actual docs).
    - **GMM/KMeans**: representative = document closest to the component
      mean / centroid in embedding space.

    Parameters
    ----------
    documents : list[dict]
        Full dataset (each has at least ``"input"``).
    embeddings : np.ndarray
        Shape ``(n, dim)`` — same dim used for clustering.
    cluster_labels : np.ndarray
        Shape ``(n,)`` — cluster assignment per document.
    clustering_method : str
        ``"kmedoids"``, ``"gmm"``, or ``"kmeans"``.
    extra : np.ndarray
        Method-specific data (medoid_indices / means / centroids).

    Returns
    -------
    rep_docs : list[dict]
        Representative documents in cluster-id order.
    rep_indices : list[int]
        Index into `documents` for each representative.
    """
    if clustering_method == "kmedoids":
        # extra = medoid_indices
        medoid_indices = sorted(int(i) for i in extra)
        rep_docs = [
            {"cluster_id": int(cluster_labels[i]), "input": documents[i]["input"]}
            for i in medoid_indices
        ]
        return rep_docs, medoid_indices

    # GMM / KMeans — find closest doc to each centroid/mean
    from sklearn.preprocessing import normalize

    centroids = extra  # shape (k0, dim)
    emb_norm = normalize(embeddings, norm="l2")
    cent_norm = normalize(centroids, norm="l2")

    unique_clusters = sorted(set(int(c) for c in cluster_labels))
    rep_indices: list[int] = []
    rep_docs: list[dict] = []

    for cid in unique_clusters:
        mask = cluster_labels == cid
        members = np.where(mask)[0]
        if len(members) == 0:
            continue
        # Find member closest to centroid
        dists = np.linalg.norm(emb_norm[members] - cent_norm[cid], axis=1)
        best = members[int(np.argmin(dists))]
        rep_indices.append(int(best))
        rep_docs.append({
            "cluster_id": cid,
            "input": documents[best]["input"],
        })

    logger.info(
        "Stage 4: Selected %d representatives via %s",
        len(rep_docs), clustering_method,
    )
    return rep_docs, rep_indices


# ---------------------------------------------------------------------------
# Stage 5 — Label Discovery (one-word, general)
# ---------------------------------------------------------------------------

def discover_labels_v3(
    representative_texts: list[str],
    client,
    chunk_size: int = 30,
    run_dir: str | None = None,
    min_labels: int = 0,
    max_retries: int = 3,
) -> list[str]:
    """Stage 5: LLM label discovery — one-word general labels.

    When *min_labels* > 0 (typically set to K*), the function will retry
    label discovery if the first pass doesn't produce enough unique labels.
    Each retry pass shuffles the representative texts into different chunks
    and uses a smaller chunk size to encourage the LLM to produce more
    diverse labels.

    Parameters
    ----------
    representative_texts : list[str]
    client : OpenAI client
    chunk_size : int
    run_dir : str | None
        For checkpoint save/resume.
    min_labels : int
        Minimum number of unique labels required.  If > 0 and the first
        pass yields fewer labels, additional retry passes are performed
        with reshuffled chunks.  Typically set to ``k_star``.
    max_retries : int
        Maximum number of extra retry passes (default 3).

    Returns
    -------
    list[str]
        Unique candidate labels discovered.
    """
    all_labels: list[str] = []
    n_chunks = math.ceil(len(representative_texts) / chunk_size)

    # Checkpoint resume
    start_chunk = 0
    ckpt_path = (
        os.path.join(run_dir, "checkpoint_v3_labels.json") if run_dir else None
    )
    if ckpt_path:
        ckpt = _load_checkpoint(ckpt_path)
        if ckpt is not None:
            start_chunk = ckpt["processed_chunks"]
            all_labels = ckpt["all_labels"]
            logger.info(
                "[checkpoint] Resuming v3 label discovery from chunk %d/%d",
                start_chunk + 1, n_chunks,
            )

    ckpt_interval = max(2, n_chunks // 10)

    logger.info(
        "Stage 5 (v3): Discovering ONE-WORD labels from %d reps in %d chunks",
        len(representative_texts), n_chunks,
    )

    all_labels = _run_discovery_pass(
        representative_texts, client, chunk_size, all_labels,
        start_chunk, n_chunks, ckpt_path, ckpt_interval,
    )

    # ── Retry passes if we haven't reached min_labels ──
    if min_labels > 0 and len(all_labels) < min_labels:
        logger.warning(
            "Stage 5 (v3): Only %d labels discovered but min_labels=%d "
            "— starting retry passes (max %d)",
            len(all_labels), min_labels, max_retries,
        )
        for retry in range(1, max_retries + 1):
            if len(all_labels) >= min_labels:
                break

            # Shuffle representatives into different groupings
            shuffled = list(representative_texts)
            random.shuffle(shuffled)

            # Use a smaller chunk size to get more diverse label sets
            retry_chunk_size = max(5, chunk_size // (retry + 1))
            n_retry_chunks = math.ceil(len(shuffled) / retry_chunk_size)

            logger.info(
                "Stage 5 (v3): Retry %d/%d — reshuffled %d reps into "
                "%d chunks (chunk_size=%d), have %d/%d labels",
                retry, max_retries, len(shuffled), n_retry_chunks,
                retry_chunk_size, len(all_labels), min_labels,
            )

            all_labels = _run_discovery_pass(
                shuffled, client, retry_chunk_size, all_labels,
                start_chunk=0, n_chunks=n_retry_chunks,
                ckpt_path=None, ckpt_interval=n_retry_chunks,
                pass_label=f"retry-{retry}",
            )

        if len(all_labels) < min_labels:
            logger.warning(
                "Stage 5 (v3): After %d retries still only %d labels "
                "(target %d) — proceeding with what we have",
                max_retries, len(all_labels), min_labels,
            )

    logger.info("Stage 5 (v3): Final count — %d unique candidate labels", len(all_labels))

    if ckpt_path:
        _remove_checkpoint(ckpt_path)

    return all_labels


def _run_discovery_pass(
    texts: list[str],
    client,
    chunk_size: int,
    existing_labels: list[str],
    start_chunk: int,
    n_chunks: int,
    ckpt_path: str | None,
    ckpt_interval: int,
    pass_label: str = "pass-1",
) -> list[str]:
    """Run one pass of label discovery over chunked texts.

    Returns the updated (deduplicated) label list.
    """
    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_v3_discover_labels

    all_labels = list(existing_labels)

    for chunk_num in range(n_chunks):
        if chunk_num < start_chunk:
            continue

        start = chunk_num * chunk_size
        chunk = texts[start : start + chunk_size]
        prompt = prompt_v3_discover_labels(chunk)
        raw = chat(prompt, client, max_tokens=4096)
        if raw is None:
            logger.warning(
                "  [%s] Chunk %d: LLM returned None — skipping",
                pass_label, chunk_num + 1,
            )
            continue

        parsed = _safe_parse_labels(raw)
        if parsed:
            for label in parsed:
                if label not in all_labels:
                    all_labels.append(label)

        logger.info(
            "  [%s] Chunk %d/%d — labels so far: %d",
            pass_label, chunk_num + 1, n_chunks, len(all_labels),
        )

        if ckpt_path and (chunk_num + 1) % ckpt_interval == 0:
            _save_checkpoint(ckpt_path, {
                "processed_chunks": chunk_num + 1,
                "all_labels": all_labels,
            })

    return all_labels


# ---------------------------------------------------------------------------
# Stage 7 — Label Consolidation (iterative chunked merge)
# ---------------------------------------------------------------------------

def consolidate_labels_v3(
    candidate_labels: list[str],
    k_star: int,
    client,
    chunk_size: int = 200,
    max_rounds: int = 10,
) -> list[str]:
    """Stage 7: Merge candidate labels into exactly K* final labels.

    Uses iterative chunked merging for robustness when there are hundreds
    of candidate labels (same strategy as tools/remerge_labels.py).

    Parameters
    ----------
    candidate_labels : list[str]
    k_star : int
    client : OpenAI client
    chunk_size : int
        Max labels per LLM call.
    max_rounds : int

    Returns
    -------
    list[str]
        Approximately K* merged labels (best effort).
    """
    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_v3_consolidate_labels

    current = list(dict.fromkeys(candidate_labels))  # deduplicate
    logger.info(
        "Stage 7 (v3): Consolidating %d → %d labels",
        len(current), k_star,
    )

    for round_num in range(1, max_rounds + 1):
        n = len(current)

        if n <= k_star:
            logger.info("  Round %d: already at %d ≤ %d", round_num, n, k_star)
            break

        # Small enough for a single final merge
        if n <= chunk_size:
            logger.info(
                "  Round %d: final merge %d → %d (single call)",
                round_num, n, k_star,
            )
            best = current
            best_dist = abs(len(current) - k_star)
            for attempt in range(1, 4):
                prompt = prompt_v3_consolidate_labels(current, k_star)
                raw = chat(prompt, client, max_tokens=4096)
                parsed = _safe_parse_labels(raw)
                if parsed and len(parsed) < len(current):
                    dist = abs(len(parsed) - k_star)
                    logger.info(
                        "    attempt %d: %d labels (off by %d)",
                        attempt, len(parsed), dist,
                    )
                    if dist < best_dist:
                        best = parsed
                        best_dist = dist
                    if dist == 0:
                        break
                else:
                    logger.info("    attempt %d: parse failed or no reduction", attempt)
            current = best
            break

        # Chunked aggressive reduction
        n_chunks = math.ceil(n / chunk_size)
        per_chunk_target = max(3, int(k_star * 1.3) // n_chunks)
        merged_round: list[str] = []

        logger.info(
            "  Round %d: %d labels → %d chunks of ~%d (each → ~%d)",
            round_num, n, n_chunks, chunk_size, per_chunk_target,
        )

        for i in range(n_chunks):
            chunk = current[i * chunk_size : (i + 1) * chunk_size]
            chunk_target = max(3, int(per_chunk_target * len(chunk) / chunk_size))
            prompt = prompt_v3_consolidate_labels(chunk, chunk_target)
            raw = chat(prompt, client, max_tokens=4096)
            parsed = _safe_parse_labels(raw)
            result = parsed if parsed and len(parsed) < len(chunk) else chunk
            merged_round.extend(result)
            logger.info(
                "    chunk %d/%d: %d → %d",
                i + 1, n_chunks, len(chunk), len(result),
            )

        current = list(dict.fromkeys(merged_round))
        logger.info(
            "  Round %d done: %d → %d (target: %d)",
            round_num, n, len(current), k_star,
        )

        if len(current) >= n:
            logger.warning("  No reduction in round %d — stopping", round_num)
            break

    logger.info(
        "Stage 7 (v3): Final label count: %d (target was %d)",
        len(current), k_star,
    )
    return current


# ---------------------------------------------------------------------------
# Stage 8 — Representative Classification (batched)
# ---------------------------------------------------------------------------

def classify_representatives_v3(
    representative_texts: list[str],
    final_labels: list[str],
    client,
    batch_size: int = 20,
    run_dir: str | None = None,
) -> dict[int, str]:
    """Stage 8: Classify each representative into one of K* labels.

    Uses batched LLM calls for efficiency.

    Parameters
    ----------
    representative_texts : list[str]
    final_labels : list[str]
    client : OpenAI client
    batch_size : int
    run_dir : str | None

    Returns
    -------
    dict[int, str]
        Mapping from representative index → assigned label.
    """
    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_v3_classify_representatives_batch

    n_reps = len(representative_texts)
    rep_labels: dict[int, str] = {}

    # Checkpoint resume
    start_idx = 0
    ckpt_path = (
        os.path.join(run_dir, "checkpoint_v3_classify.json") if run_dir else None
    )
    if ckpt_path:
        ckpt = _load_checkpoint(ckpt_path)
        if ckpt is not None:
            start_idx = ckpt["processed_count"]
            rep_labels = {int(k): v for k, v in ckpt["rep_labels"].items()}
            logger.info(
                "[checkpoint] Resuming v3 classification from %d/%d",
                start_idx, n_reps,
            )

    ckpt_interval = max(1, n_reps // (batch_size * 10))

    logger.info(
        "Stage 8 (v3): Classifying %d representatives into %d labels (batch_size=%d)",
        n_reps, len(final_labels), batch_size,
    )

    batch_num = 0
    for i in range(0, n_reps, batch_size):
        if i < start_idx:
            continue

        batch = representative_texts[i : i + batch_size]
        prompt = prompt_v3_classify_representatives_batch(final_labels, batch)
        raw = chat(prompt, client, max_tokens=4096)

        if raw:
            parsed = _safe_parse_dict(raw)
            if parsed:
                for key, label in parsed.items():
                    idx = i + int(key) - 1  # keys are 1-indexed
                    if label in final_labels:
                        rep_labels[idx] = label
                    else:
                        # Fuzzy match: pick closest label
                        label_lower = label.lower()
                        for fl in final_labels:
                            if fl.lower() == label_lower:
                                rep_labels[idx] = fl
                                break
                        else:
                            rep_labels[idx] = final_labels[0]

        batch_num += 1
        classified = len(rep_labels)
        logger.info(
            "  Batch %d: classified %d / %d representatives",
            batch_num, classified, n_reps,
        )

        if ckpt_path and batch_num % ckpt_interval == 0:
            _save_checkpoint(ckpt_path, {
                "processed_count": i + len(batch),
                "rep_labels": rep_labels,
            })

    # Fill any missing
    for idx in range(n_reps):
        if idx not in rep_labels:
            rep_labels[idx] = "Unsuccessful"
            logger.warning("  Representative %d: no label assigned → Unsuccessful", idx)

    logger.info(
        "Stage 8 (v3): Classified %d / %d representatives",
        sum(1 for v in rep_labels.values() if v != "Unsuccessful"), n_reps,
    )

    if ckpt_path:
        _remove_checkpoint(ckpt_path)

    return rep_labels


# ---------------------------------------------------------------------------
# Stage 9 — Label Propagation
# ---------------------------------------------------------------------------

def propagate_labels_v3(
    rep_labels: dict[int, str],
    rep_indices: list[int],
    cluster_assignments: np.ndarray,
    n_documents: int,
) -> list[str]:
    """Stage 9: Propagate representative labels to all documents.

    Each document inherits the label of its cluster's representative.

    Parameters
    ----------
    rep_labels : dict[int, str]
        Mapping from rep_order → label (from Stage 8).
        ``rep_order`` is the positional index in the rep_indices list
        (which matches the order of rep_texts fed to Stage 8).
    rep_indices : list[int]
        Indices of representative documents in **original order** from
        Stage 4 (cluster-id order for GMM/KMeans, sorted-doc-index for
        K-Medoids).  Must NOT be re-sorted.
    cluster_assignments : np.ndarray
        Shape ``(n_documents,)`` — cluster id for each document.
    n_documents : int

    Returns
    -------
    list[str]
        Label for each document.
    """
    # Build cluster_id → label
    # rep_indices is in the same order as rep_texts that were classified,
    # so rep_labels[i] corresponds to rep_indices[i].
    cluster_to_label: dict[int, str] = {}
    for rep_order, rep_idx in enumerate(rep_indices):
        cid = int(cluster_assignments[rep_idx])
        label = rep_labels.get(rep_order, "Unsuccessful")
        cluster_to_label[cid] = label

    all_labels: list[str] = []
    for doc_idx in range(n_documents):
        cid = int(cluster_assignments[doc_idx])
        all_labels.append(cluster_to_label.get(cid, "Unsuccessful"))

    n_unsuccessful = sum(1 for lbl in all_labels if lbl == "Unsuccessful")
    if n_unsuccessful:
        logger.warning(
            "Stage 9 (v3): %d / %d documents labelled 'Unsuccessful'",
            n_unsuccessful, n_documents,
        )
    else:
        logger.info("Stage 9 (v3): All %d documents labelled", n_documents)

    return all_labels


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _safe_parse_labels(raw: str | None) -> list[str] | None:
    """Parse LLM response into a flat list of label strings."""
    if raw is None:
        return None

    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if isinstance(obj, list):
        return [str(x).strip() for x in obj if isinstance(x, str)]
    if isinstance(obj, dict):
        for val in obj.values():
            if isinstance(val, list):
                return [str(x).strip() for x in val if isinstance(x, str)]
    return None


def _safe_parse_dict(raw: str | None) -> dict[str, str] | None:
    """Parse LLM response into a {key: label} dict."""
    if raw is None:
        return None

    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if isinstance(obj, dict):
        return {str(k): str(v) for k, v in obj.items()}
    return None
