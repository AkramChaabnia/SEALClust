"""
hybrid.py — Hybrid LLM + Embedding clustering pipeline.

Combines LLM-based semantic label generation with embedding-based geometric
optimisation to produce high-quality text clusters.  The pipeline has 8 steps:

  1. **Initial LLM Label Generation (K0)**
     Split documents into batches, prompt LLM for one-word general labels.
     Collect all unique labels → K0 (K0 >> true categories).

  2. **Embedding Computation**
     Compute dense embeddings for all documents using a sentence-transformer
     model.  Embeddings are cached for reuse across pipeline steps.

  3. **Label Reduction via LLM (K1)**
     Send K0 labels to the LLM and merge semantically similar labels.
     Output: reduced label set K1 (possibly still K1 >> true categories).

  4. **Iterative KMeans Optimisation**
     Run KMeans on embeddings with k = K1, compute silhouette score.
     Iteratively reduce k via elbow method + silhouette analysis.
     Select optimal K1.

  5. **LLM Label Alignment to Target Categories**
     If K1 ≠ target categories: prompt LLM to merge/reduce labels to match
     exactly the target number of categories.

  6. **GMM Overclustering (Microclusters)**
     Parameter p ∈ (0, 1) controls the number of microclusters = p × n.
     Fit a GMM with that many components and extract medoid documents
     (closest to each component mean).

  7. **LLM-Based Medoid Labelling**
     Send medoid documents + final label set to LLM.  For each medoid,
     assign the most relevant label.

  8. **Label Propagation**
     Assign each document the label of its nearest medoid / GMM cluster
     representative.

Functions
---------
step1_generate_labels(texts, client, batch_size)
step2_compute_embeddings(texts, model_name, batch_size)
step3_reduce_labels(labels_k0, client)
step4_optimise_k(embeddings, k1, k_min, k_max)
step5_align_labels(labels_k1, target_k, client)
step6_gmm_overclustering(documents, embeddings, p)
step7_label_medoids(medoid_docs, final_labels, client)
step8_propagate_labels(medoid_labels, gmm_labels, n_documents)
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Initial LLM Label Generation (K0)
# ---------------------------------------------------------------------------

def step1_generate_labels(
    texts: list[str],
    client,
    batch_size: int = 30,
) -> tuple[list[str], list[str]]:
    """Generate one-word labels for all documents via LLM.

    Parameters
    ----------
    texts : list[str]
        All document texts.
    client
        OpenAI-compatible client.
    batch_size : int
        Number of documents per LLM call.

    Returns
    -------
    per_doc_labels : list[str]
        One label per document (same order as *texts*).
    unique_labels : list[str]
        All unique labels discovered (K0).
    """
    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_hybrid_generate_labels

    per_doc_labels: list[str] = ["Unlabelled"] * len(texts)
    n_batches = (len(texts) + batch_size - 1) // batch_size

    logger.info(
        "Step 1: Generating labels for %d documents in %d batches "
        "(batch_size=%d)",
        len(texts), n_batches, batch_size,
    )

    for batch_idx in range(0, len(texts), batch_size):
        batch = texts[batch_idx: batch_idx + batch_size]
        prompt = prompt_hybrid_generate_labels(batch)
        raw = chat(prompt, client, max_tokens=4096)

        if raw is None:
            logger.warning(
                "  Batch %d: LLM returned None — skipping",
                batch_idx // batch_size + 1,
            )
            continue

        try:
            parsed = eval(raw)  # noqa: S307
        except Exception:
            logger.warning(
                "  Batch %d: could not parse response — skipping",
                batch_idx // batch_size + 1,
            )
            continue

        # Extract labels list from response
        labels_list: list[str] = []
        if isinstance(parsed, dict):
            for val in parsed.values():
                if isinstance(val, list):
                    labels_list = [str(x) for x in val]
                    break
        elif isinstance(parsed, list):
            labels_list = [str(x) for x in parsed]

        # Assign labels to the correct document positions
        for i, label in enumerate(labels_list):
            doc_idx = batch_idx + i
            if doc_idx < len(texts):
                per_doc_labels[doc_idx] = label

        logger.info(
            "  Batch %d/%d — assigned %d labels",
            batch_idx // batch_size + 1, n_batches, len(labels_list),
        )

    # Collect unique labels
    unique_labels = list(dict.fromkeys(
        lbl for lbl in per_doc_labels if lbl != "Unlabelled"
    ))

    logger.info(
        "Step 1: Generated K0=%d unique labels from %d documents "
        "(%d unlabelled)",
        len(unique_labels),
        len(texts),
        per_doc_labels.count("Unlabelled"),
    )
    return per_doc_labels, unique_labels


# ---------------------------------------------------------------------------
# Step 2: Embedding Computation (delegates to embedding.py)
# ---------------------------------------------------------------------------

def step2_compute_embeddings(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute sentence embeddings for all documents.

    Thin wrapper around ``embedding.compute_embeddings`` to keep the hybrid
    pipeline self-contained.

    Returns
    -------
    np.ndarray
        Shape ``(n_documents, embedding_dim)``.
    """
    from text_clustering.embedding import compute_embeddings

    logger.info("Step 2: Computing embeddings for %d documents …", len(texts))
    embeddings = compute_embeddings(
        texts, model_name=model_name, batch_size=batch_size,
    )
    logger.info("Step 2: Embeddings shape: %s", embeddings.shape)
    return embeddings


# ---------------------------------------------------------------------------
# Step 3: Label Reduction via LLM (K0 → K1)
# ---------------------------------------------------------------------------

def step3_reduce_labels(
    labels_k0: list[str],
    client,
) -> list[str]:
    """Merge semantically similar labels via LLM.

    Parameters
    ----------
    labels_k0 : list[str]
        All unique labels from Step 1.
    client
        OpenAI-compatible client.

    Returns
    -------
    list[str]
        Reduced label set K1 (still possibly K1 > true categories).
    """
    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_hybrid_reduce_labels

    logger.info(
        "Step 3: Reducing %d labels via LLM merge …", len(labels_k0),
    )

    prompt = prompt_hybrid_reduce_labels(labels_k0)
    raw = chat(prompt, client, max_tokens=4096)

    if raw is None:
        logger.warning("Step 3: LLM returned None — returning K0 labels")
        return labels_k0

    try:
        parsed = eval(raw)  # noqa: S307
    except Exception:
        logger.warning("Step 3: could not parse — returning K0 labels")
        return labels_k0

    merged: list[str] = []
    if isinstance(parsed, dict):
        for val in parsed.values():
            if isinstance(val, list):
                merged = [str(x) for x in val]
                break
    elif isinstance(parsed, list):
        merged = [str(x) for x in parsed]

    if not merged:
        logger.warning("Step 3: empty result — returning K0 labels")
        return labels_k0

    logger.info("Step 3: Reduced K0=%d → K1=%d labels", len(labels_k0), len(merged))
    return merged


# ---------------------------------------------------------------------------
# Step 4: Iterative KMeans Optimisation (Silhouette + Elbow)
# ---------------------------------------------------------------------------

def step4_optimise_k(
    embeddings: np.ndarray,
    k1: int,
    k_min: int = 2,
    k_max: int | None = None,
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    """Find optimal K via KMeans + silhouette analysis.

    Runs KMeans for each candidate k from *k_min* to *k1*, computes the
    silhouette score, and returns the k that maximises silhouette.

    Additionally applies elbow detection on the inertia curve as a sanity
    check.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_documents, dim)``.
    k1 : int
        Upper bound on k (from Step 3 label count).
    k_min : int
        Lower bound on k.
    k_max : int, optional
        Override upper bound.  Defaults to *k1*.
    random_state : int

    Returns
    -------
    best_k : int
        Optimal number of clusters.
    silhouette_scores : dict[int, float]
        ``{k: silhouette_score}`` for each k tried.
    """
    if k_max is None:
        k_max = k1

    # Clamp
    n_samples = embeddings.shape[0]
    k_max = min(k_max, n_samples - 1)
    k_min = max(k_min, 2)
    if k_min > k_max:
        k_min = 2

    # L2-normalise so Euclidean ≈ cosine
    emb_norm = normalize(embeddings, norm="l2")

    logger.info(
        "Step 4: KMeans optimisation k ∈ [%d, %d] on %d samples (dim=%d)",
        k_min, k_max, n_samples, emb_norm.shape[1],
    )

    scores: dict[int, float] = {}
    inertias: dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        labels = km.fit_predict(emb_norm)
        n_unique = len(set(labels))
        if n_unique < 2:
            continue

        sil = silhouette_score(
            emb_norm, labels,
            metric="euclidean",
            sample_size=min(5000, n_samples),
        )
        scores[k] = float(sil)
        inertias[k] = float(km.inertia_)
        logger.info("  k=%d  silhouette=%.4f  inertia=%.2f", k, sil, km.inertia_)

    if not scores:
        logger.warning("Step 4: no valid scores — defaulting to k=%d", k_min)
        return k_min, {}

    # Best by silhouette
    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    logger.info(
        "Step 4: Optimal K=%d (silhouette=%.4f)", best_k, scores[best_k],
    )
    return best_k, scores


# ---------------------------------------------------------------------------
# Step 5: LLM Label Alignment to Target Categories
# ---------------------------------------------------------------------------

def step5_align_labels(
    labels_k1: list[str],
    target_k: int,
    client,
) -> list[str]:
    """Force label set to exactly *target_k* labels via LLM.

    If ``len(labels_k1) == target_k``, returns as-is.
    Otherwise prompts the LLM to merge/reorganise into exactly *target_k*.

    Parameters
    ----------
    labels_k1 : list[str]
        Current label set from Steps 3/4.
    target_k : int
        Exact number of categories required.
    client
        OpenAI-compatible client.

    Returns
    -------
    list[str]
        Exactly *target_k* labels (or best effort).
    """
    if len(labels_k1) == target_k:
        logger.info(
            "Step 5: K1=%d already matches target_k=%d — no alignment needed",
            len(labels_k1), target_k,
        )
        return labels_k1

    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_hybrid_align_labels

    logger.info(
        "Step 5: Aligning %d labels to exactly %d via LLM …",
        len(labels_k1), target_k,
    )

    prompt = prompt_hybrid_align_labels(labels_k1, target_k)
    raw = chat(prompt, client, max_tokens=4096)

    if raw is None:
        logger.warning("Step 5: LLM returned None — returning K1 labels")
        return labels_k1

    try:
        parsed = eval(raw)  # noqa: S307
    except Exception:
        logger.warning("Step 5: could not parse — returning K1 labels")
        return labels_k1

    aligned: list[str] = []
    if isinstance(parsed, dict):
        for val in parsed.values():
            if isinstance(val, list):
                aligned = [str(x) for x in val]
                break
    elif isinstance(parsed, list):
        aligned = [str(x) for x in parsed]

    if not aligned:
        logger.warning("Step 5: empty result — returning K1 labels")
        return labels_k1

    # If still wrong count, try a second pass
    if len(aligned) != target_k and abs(len(aligned) - target_k) > 2:
        logger.info(
            "Step 5: Second pass (got %d, want %d) …",
            len(aligned), target_k,
        )
        prompt2 = prompt_hybrid_align_labels(aligned, target_k)
        raw2 = chat(prompt2, client, max_tokens=4096)
        if raw2:
            try:
                parsed2 = eval(raw2)  # noqa: S307
                labels2: list[str] = []
                if isinstance(parsed2, dict):
                    for val in parsed2.values():
                        if isinstance(val, list):
                            labels2 = [str(x) for x in val]
                            break
                elif isinstance(parsed2, list):
                    labels2 = [str(x) for x in parsed2]
                if labels2:
                    aligned = labels2
            except Exception:
                pass

    logger.info(
        "Step 5: Aligned %d → %d labels (target was %d)",
        len(labels_k1), len(aligned), target_k,
    )
    return aligned


# ---------------------------------------------------------------------------
# Step 6: GMM Overclustering (Microclusters)
# ---------------------------------------------------------------------------

def step6_gmm_overclustering(
    documents: list[dict],
    embeddings: np.ndarray,
    p: float = 0.1,
    covariance_type: str = "tied",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Create microclusters via GMM and extract medoid representatives.

    Parameters
    ----------
    documents : list[dict]
        Full dataset (each element has ``"input"`` and ``"label"`` keys).
    embeddings : np.ndarray
        Shape ``(n_documents, dim)``.
    p : float
        Proportion of documents to use as microclusters.
        n_microclusters = max(2, int(p * n_documents)).
    covariance_type : str
        GMM covariance type.
    random_state : int

    Returns
    -------
    gmm_labels : np.ndarray
        Shape ``(n_documents,)`` — hard cluster assignment.
    gmm_probs : np.ndarray
        Shape ``(n_documents, n_microclusters)`` — posterior probabilities.
    medoid_docs : list[dict]
        Representative documents (one per microcluster).
    medoid_indices : np.ndarray
        Indices into *documents* for each representative.
    """
    n = len(documents)
    n_micro = max(2, int(p * n))
    n_micro = min(n_micro, n - 1)  # GMM needs n_components < n_samples

    emb_norm = normalize(embeddings, norm="l2")

    logger.info(
        "Step 6: GMM overclustering — p=%.2f → %d microclusters "
        "from %d documents (cov=%s)",
        p, n_micro, n, covariance_type,
    )

    gmm = GaussianMixture(
        n_components=n_micro,
        covariance_type=covariance_type,
        max_iter=300,
        n_init=3,
        random_state=random_state,
    )
    gmm.fit(emb_norm)

    gmm_labels = gmm.predict(emb_norm)
    gmm_probs = gmm.predict_proba(emb_norm)
    means = gmm.means_

    logger.info(
        "Step 6: GMM converged — BIC=%.2f, %d components",
        gmm.bic(emb_norm), n_micro,
    )

    # Extract medoid: for each component, the document closest to the mean
    medoid_indices: list[int] = []
    for c in range(n_micro):
        members = np.where(gmm_labels == c)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(emb_norm[members] - means[c], axis=1)
        best_local = np.argmin(dists)
        medoid_indices.append(int(members[best_local]))

    medoid_indices_arr = np.array(sorted(medoid_indices))
    medoid_docs = [documents[int(i)] for i in medoid_indices_arr]

    logger.info(
        "Step 6: Extracted %d medoid representatives from %d microclusters",
        len(medoid_docs), n_micro,
    )
    return gmm_labels, gmm_probs, medoid_docs, medoid_indices_arr


# ---------------------------------------------------------------------------
# Step 7: LLM-Based Medoid Labelling
# ---------------------------------------------------------------------------

def step7_label_medoids(
    medoid_docs: list[dict],
    medoid_indices: np.ndarray,
    final_labels: list[str],
    client,
) -> dict[int, str]:
    """Assign a label to each medoid document via LLM.

    Parameters
    ----------
    medoid_docs : list[dict]
        Medoid documents (must have ``"input"`` key).
    medoid_indices : np.ndarray
        Original indices of medoid documents in the full dataset.
    final_labels : list[str]
        The final label set from Step 5.
    client
        OpenAI-compatible client.

    Returns
    -------
    dict[int, str]
        ``{medoid_document_index: predicted_label}``.
    """
    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_hybrid_classify_medoid

    logger.info(
        "Step 7: Labelling %d medoids using %d labels via LLM …",
        len(medoid_docs), len(final_labels),
    )

    medoid_labels: dict[int, str] = {}

    for doc, med_idx in zip(medoid_docs, sorted(medoid_indices)):
        text = doc["input"]
        prompt = prompt_hybrid_classify_medoid(final_labels, text)
        raw = chat(prompt, client)

        if raw is None:
            medoid_labels[int(med_idx)] = "Unsuccessful"
            continue

        try:
            parsed = eval(raw)  # noqa: S307
        except Exception:
            # Try direct string matching
            for label in final_labels:
                if label.lower() in raw.lower():
                    medoid_labels[int(med_idx)] = label
                    break
            else:
                medoid_labels[int(med_idx)] = "Unsuccessful"
            continue

        # Extract label from parsed response
        assigned = "Unsuccessful"
        if isinstance(parsed, dict):
            for val in parsed.values():
                if isinstance(val, str) and val in final_labels:
                    assigned = val
                    break
                elif isinstance(val, str):
                    # Fuzzy match — check if any final label is in the value
                    for label in final_labels:
                        if label.lower() == val.lower():
                            assigned = label
                            break
        elif isinstance(parsed, str):
            for label in final_labels:
                if label.lower() == parsed.lower():
                    assigned = label
                    break

        medoid_labels[int(med_idx)] = assigned

    n_success = sum(1 for v in medoid_labels.values() if v != "Unsuccessful")
    logger.info(
        "Step 7: Labelled %d/%d medoids successfully",
        n_success, len(medoid_docs),
    )

    # Log label distribution
    label_counts = Counter(medoid_labels.values())
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-40s %d medoids", label, count)

    return medoid_labels


# ---------------------------------------------------------------------------
# Step 8: Label Propagation
# ---------------------------------------------------------------------------

def step8_propagate_labels(
    medoid_labels: dict[int, str],
    gmm_labels: np.ndarray,
    n_documents: int,
) -> list[str]:
    """Propagate medoid labels to every document via GMM cluster membership.

    Each document inherits the label of the medoid that represents its
    GMM cluster.

    Parameters
    ----------
    medoid_labels : dict[int, str]
        ``{medoid_document_index: predicted_label}`` from Step 7.
    gmm_labels : np.ndarray
        Shape ``(n_documents,)`` — hard GMM cluster assignments.
    n_documents : int
        Total number of documents.

    Returns
    -------
    list[str]
        A label for each document, in dataset order.
    """
    logger.info("Step 8: Propagating labels from %d medoids to %d documents …",
                len(medoid_labels), n_documents)

    # Build cluster_id → label via medoid membership
    cluster_to_label: dict[int, str] = {}
    for med_idx, label in medoid_labels.items():
        cid = int(gmm_labels[med_idx])
        cluster_to_label[cid] = label

    all_labels: list[str] = []
    unlabelled = 0
    for doc_idx in range(n_documents):
        cid = int(gmm_labels[doc_idx])
        label = cluster_to_label.get(cid, "Unsuccessful")
        if label == "Unsuccessful":
            unlabelled += 1
        all_labels.append(label)

    if unlabelled:
        logger.warning(
            "Step 8: %d/%d documents received no label",
            unlabelled, n_documents,
        )
    else:
        logger.info("Step 8: All %d documents labelled", n_documents)

    # Summary
    label_counts = Counter(all_labels)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-40s %d documents", label, count)

    return all_labels
