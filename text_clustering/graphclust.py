"""
graphclust.py — Graph Community Clustering with LLM Post-hoc Labelling.

A fundamentally different approach to text clustering that discovers clusters
via **graph community detection** on a k-NN embedding graph, then uses the
LLM **only for post-hoc labelling** — naming the discovered communities.

Paradigm Shift
--------------
All existing approaches (Original, K-Medoids, GMM, SEAL-Clust, Hybrid)
share one paradigm: **LLM-as-classifier**.  The LLM sees documents and
assigns/generates labels, then those labels drive the clustering.

This module introduces a fundamentally different paradigm:

1. **No labels exist during clustering** — clustering is label-free.
2. A **k-NN embedding similarity graph** is built (documents = nodes,
   edges = cosine similarity above threshold).
3. Clusters emerge from **Louvain community detection** on the graph
   topology — exploiting *graph structure*, not geometric partitioning
   (no cluster centers, no Gaussians, no Voronoi cells).
4. Labels are extracted **post-hoc** by asking the LLM to name each
   discovered community from a sample of its members.

Why This Is Different From KMeans/GMM
-------------------------------------
KMeans and GMM partition a *vector space* using geometric prototypes
(centroids / Gaussian means).  Louvain community detection partitions
a *graph* by maximising **modularity** — a measure of how densely
connected each community is internally vs. randomly expected.

Same embeddings, fundamentally different mathematics:
  - KMeans: assigns each point to the nearest centroid (Voronoi)
  - GMM:    assigns each point to the most-probable Gaussian
  - Louvain: assigns each node to the community that maximises
    ΔQ = [A_ij − γ k_i k_j / 2m] — a *topological* criterion

Louvain can discover **non-convex, non-spherical** clusters that
KMeans/GMM cannot, because it reasons about *connectivity patterns*
rather than *distance to a center*.

The 3-Step Pipeline
-------------------
  1. **Build k-NN Graph** — Cosine similarity between embeddings.
     Each document connects to its k nearest neighbours (above a
     minimum threshold).  No LLM calls.
  2. **Community Detection** — Louvain modularity optimisation with
     resolution parameter γ to control cluster granularity.  If
     target_k is specified, binary search over γ.
  3. **LLM Post-hoc Labelling** — For each community, sample
     representative documents and ask the LLM to name the topic.
     Only K LLM calls needed (~18 for massive_scenario).

Complexity
----------
Embedding:  O(n × d) — shared, cached.
Step 1:     O(n × k × d) — k-NN graph construction, no LLM.
Step 2:     O(n × iterations) — Louvain (typically ~5-20 iterations).
Step 3:     O(K) LLM calls — one per community.

Total LLM calls: **K** (just the number of clusters).  For massive_scenario
with K=18: **~18 LLM calls**.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Checkpoint helpers ────────────────────────────────────────────────────

def _save_gc_checkpoint(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_gc_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


def _remove_gc_checkpoint(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Step 1: Build k-NN Similarity Graph
# ---------------------------------------------------------------------------

def step1_build_knn_graph(
    embeddings: np.ndarray,
    knn: int = 15,
    min_similarity: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a k-nearest-neighbour graph from embeddings.

    Each document is connected to its k most similar documents (by cosine
    similarity), provided the similarity exceeds *min_similarity*.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_documents, dim)`` — raw or L2-normalised embeddings.
    knn : int
        Number of nearest neighbours per document.
    min_similarity : float
        Minimum cosine similarity to create an edge.

    Returns
    -------
    row_indices, col_indices, weights : np.ndarray
        Sparse edge list.  ``weights[i]`` is the cosine similarity between
        ``row_indices[i]`` and ``col_indices[i]``.
    """
    from sklearn.preprocessing import normalize

    n = embeddings.shape[0]
    emb_norm = normalize(embeddings, norm="l2")

    logger.info(
        "Step 1: Building k-NN graph — %d nodes, k=%d, min_sim=%.2f",
        n, knn, min_similarity,
    )

    rows_list: list[int] = []
    cols_list: list[int] = []
    weights_list: list[float] = []

    # Process in blocks to manage memory
    block_size = 500
    for i_start in tqdm(
        range(0, n, block_size),
        desc="Step 1: Building k-NN graph",
        unit="block",
        ncols=90,
        total=(n + block_size - 1) // block_size,
    ):
        i_end = min(i_start + block_size, n)
        block = emb_norm[i_start:i_end]  # (block_size, dim)
        sim = block @ emb_norm.T  # (block_size, n)

        for local_i in range(i_end - i_start):
            global_i = i_start + local_i
            row_sim = sim[local_i].copy()
            row_sim[global_i] = -np.inf  # exclude self

            # Top-k neighbours
            if knn < n - 1:
                top_k_idx = np.argpartition(row_sim, -knn)[-knn:]
            else:
                top_k_idx = np.arange(n)
                top_k_idx = top_k_idx[top_k_idx != global_i]

            for j in top_k_idx:
                w = float(sim[local_i, j])
                if w >= min_similarity:
                    rows_list.append(global_i)
                    cols_list.append(int(j))
                    weights_list.append(w)

    row_indices = np.array(rows_list, dtype=np.int32)
    col_indices = np.array(cols_list, dtype=np.int32)
    weights = np.array(weights_list, dtype=np.float32)

    # Compute stats
    n_edges = len(weights)
    n_connected = len(set(row_indices) | set(col_indices)) if n_edges > 0 else 0
    n_isolated = n - n_connected

    logger.info(
        "Step 1: Graph complete — %d nodes, %d edges (avg degree=%.1f), "
        "%d isolated nodes",
        n, n_edges, 2 * n_edges / max(n, 1), n_isolated,
    )

    return row_indices, col_indices, weights


# ---------------------------------------------------------------------------
# Step 2: Community Detection (Louvain)
# ---------------------------------------------------------------------------

def step2_detect_communities(
    n_nodes: int,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    weights: np.ndarray,
    resolution: float = 1.0,
    target_k: int = 0,
    resolution_range: tuple[float, float] = (0.1, 10.0),
    resolution_steps: int = 30,
) -> tuple[np.ndarray, int, float]:
    """Detect communities in the similarity graph via Louvain algorithm.

    If *target_k* > 0, uses **binary search** over resolution parameter
    to find the value that produces closest to *target_k* communities.

    Parameters
    ----------
    n_nodes : int
        Number of nodes (documents).
    row_indices, col_indices, weights : np.ndarray
        Edge list from Step 1.
    resolution : float
        Louvain resolution parameter (γ).  Higher → more communities.
    target_k : int
        If > 0, auto-tune resolution to approach this many communities.
    resolution_range : tuple[float, float]
        Range for resolution search.
    resolution_steps : int
        Max number of binary-search iterations.

    Returns
    -------
    labels : np.ndarray
        Community assignment for each node, shape ``(n_nodes,)``.
    n_communities : int
        Number of discovered communities.
    best_resolution : float
        The resolution that was used.
    """
    from scipy.sparse import csr_matrix

    logger.info(
        "Step 2: Community detection — %d nodes, %d edges, target_k=%s",
        n_nodes, len(weights), target_k or "auto",
    )

    # Build symmetric sparse adjacency matrix
    all_rows = np.concatenate([row_indices, col_indices])
    all_cols = np.concatenate([col_indices, row_indices])
    all_weights = np.concatenate([weights, weights])

    adj = csr_matrix(
        (all_weights, (all_rows, all_cols)),
        shape=(n_nodes, n_nodes),
    )

    def _louvain(adj_matrix, gamma: float) -> np.ndarray:
        """Louvain community detection on a sparse matrix.

        Modularity:  Q = (1/2m) Σ [A_ij − γ k_i k_j / 2m] δ(c_i, c_j)

        This is a *graph-topological* criterion — fundamentally different
        from geometric partitioning (KMeans centroids / GMM Gaussians).
        """
        n = adj_matrix.shape[0]
        communities = np.arange(n)
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        m2 = degrees.sum()  # 2m

        if m2 < 1e-10:
            return communities

        improved = True
        iteration = 0
        max_iterations = 50

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            order = np.random.permutation(n)

            for node in order:
                current_comm = communities[node]
                node_degree = degrees[node]

                # Get neighbour weights as dense row
                row = adj_matrix[node]
                if hasattr(row, "toarray"):
                    row = row.toarray().flatten()
                else:
                    row = np.array(row.todense()).flatten()

                # Only consider communities of actual neighbours
                neighbour_comms = set(communities[row > 0]) - {current_comm}
                if not neighbour_comms:
                    continue

                best_gain = 0.0
                best_comm = current_comm

                current_members = communities == current_comm
                w_to_current = float(row[current_members].sum())
                k_current = float(degrees[current_members].sum())

                for comm in neighbour_comms:
                    comm_members = communities == comm
                    w_to_comm = float(row[comm_members].sum())
                    k_comm = float(degrees[comm_members].sum())

                    # ΔQ for removing from current and adding to comm
                    gain = (
                        (w_to_comm - w_to_current)
                        - gamma * node_degree
                        * (k_comm - k_current + node_degree) / m2
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_comm = comm

                if best_comm != current_comm:
                    communities[node] = best_comm
                    improved = True

            # Re-number communities contiguously
            unique_comms = np.unique(communities)
            remap = {old: new for new, old in enumerate(unique_comms)}
            communities = np.array([remap[c] for c in communities])

        return communities

    if target_k > 0:
        # Binary search for best resolution
        lo, hi = resolution_range
        logger.info(
            "  Binary search for target k=%d in γ ∈ [%.2f, %.2f]",
            target_k, lo, hi,
        )

        best_labels = None
        best_k = 0
        best_res = resolution
        best_diff = float("inf")

        for step in range(resolution_steps):
            gamma = (lo + hi) / 2.0
            labels = _louvain(adj, gamma)
            k = len(np.unique(labels))
            diff = abs(k - target_k)

            logger.info(
                "    [%d] γ=%.4f → k=%d (target=%d, diff=%d)",
                step + 1, gamma, k, target_k, diff,
            )

            if diff < best_diff:
                best_diff = diff
                best_k = k
                best_labels = labels.copy()
                best_res = gamma

            if diff == 0:
                break

            # Binary search direction
            if k < target_k:
                lo = gamma  # more communities → raise resolution
            else:
                hi = gamma  # fewer communities → lower resolution

            # Converged
            if hi - lo < 0.001:
                break

        labels = best_labels
        n_communities = best_k
        best_resolution = best_res
    else:
        labels = _louvain(adj, resolution)
        n_communities = len(np.unique(labels))
        best_resolution = resolution

    logger.info(
        "Step 2: Detected %d communities (resolution=%.4f)",
        n_communities, best_resolution,
    )

    # Log community sizes
    comm_sizes = Counter(labels)
    for comm_id, size in sorted(comm_sizes.items(), key=lambda x: -x[1])[:20]:
        logger.info("  Community %d: %d documents", comm_id, size)
    if len(comm_sizes) > 20:
        logger.info("  ... (%d more communities)", len(comm_sizes) - 20)

    return labels, n_communities, best_resolution


# ---------------------------------------------------------------------------
# Step 3: LLM Post-hoc Label Extraction
# ---------------------------------------------------------------------------

def _build_label_prompt(sample_texts: list[str]) -> str:
    """Build a prompt asking the LLM to name the common topic of a group."""
    json_example = {"topic": "weather forecasts"}
    prompt = (
        "You are an expert text analyst. The following texts all belong to the "
        "same topic cluster. Read them carefully and provide a single, short "
        "descriptive label (2-4 words) that captures the common theme.\n\n"
        "Texts:\n"
    )
    for i, text in enumerate(sample_texts, 1):
        prompt += f"{i}. \"{text}\"\n"
    prompt += (
        f"\nReturn the topic label in JSON format like: {json_example}\n"
        "Use a descriptive phrase, NOT generic labels like 'Other' or 'Miscellaneous'."
    )
    return prompt


def step3_label_communities(
    community_labels: np.ndarray,
    texts: list[str],
    client,
    samples_per_community: int = 8,
    random_state: int = 42,
    run_dir: str | None = None,
) -> dict[int, str]:
    """Name each discovered community by sampling representatives.

    For each community, randomly sample a few documents and ask the LLM
    to identify the common theme.  This is post-hoc labelling — the
    clustering was done without labels.

    Parameters
    ----------
    community_labels : np.ndarray
        Community ID for each document.
    texts : list[str]
        All document texts.
    client
        OpenAI-compatible client.
    samples_per_community : int
        Number of documents to sample per community for labelling.
    random_state : int
    run_dir : str | None
        If provided, save/load checkpoints to this directory.

    Returns
    -------
    dict[int, str]
        ``{community_id: topic_label}``.
    """
    from text_clustering.llm import chat

    rng = np.random.RandomState(random_state)
    n_communities = len(np.unique(community_labels))

    logger.info(
        "Step 3: Labelling %d communities (%d samples each) …",
        n_communities, samples_per_community,
    )

    community_names: dict[int, str] = {}
    unique_comms = sorted(set(community_labels))

    # ── Checkpoint resume ──
    ckpt_path = os.path.join(run_dir, "checkpoint_graphclust_step3.json") if run_dir else None
    processed_comms: set[int] = set()
    if ckpt_path:
        ckpt = _load_gc_checkpoint(ckpt_path)
        if ckpt is not None:
            community_names = {int(k): v for k, v in ckpt["community_names"].items()}
            processed_comms = set(int(c) for c in ckpt["processed_comms"])
            logger.info("[checkpoint] Resuming graphclust step 3 from %d/%d communities", len(processed_comms), n_communities)

    ckpt_interval = max(5, n_communities // 10)  # save every ~10%
    labelled_count = len(processed_comms)

    for comm_id in tqdm(
        unique_comms,
        desc="Step 3: Labelling communities",
        unit="comm",
        ncols=90,
    ):
        if int(comm_id) in processed_comms:
            continue

        members = np.where(community_labels == comm_id)[0]
        n_sample = min(samples_per_community, len(members))
        sample_idx = rng.choice(members, size=n_sample, replace=False)
        sample_texts = [texts[int(i)] for i in sample_idx]

        prompt = _build_label_prompt(sample_texts)
        raw = chat(prompt, client, max_tokens=512)

        label = f"Topic_{comm_id}"  # fallback

        if raw is not None:
            try:
                parsed = eval(raw)  # noqa: S307
                if isinstance(parsed, dict):
                    for val in parsed.values():
                        if isinstance(val, str) and len(val) > 1:
                            label = val.strip()
                            break
                elif isinstance(parsed, str):
                    label = parsed.strip()
            except Exception:
                # Try extracting from raw text
                if len(raw.strip()) < 100:
                    label = raw.strip().strip('"').strip("'")

        community_names[int(comm_id)] = label
        labelled_count += 1
        logger.info(
            "  Community %d (%d docs): \"%s\"",
            comm_id, len(members), label,
        )

        # ── Checkpoint save ──
        if ckpt_path and labelled_count % ckpt_interval == 0:
            _save_gc_checkpoint(ckpt_path, {
                "processed_comms": list(community_names.keys()),
                "community_names": {str(k): v for k, v in community_names.items()},
            })
            logger.info("[checkpoint] Saved graphclust step 3: %d/%d communities", labelled_count, n_communities)

    # Clean up checkpoint on successful completion
    if ckpt_path:
        _remove_gc_checkpoint(ckpt_path)

    return community_names


# ---------------------------------------------------------------------------
# Full Pipeline Helper
# ---------------------------------------------------------------------------

def build_classifications(
    community_labels: np.ndarray,
    community_names: dict[int, str],
    texts: list[str],
) -> dict[str, list[str]]:
    """Convert community assignments to the standard classifications format.

    Returns
    -------
    dict[str, list[str]]
        ``{label_name: [text1, text2, ...]}``.
    """
    classifications: dict[str, list[str]] = {}
    for doc_idx, comm_id in enumerate(community_labels):
        label = community_names.get(int(comm_id), f"Topic_{comm_id}")
        classifications.setdefault(label, []).append(texts[doc_idx])
    return classifications
