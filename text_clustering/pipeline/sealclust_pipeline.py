"""
sealclust_pipeline.py — SEALClust 9-stage pipeline CLI.

Implements the complete SEALClust (Scalable Efficient Autonomous LLM
Clustering) framework as a single entry point (``tc-sealclust``).

The 9-Stage Pipeline
--------------------
  1. **Document Embedding** — all-MiniLM-L6-v2 (384d)
  2. **Dimensionality Reduction** — PCA (default 50d) or t-SNE
  3. **Overclustering** — K-Medoids with K₀ >> K* (e.g. 300)
  4. **Representative Selection** — medoid documents (actual docs, not centroids)
  5. **Label Discovery** — LLM proposes labels from representative docs ONLY
  6. **K* Estimation** — GMM + BIC on representative embeddings
  7. **Label Consolidation** — LLM merges candidate labels to exactly K*
  8. **Representative Classification** — LLM classifies K₀ reps into K* labels
  9. **Label Propagation** — propagate rep labels to all documents

Usage Modes
-----------
**Full pipeline** (Stages 1–7, then 8–9 separately)::

    # Stages 1–7: Embed + PCA + Overcluster + Label Discovery + BIC + Consolidate
    tc-sealclust --data massive_scenario

    # Stage 8: Classify representatives with K* labels
    tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode

    # Stage 9: Propagate labels to full dataset
    tc-sealclust --data massive_scenario --run_dir ./runs/<run_dir> --propagate

    # Evaluate
    tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>

**Custom K₀ (overclustering size)**::

    tc-sealclust --data massive_scenario --k0 200

**Manual K* (skip BIC estimation)**::

    tc-sealclust --data massive_scenario --k_star 18

**Use t-SNE instead of PCA**::

    tc-sealclust --data massive_scenario --reduction tsne
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np

from text_clustering.config import (
    EMBEDDING_MODEL,
    SEALCLUST_K0,
    SEALCLUST_BIC_K_MIN,
    SEALCLUST_BIC_K_MAX,
    SEALCLUST_REDUCTION,
    SEALCLUST_PCA_DIMS,
    SEALCLUST_LABEL_CHUNK_SIZE,
    SEALCLUST_K,
    TSNE_N_COMPONENTS,
    TSNE_PERPLEXITY,
    TSNE_N_ITER,
    TSNE_METRIC,
)
from text_clustering.data import load_dataset
from text_clustering.embedding import compute_embeddings
from text_clustering.dimreduce import reduce_tsne, reduce_pca
from text_clustering.sealclust import (
    run_sealclust_clustering,
    estimate_k_star_bic,
    discover_labels,
    consolidate_labels,
    get_prototypes,
    propagate_labels,
)
from text_clustering.kmedoids import build_cluster_map
from text_clustering.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _make_run_dir(runs_dir: str, data: str, size: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"{data}_{size}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Wrote %s", path)


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Wrote %s (%d records)", path, len(records))


# ── full pipeline: Stages 1–7 ─────────────────────────────────────────────

def run_pipeline(args) -> str:
    """Run the full SEALClust pipeline (Stages 1–7).

    Stages executed:
      1. Document Embedding (or cache reuse)
      2. Dimensionality Reduction — PCA (default) or t-SNE
      3. Overclustering — K-Medoids with K₀
      4. Representative Selection — medoid documents
      5. Label Discovery — LLM on representatives only
      6. K* Estimation — GMM + BIC on representative embeddings
      7. Label Consolidation — merge to exactly K*

    Returns the run directory path.
    """
    size = "large" if args.use_large else "small"

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size)

    setup_logging(os.path.join(run_dir, "sealclust_pipeline.log"))

    logger.info("=" * 70)
    logger.info("SEALClust — 9-Stage Pipeline (Stages 1–7)")
    logger.info("=" * 70)
    logger.info("Dataset       : %s  |  split: %s", args.data, size)
    logger.info("Embedding     : %s", args.embedding_model)
    logger.info("Reduction     : %s", args.reduction)
    if args.reduction == "pca":
        logger.info("PCA dims      : %d", args.pca_dims)
    else:
        logger.info("t-SNE         : n_components=%d, perplexity=%.1f, metric=%s",
                    args.tsne_n_components, args.tsne_perplexity, args.tsne_metric)
    logger.info("K₀ (overclust): %d", args.k0)
    if args.k_star:
        logger.info("K* (manual)   : %d", args.k_star)
    else:
        logger.info("K* (auto)     : GMM + BIC [%d–%d]", args.bic_k_min, args.bic_k_max)
    logger.info("Run dir       : %s", run_dir)
    logger.info("-" * 70)
    start = time.time()

    # ── Stage 1: Load dataset + Compute embeddings ──
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    texts = [item["input"] for item in data_list]
    n_documents = len(texts)
    logger.info("Stage 1: Loaded %d documents", n_documents)

    emb_path = os.path.join(run_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        logger.info("[cache] Loading embeddings from %s", emb_path)
        embeddings = np.load(emb_path)
    else:
        embeddings = compute_embeddings(
            texts, model_name=args.embedding_model, batch_size=args.batch_size,
        )
        np.save(emb_path, embeddings)
        logger.info("Stage 1: Saved embeddings shape=%s", embeddings.shape)

    # ── Stage 2: Dimensionality Reduction ──
    reduced_path = os.path.join(run_dir, "embeddings_reduced.npy")
    if os.path.exists(reduced_path):
        logger.info("[cache] Loading reduced embeddings from %s", reduced_path)
        embeddings_reduced = np.load(reduced_path)
    else:
        logger.info("Stage 2: Dimensionality reduction (%s) …", args.reduction)
        if args.reduction == "pca":
            embeddings_reduced = reduce_pca(
                embeddings,
                n_components=args.pca_dims,
                random_state=args.seed,
            )
        else:
            embeddings_reduced = reduce_tsne(
                embeddings,
                n_components=args.tsne_n_components,
                perplexity=args.tsne_perplexity,
                n_iter=args.tsne_n_iter,
                random_state=args.seed,
                metric=args.tsne_metric,
            )
        np.save(reduced_path, embeddings_reduced)
        logger.info("Stage 2: Reduced %s → %s", embeddings.shape, embeddings_reduced.shape)

    # ── Stage 3: Overclustering with K₀ ──
    meta_path = os.path.join(run_dir, "sealclust_metadata.json")

    if os.path.exists(meta_path):
        logger.info("[cache] Loading SEAL-Clust metadata from %s", meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        k0 = meta["k"]
        cluster_labels = np.array(meta["cluster_assignments"])
        medoid_indices = np.array(meta["medoid_indices"])
        logger.info("[cache] K₀=%d, %d medoids", k0, len(medoid_indices))
    else:
        k0 = min(args.k0, n_documents - 1)
        logger.info("Stage 3: Overclustering with K₀=%d (K-Medoids on %dD embeddings) …",
                    k0, embeddings_reduced.shape[1])
        cluster_labels, medoid_indices = run_sealclust_clustering(
            embeddings_reduced, k=k0, random_state=args.seed,
        )
        logger.info("Stage 3: %d micro-clusters, %d medoids", k0, len(medoid_indices))

    # ── Stage 4: Representative Selection ──
    logger.info("Stage 4: Extracting representative documents …")
    prototype_docs = get_prototypes(data_list, medoid_indices)
    _write_jsonl(os.path.join(run_dir, "medoid_documents.jsonl"), prototype_docs)

    # Cluster sizes
    cluster_map = build_cluster_map(cluster_labels, medoid_indices)
    cluster_sizes = {str(cid): len(members) for cid, members in sorted(cluster_map.items())}
    _write_json(os.path.join(run_dir, "cluster_sizes.json"), cluster_sizes)

    # ── Stage 5: Label Discovery (LLM on representatives ONLY) ──
    labels_proposed_path = os.path.join(run_dir, "labels_proposed.json")
    if os.path.exists(labels_proposed_path):
        logger.info("[cache] Loading proposed labels from %s", labels_proposed_path)
        with open(labels_proposed_path) as f:
            candidate_labels = json.load(f)
    else:
        from text_clustering.llm import ini_client
        client = ini_client()

        representative_texts = [doc["input"] for doc in prototype_docs]
        candidate_labels = discover_labels(
            representative_texts, client, chunk_size=args.label_chunk_size,
        )
        _write_json(labels_proposed_path, candidate_labels)

    # ── Stage 6: K* Estimation via GMM + BIC on representative embeddings ──
    # Use the REDUCED (PCA/t-SNE) embeddings of representatives for BIC.
    # Full 384D embeddings cause BIC to over-penalise and always pick smallest K.
    medoid_indices_sorted = sorted(int(i) for i in medoid_indices)
    representative_embeddings = embeddings_reduced[medoid_indices_sorted]

    if args.k_star:
        k_star = args.k_star
        logger.info("Stage 6: Using manual K*=%d (BIC skipped)", k_star)
        bic_scores = {}
    else:
        logger.info("Stage 6: Estimating K* via GMM + BIC on %d representative embeddings …",
                    representative_embeddings.shape[0])
        k_star, bic_scores = estimate_k_star_bic(
            representative_embeddings,
            k_min=args.bic_k_min,
            k_max=args.bic_k_max,
            random_state=args.seed,
        )
        _write_json(os.path.join(run_dir, "bic_scores.json"), {
            "k_star": k_star,
            "k_min": args.bic_k_min,
            "k_max": args.bic_k_max,
            "scores": {str(k): v for k, v in bic_scores.items()},
        })

    # ── Stage 7: Label Consolidation — merge to exactly K* ──
    labels_merged_path = os.path.join(run_dir, "labels_merged.json")
    if os.path.exists(labels_merged_path):
        logger.info("[cache] Loading merged labels from %s", labels_merged_path)
        with open(labels_merged_path) as f:
            final_labels = json.load(f)
    else:
        if 'client' not in dir():
            from text_clustering.llm import ini_client
            client = ini_client()

        final_labels = consolidate_labels(candidate_labels, k_star, client)
        _write_json(labels_merged_path, final_labels)

    # ── Save ground-truth labels ──
    from text_clustering.data import get_label_list
    true_labels = get_label_list(data_list)
    _write_json(os.path.join(run_dir, "labels_true.json"), true_labels)

    # ── Save metadata ──
    if not os.path.exists(meta_path):
        metadata = {
            "dataset": args.data,
            "split": size,
            "n_documents": n_documents,
            "pipeline": "sealclust_v2",
            "reduction": args.reduction,
            "reduction_dims": args.pca_dims if args.reduction == "pca" else args.tsne_n_components,
            "k": k0,
            "k0": k0,
            "k_star": k_star,
            "k_star_method": "manual" if args.k_star else "bic",
            "n_candidate_labels": len(candidate_labels),
            "n_final_labels": len(final_labels),
            "embedding_model": args.embedding_model,
            "random_state": args.seed,
            "n_medoids": len(medoid_indices),
            "medoid_indices": medoid_indices_sorted,
            "cluster_assignments": [int(c) for c in cluster_labels],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if args.reduction == "tsne":
            metadata.update({
                "tsne_n_components": args.tsne_n_components,
                "tsne_perplexity": args.tsne_perplexity,
                "tsne_metric": args.tsne_metric,
                "tsne_n_iter": args.tsne_n_iter,
            })
        _write_json(meta_path, metadata)
    else:
        # Update existing metadata with K* and label info
        with open(meta_path) as f:
            metadata = json.load(f)
        metadata.update({
            "pipeline": "sealclust_v2",
            "k_star": k_star,
            "k_star_method": "manual" if args.k_star else "bic",
            "n_candidate_labels": len(candidate_labels),
            "n_final_labels": len(final_labels),
        })
        _write_json(meta_path, metadata)

    elapsed = time.time() - start
    logger.info("=" * 70)
    logger.info("SEALClust Stages 1–7 complete in %.1fs", elapsed)
    logger.info("  Documents     : %d", n_documents)
    logger.info("  K₀ (overcl.)  : %d  (%.1f× compression)", k0, n_documents / max(k0, 1))
    logger.info("  K* (optimal)  : %d  (%s)", k_star, "manual" if args.k_star else "BIC")
    logger.info("  Candidate labs: %d → Final labs: %d", len(candidate_labels), len(final_labels))
    logger.info("  Reduction     : %s (%dD → %dD)", args.reduction,
                embeddings.shape[1], embeddings_reduced.shape[1])
    logger.info("  Run dir       : %s", run_dir)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  Stage 8: tc-classify --data %s --run_dir %s --medoid_mode", args.data, run_dir)
    logger.info("  Stage 9: tc-sealclust --data %s --run_dir %s --propagate", args.data, run_dir)
    logger.info("  Evaluate: tc-evaluate --data %s --run_dir %s", args.data, run_dir)
    logger.info("=" * 70)

    return run_dir


# ── propagate sub-command (Stage 9) ───────────────────────────────────────

def propagate(args) -> None:
    """Stage 9: Propagate representative labels to the full dataset.

    Reads:
      - ``sealclust_metadata.json``
      - ``classifications.json`` (from tc-classify --medoid_mode)
      - the original dataset

    Writes:
      - ``classifications_full.json``
    """
    run_dir = args.run_dir
    setup_logging(os.path.join(run_dir, "sealclust_propagate.log"))

    logger.info("=" * 70)
    logger.info("SEALClust Stage 9 — Label Propagation")
    logger.info("=" * 70)
    logger.info("Run dir: %s", run_dir)

    # Load metadata
    meta_path = os.path.join(run_dir, "sealclust_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    cluster_assignments = np.array(meta["cluster_assignments"])
    medoid_indices_sorted = sorted(meta["medoid_indices"])
    n_documents = meta["n_documents"]

    # Load dataset
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    assert len(data_list) == n_documents, (
        f"Dataset size mismatch: expected {n_documents}, got {len(data_list)}"
    )

    # Load prototype-level classifications
    class_path = os.path.join(run_dir, "classifications.json")
    with open(class_path) as f:
        classifications = json.load(f)
    logger.info("Loaded prototype classifications from %s", class_path)

    # Build medoid_index → label
    medoid_docs_text = [data_list[i]["input"] for i in medoid_indices_sorted]
    medoid_text_to_idx = {
        text: idx for idx, text in zip(medoid_indices_sorted, medoid_docs_text)
    }

    medoid_labels: dict[int, str] = {}
    for label, sentences in classifications.items():
        for sentence in sentences:
            if sentence in medoid_text_to_idx:
                med_idx = medoid_text_to_idx[sentence]
                medoid_labels[med_idx] = label

    logger.info(
        "Resolved labels for %d / %d representatives",
        len(medoid_labels), len(medoid_indices_sorted),
    )

    # Propagate
    all_labels = propagate_labels(medoid_labels, cluster_assignments, n_documents)

    # Build output
    full_classifications: dict[str, list[str]] = {}
    for doc_idx, label in enumerate(all_labels):
        full_classifications.setdefault(label, []).append(data_list[doc_idx]["input"])
    full_classifications = {k_: v for k_, v in full_classifications.items() if v}

    out_path = os.path.join(run_dir, "classifications_full.json")
    _write_json(out_path, full_classifications)

    for label, members in sorted(full_classifications.items(), key=lambda x: -len(x[1])):
        logger.info("  %-40s %d documents", label, len(members))

    total = sum(len(v) for v in full_classifications.values())
    unsuccessful = len(full_classifications.get("Unsuccessful", []))
    logger.info("Propagation complete — %d / %d documents labelled", total, n_documents)
    if unsuccessful:
        logger.warning("  %d documents labelled 'Unsuccessful'", unsuccessful)


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SEALClust — Scalable Efficient Autonomous LLM Clustering.\n\n"
            "9-Stage pipeline:\n"
            "  1. Embed  2. DimReduce  3. Overcluster  4. Representatives\n"
            "  5. Label Discovery  6. BIC K*  7. Consolidate\n"
            "  8. Classify reps (separate: tc-classify --medoid_mode)\n"
            "  9. Propagate (--propagate flag)\n\n"
            "Default: runs Stages 1–7. Use --propagate for Stage 9."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Dataset
    parser.add_argument("--data_path", type=str, default="./datasets/")
    parser.add_argument("--data", type=str, default="massive_scenario",
                        help="Dataset name (subfolder under data_path)")
    parser.add_argument("--runs_dir", type=str, default="./runs")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Existing run directory (for --propagate or to reuse cache)")
    parser.add_argument("--use_large", action="store_true")

    # Embedding
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--batch_size", type=int, default=64)

    # Dimensionality reduction
    parser.add_argument("--reduction", type=str, default=SEALCLUST_REDUCTION,
                        choices=["pca", "tsne"],
                        help="Reduction method: pca (default, recommended) or tsne")
    parser.add_argument("--pca_dims", type=int, default=SEALCLUST_PCA_DIMS,
                        help="PCA output dimensions (default: 50)")

    # t-SNE (when --reduction=tsne)
    parser.add_argument("--tsne_n_components", type=int, default=TSNE_N_COMPONENTS)
    parser.add_argument("--tsne_perplexity", type=float, default=TSNE_PERPLEXITY)
    parser.add_argument("--tsne_n_iter", type=int, default=TSNE_N_ITER)
    parser.add_argument("--tsne_metric", type=str, default=TSNE_METRIC)

    # Overclustering (K₀)
    parser.add_argument("--k0", type=int, default=SEALCLUST_K0,
                        help="Overclustering size K₀ (default: 300)")

    # K* estimation
    parser.add_argument("--k_star", type=int, default=SEALCLUST_K if SEALCLUST_K > 0 else 0,
                        help="Manual K* override. 0 = auto via BIC (default).")
    parser.add_argument("--bic_k_min", type=int, default=SEALCLUST_BIC_K_MIN,
                        help="Min K for BIC search (default: 5)")
    parser.add_argument("--bic_k_max", type=int, default=SEALCLUST_BIC_K_MAX,
                        help="Max K for BIC search (default: 50)")

    # Label discovery
    parser.add_argument("--label_chunk_size", type=int, default=SEALCLUST_LABEL_CHUNK_SIZE,
                        help="Representatives per LLM call for label discovery (default: 30)")

    # General
    parser.add_argument("--seed", type=int, default=42)

    # Propagation (Stage 9)
    parser.add_argument("--propagate", action="store_true",
                        help="Run Stage 9 (label propagation) — requires --run_dir")

    # Legacy compatibility
    parser.add_argument("--sealclust_k", type=int, default=0,
                        help="Legacy: same as --k_star (backward compatibility)")
    parser.add_argument("--elbow_k_min", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--elbow_k_max", type=int, default=200, help=argparse.SUPPRESS)
    parser.add_argument("--elbow_step", type=int, default=5, help=argparse.SUPPRESS)

    return parser


def main(args) -> None:
    # Legacy compatibility: --sealclust_k maps to --k_star
    if args.sealclust_k > 0 and not args.k_star:
        args.k_star = args.sealclust_k

    if args.propagate:
        if not args.run_dir:
            raise SystemExit("--run_dir is required when --propagate is set")
        propagate(args)
    else:
        run_pipeline(args)


def main_cli() -> None:
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())


if __name__ == "__main__":
    main_cli()
