"""
baseline_pipeline.py — Non-LLM baseline clustering pipelines CLI.

Provides KMeans and GMM baseline pipelines that use only embeddings for
clustering, with no LLM involvement.  Used for benchmarking against the
hybrid and SEAL-Clust pipelines.

Pipelines
---------
**KMeans baseline** — Embedding → L2-normalise → KMeans → evaluate::

    tc-baseline --method kmeans --data massive_scenario --k 18

**GMM baseline** — Embedding → L2-normalise → GMM → evaluate::

    tc-baseline --method gmm --data massive_scenario --k 18

**Auto-K** — let the algorithm choose the best K::

    tc-baseline --method kmeans --data massive_scenario --auto_k
    tc-baseline --method gmm --data massive_scenario --auto_k

**With PCA** — reduce dimensionality before clustering::

    tc-baseline --method kmeans --data massive_scenario --k 18 --pca_dims 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np

from text_clustering.config import EMBEDDING_MODEL
from text_clustering.data import get_label_list, load_dataset
from text_clustering.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_run_dir(runs_dir: str, data: str, size: str, method: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"{data}_{size}_baseline_{method}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Wrote %s", path)


# ── KMeans Baseline Pipeline ──────────────────────────────────────────────

def run_kmeans_pipeline(args) -> None:
    """Full KMeans baseline: embed → (PCA) → KMeans → evaluate."""
    from text_clustering.baselines import (
        auto_select_k_kmeans,
        run_kmeans_baseline,
    )
    from text_clustering.embedding import compute_embeddings

    size = "large" if args.use_large else "small"

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size, "kmeans")

    setup_logging(os.path.join(run_dir, "baseline_kmeans.log"))

    logger.info("=" * 70)
    logger.info("BASELINE — KMeans (No LLM)")
    logger.info("=" * 70)
    logger.info("Dataset       : %s  |  split: %s", args.data, size)
    logger.info("K             : %s", args.k if args.k else "auto")
    if args.pca_dims:
        logger.info("PCA dims      : %d", args.pca_dims)
    logger.info("Run dir       : %s", run_dir)
    start = time.time()

    # Load dataset
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    texts = [item["input"] for item in data_list]
    n_documents = len(texts)

    # Save ground-truth labels
    true_labels = get_label_list(data_list)
    _write_json(os.path.join(run_dir, "labels_true.json"), true_labels)
    logger.info("Ground-truth K = %d", len(true_labels))

    # Compute embeddings (or load from cache)
    emb_path = os.path.join(run_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        logger.info("[cache] Loading embeddings from %s", emb_path)
        embeddings = np.load(emb_path)
    else:
        embeddings = compute_embeddings(
            texts, model_name=args.embedding_model,
            batch_size=args.batch_size,
        )
        np.save(emb_path, embeddings)

    # Optional PCA
    if args.pca_dims:
        from text_clustering.dimreduce import reduce_pca
        reduced_path = os.path.join(run_dir, "embeddings_reduced.npy")
        if os.path.exists(reduced_path):
            logger.info("[cache] Loading reduced embeddings from %s", reduced_path)
            embeddings = np.load(reduced_path)
        else:
            embeddings = reduce_pca(
                embeddings, n_components=args.pca_dims,
                random_state=args.seed,
            )
            np.save(reduced_path, embeddings)

    # Determine K
    if args.auto_k or not args.k:
        best_k, scores = auto_select_k_kmeans(
            embeddings,
            k_min=args.k_min, k_max=args.k_max,
            random_state=args.seed,
        )
        k = best_k
        _write_json(os.path.join(run_dir, "k_selection.json"), {
            "method": "kmeans_silhouette",
            "best_k": k,
            "scores": {str(kk): v for kk, v in scores.items()},
        })
    else:
        k = args.k

    # Run KMeans
    labels, inertia, sil = run_kmeans_baseline(
        embeddings, k=k, random_state=args.seed,
    )

    # Build cluster-to-documents mapping using cluster IDs as label names
    classifications: dict[str, list[str]] = {}
    for doc_idx, cluster_id in enumerate(labels):
        label = f"Cluster_{int(cluster_id)}"
        classifications.setdefault(label, []).append(data_list[doc_idx]["input"])

    _write_json(os.path.join(run_dir, "classifications_full.json"), classifications)

    # Build a simple labels_merged.json for evaluation compatibility
    merged_labels = [f"Cluster_{i}" for i in range(k)]
    _write_json(os.path.join(run_dir, "labels_merged.json"), merged_labels)

    # Save metadata
    _write_json(os.path.join(run_dir, "baseline_metadata.json"), {
        "pipeline": "baseline_kmeans",
        "dataset": args.data,
        "split": size,
        "n_documents": n_documents,
        "k": k,
        "auto_k": args.auto_k or not args.k,
        "pca_dims": args.pca_dims,
        "inertia": inertia,
        "silhouette": sil,
        "embedding_model": args.embedding_model,
        "random_state": args.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    elapsed = time.time() - start
    logger.info("KMeans baseline complete in %.1fs", elapsed)
    logger.info("  K=%d, silhouette=%.4f, inertia=%.2f", k, sil, inertia)

    # Evaluate
    _run_evaluation(args, run_dir, data_list, labels)


# ── GMM Baseline Pipeline ─────────────────────────────────────────────────

def run_gmm_pipeline(args) -> None:
    """Full GMM baseline: embed → (PCA) → GMM → evaluate."""
    from text_clustering.baselines import (
        auto_select_k_gmm,
        run_gmm_baseline,
    )
    from text_clustering.embedding import compute_embeddings

    size = "large" if args.use_large else "small"

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size, "gmm")

    setup_logging(os.path.join(run_dir, "baseline_gmm.log"))

    logger.info("=" * 70)
    logger.info("BASELINE — GMM (No LLM)")
    logger.info("=" * 70)
    logger.info("Dataset       : %s  |  split: %s", args.data, size)
    logger.info("K             : %s", args.k if args.k else "auto")
    logger.info("Covariance    : %s", args.covariance_type)
    if args.pca_dims:
        logger.info("PCA dims      : %d", args.pca_dims)
    logger.info("Run dir       : %s", run_dir)
    start = time.time()

    # Load dataset
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    texts = [item["input"] for item in data_list]
    n_documents = len(texts)

    # Save ground-truth labels
    true_labels = get_label_list(data_list)
    _write_json(os.path.join(run_dir, "labels_true.json"), true_labels)
    logger.info("Ground-truth K = %d", len(true_labels))

    # Compute embeddings
    emb_path = os.path.join(run_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        logger.info("[cache] Loading embeddings from %s", emb_path)
        embeddings = np.load(emb_path)
    else:
        embeddings = compute_embeddings(
            texts, model_name=args.embedding_model,
            batch_size=args.batch_size,
        )
        np.save(emb_path, embeddings)

    # Optional PCA
    if args.pca_dims:
        from text_clustering.dimreduce import reduce_pca
        reduced_path = os.path.join(run_dir, "embeddings_reduced.npy")
        if os.path.exists(reduced_path):
            logger.info("[cache] Loading reduced embeddings from %s", reduced_path)
            embeddings = np.load(reduced_path)
        else:
            embeddings = reduce_pca(
                embeddings, n_components=args.pca_dims,
                random_state=args.seed,
            )
            np.save(reduced_path, embeddings)

    # Determine K
    if args.auto_k or not args.k:
        best_k, scores = auto_select_k_gmm(
            embeddings,
            k_min=args.k_min, k_max=args.k_max,
            covariance_type=args.covariance_type,
            random_state=args.seed,
        )
        k = best_k
        _write_json(os.path.join(run_dir, "k_selection.json"), {
            "method": "gmm_bic",
            "best_k": k,
            "scores": {str(kk): v for kk, v in scores.items()},
        })
    else:
        k = args.k

    # Run GMM
    labels, bic, sil = run_gmm_baseline(
        embeddings, k=k,
        covariance_type=args.covariance_type,
        random_state=args.seed,
    )

    # Build cluster-to-documents mapping
    classifications: dict[str, list[str]] = {}
    for doc_idx, cluster_id in enumerate(labels):
        label = f"Cluster_{int(cluster_id)}"
        classifications.setdefault(label, []).append(data_list[doc_idx]["input"])

    _write_json(os.path.join(run_dir, "classifications_full.json"), classifications)

    merged_labels = [f"Cluster_{i}" for i in range(k)]
    _write_json(os.path.join(run_dir, "labels_merged.json"), merged_labels)

    # Save metadata
    _write_json(os.path.join(run_dir, "baseline_metadata.json"), {
        "pipeline": "baseline_gmm",
        "dataset": args.data,
        "split": size,
        "n_documents": n_documents,
        "k": k,
        "auto_k": args.auto_k or not args.k,
        "pca_dims": args.pca_dims,
        "bic": bic,
        "silhouette": sil,
        "covariance_type": args.covariance_type,
        "embedding_model": args.embedding_model,
        "random_state": args.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    elapsed = time.time() - start
    logger.info("GMM baseline complete in %.1fs", elapsed)
    logger.info("  K=%d, BIC=%.2f, silhouette=%.4f", k, bic, sil)

    # Evaluate
    _run_evaluation(args, run_dir, data_list, labels)


# ── Shared Evaluation ──────────────────────────────────────────────────────

def _run_evaluation(
    args, run_dir: str, data_list: list[dict], labels: np.ndarray,
) -> None:
    """Compute ACC/NMI/ARI for baseline pipelines.

    Since baselines produce numeric cluster IDs (not semantic labels), we
    use the standard Hungarian matching evaluation from the evaluation module.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    from text_clustering.config import MODEL

    # Ground-truth labels → integer IDs
    true_label_strs = [item["label"] for item in data_list]
    unique_true = list(set(true_label_strs))
    true_map = {lbl: i for i, lbl in enumerate(unique_true)}
    y_true = np.array([true_map[lbl] for lbl in true_label_strs])

    # Predicted labels
    y_pred = np.array(labels, dtype=int)

    # Hungarian matching → ACC
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    acc = float(sum(w[i, j] for i, j in ind) / y_pred.size)

    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    ari = float(adjusted_rand_score(y_true, y_pred))

    logger.info("Results:")
    logger.info("  ACC : %.4f", acc)
    logger.info("  NMI : %.4f", nmi)
    logger.info("  ARI : %.4f", ari)

    size = "large" if args.use_large else "small"
    results = {
        "dataset": args.data,
        "split": size,
        "model": MODEL,
        "pipeline": f"baseline_{args.method}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(data_list),
        "n_clusters_true": len(unique_true),
        "n_clusters_pred": int(len(set(labels))),
        "ACC": round(acc, 6),
        "NMI": round(nmi, 6),
        "ARI": round(ari, 6),
    }

    results_path = os.path.join(run_dir, "results.json")
    _write_json(results_path, results)


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline clustering pipelines (no LLM).\n\n"
            "Supported methods:\n"
            "  kmeans — Embedding + KMeans\n"
            "  gmm    — Embedding + GMM\n\n"
            "Used for benchmarking against LLM-based pipelines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Dataset
    parser.add_argument("--data_path", type=str, default="./datasets/")
    parser.add_argument("--data", type=str, default="massive_scenario",
                        help="Dataset name")
    parser.add_argument("--runs_dir", type=str, default="./runs")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Existing run directory (for cache reuse)")
    parser.add_argument("--use_large", action="store_true")

    # Method
    parser.add_argument("--method", type=str, default="kmeans",
                        choices=["kmeans", "gmm"],
                        help="Clustering method: kmeans (default) or gmm")

    # Clustering parameters
    parser.add_argument("--k", type=int, default=0,
                        help="Number of clusters (0 = auto-select)")
    parser.add_argument("--auto_k", action="store_true",
                        help="Automatically select optimal K")
    parser.add_argument("--k_min", type=int, default=2,
                        help="Min K for auto-selection (default: 2)")
    parser.add_argument("--k_max", type=int, default=50,
                        help="Max K for auto-selection (default: 50)")

    # GMM-specific
    parser.add_argument("--covariance_type", type=str, default="tied",
                        choices=["full", "tied", "diag", "spherical"],
                        help="GMM covariance type")

    # PCA
    parser.add_argument("--pca_dims", type=int, default=0,
                        help="PCA dimensions (0 = no PCA)")

    # Embedding
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--batch_size", type=int, default=64)

    # General
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main(args) -> None:
    if args.method == "kmeans":
        run_kmeans_pipeline(args)
    elif args.method == "gmm":
        run_gmm_pipeline(args)
    else:
        raise SystemExit(f"Unknown method: {args.method}")


def main_cli() -> None:
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())


if __name__ == "__main__":
    main_cli()
