"""
gmm_preprocessing.py — GMM pre-clustering step for the pipeline.

Provides the same workflow as ``kmedoids_preprocessing.py`` but uses a
Gaussian Mixture Model instead of K-Medoids:

  1. Load the full dataset.
  2. Compute (or reuse) sentence embeddings.
  3. Fit a GMM — optionally auto-select k via BIC over a range.
  4. Extract one *representative* document per cluster (closest to mean).
  5. Save artefacts to a timestamped run directory.

After this step the existing LLM pipeline (label generation → classification)
runs on ``representative_documents.jsonl``.  Once classification is done, a
**propagation** post-step maps labels back to every document using hard
(or soft) cluster assignments.

Embedding reuse
---------------
  If an existing ``--run_dir`` already contains ``embeddings.npy``, the
  embedding step is skipped — this lets you switch between K-Medoids and GMM
  without recomputing embeddings.

Usage
-----
  # Pre-cluster
  tc-gmm --data massive_scenario --gmm_k 100

  # … run tc-label-gen + tc-classify --representative_mode …

  # Propagate
  tc-gmm --data massive_scenario --run_dir ./runs/<run> --propagate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np

from text_clustering.config import EMBEDDING_MODEL, GMM_COVARIANCE_TYPE, GMM_K
from text_clustering.data import load_dataset
from text_clustering.embedding import compute_embeddings
from text_clustering.gmm import (
    auto_select_k,
    build_cluster_map,
    get_representative_documents,
    propagate_labels,
    propagate_labels_soft,
    run_gmm,
)
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


# ── pre-cluster sub-command ────────────────────────────────────────────────

def precluster(args) -> str:
    """Run embedding + GMM and write artefacts.

    Checkpoint support:
      - ``embeddings.npy`` — skip embedding recomputation
      - ``gmm_metadata.json`` — skip GMM refitting

    Returns the run directory path.
    """
    size = "large" if args.use_large else "small"

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size)

    setup_logging(os.path.join(run_dir, "gmm_precluster.log"))

    logger.info("=== GMM Pre-Clustering ===")
    logger.info("Dataset : %s  |  split: %s", args.data, size)
    logger.info(
        "GMM k   : %s",
        args.gmm_k if args.gmm_k else f"auto [{args.gmm_k_min}-{args.gmm_k_max}]",
    )
    logger.info("Cov type: %s", args.covariance_type)
    logger.info("Embed   : %s", args.embedding_model)
    logger.info("Run dir : %s", run_dir)
    start = time.time()

    # 1. Load dataset
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    texts = [item["input"] for item in data_list]
    logger.info("Loaded %d documents", len(texts))

    # 2. Embeddings (reuse if cached)
    emb_path = os.path.join(run_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        logger.info("[checkpoint] Loading cached embeddings from %s", emb_path)
        embeddings = np.load(emb_path)
        logger.info("[checkpoint] shape: %s", embeddings.shape)
    else:
        embeddings = compute_embeddings(
            texts, model_name=args.embedding_model, batch_size=args.batch_size,
        )
        np.save(emb_path, embeddings)
        logger.info("Saved embeddings to %s", emb_path)

    # 3. GMM (or load checkpoint)
    meta_path = os.path.join(run_dir, "gmm_metadata.json")
    if os.path.exists(meta_path):
        logger.info("[checkpoint] Loading cached GMM metadata from %s", meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        labels = np.array(meta["cluster_assignments"])
        k = meta["gmm_k"]
        # Rebuild means from metadata (stored as list-of-lists)
        means = np.array(meta["component_means"]) if "component_means" in meta else None
        probs = None  # will be recomputed if needed during propagation
        representative_indices = np.array(meta["representative_indices"])
        logger.info(
            "[checkpoint] Loaded GMM: k=%d, %d representatives",
            k, len(representative_indices),
        )
    else:
        # Auto-select k or use fixed
        if args.gmm_k:
            k = args.gmm_k
        else:
            k, scores = auto_select_k(
                embeddings,
                k_range=(args.gmm_k_min, args.gmm_k_max),
                criterion=args.selection_criterion,
                covariance_type=args.covariance_type,
                random_state=args.seed,
            )
            _write_json(os.path.join(run_dir, "gmm_k_selection.json"), {
                "best_k": k,
                "criterion": args.selection_criterion,
                "scores": {str(kk): v for kk, v in scores.items()},
            })

        labels, probs, means = run_gmm(
            embeddings, k=k,
            covariance_type=args.covariance_type,
            random_state=args.seed,
        )

        # 4. Extract representatives
        rep_docs, representative_indices = get_representative_documents(
            data_list, embeddings, means, labels, k,
        )

        # Save metadata
        metadata = {
            "dataset": args.data,
            "split": size,
            "n_documents": len(data_list),
            "clustering_method": "gmm",
            "gmm_k": k,
            "covariance_type": args.covariance_type,
            "embedding_model": args.embedding_model,
            "random_state": args.seed,
            "n_representatives": len(representative_indices),
            "representative_indices": [int(i) for i in representative_indices],
            "cluster_assignments": [int(c) for c in labels],
            "component_means": means.tolist(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(meta_path, metadata)

    # 5. Write representative documents
    rep_docs_from_idx = [data_list[int(i)] for i in sorted(representative_indices)]
    _write_jsonl(os.path.join(run_dir, "representative_documents.jsonl"), rep_docs_from_idx)

    # Cluster sizes
    cluster_map = build_cluster_map(labels)
    cluster_sizes = {str(cid): len(members) for cid, members in sorted(cluster_map.items())}
    _write_json(os.path.join(run_dir, "cluster_sizes.json"), cluster_sizes)

    # Save posterior probabilities for soft propagation later
    if probs is not None:
        np.save(os.path.join(run_dir, "gmm_probs.npy"), probs)
        logger.info("Saved posterior probabilities to gmm_probs.npy")

    elapsed = time.time() - start
    logger.info("Pre-clustering complete in %.1fs", elapsed)
    logger.info("  %d documents → %d representatives (%.1fx reduction)",
                len(data_list), len(representative_indices),
                len(data_list) / max(len(representative_indices), 1))
    logger.info("  Run dir: %s", run_dir)
    logger.info(
        "  Next: run tc-label-gen --run_dir %s  then  "
        "tc-classify --run_dir %s --representative_mode",
        run_dir, run_dir,
    )

    return run_dir


# ── propagate sub-command ──────────────────────────────────────────────────

def propagate(args) -> None:
    """Propagate representative-level classifications to the full dataset.

    Reads:
      - ``gmm_metadata.json``
      - ``classifications.json``
      - the original dataset

    Writes:
      - ``classifications_full.json``
    """
    run_dir = args.run_dir
    setup_logging(os.path.join(run_dir, "gmm_propagate.log"))

    logger.info("=== GMM Label Propagation ===")
    logger.info("Run dir : %s", run_dir)

    meta_path = os.path.join(run_dir, "gmm_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    cluster_assignments = np.array(meta["cluster_assignments"])
    representative_indices = sorted(meta["representative_indices"])
    n_documents = meta["n_documents"]

    data_list = load_dataset(args.data_path, args.data, args.use_large)
    assert len(data_list) == n_documents, (
        f"Dataset size mismatch: expected {n_documents}, got {len(data_list)}"
    )

    # Load classifications
    class_path = os.path.join(run_dir, "classifications.json")
    with open(class_path) as f:
        classifications = json.load(f)
    logger.info("Loaded representative classifications from %s", class_path)

    # Build representative_index → label
    rep_docs_text = [data_list[i]["input"] for i in representative_indices]
    rep_text_to_idx = {text: idx for idx, text in zip(representative_indices, rep_docs_text)}

    representative_labels: dict[int, str] = {}
    for label, sentences in classifications.items():
        for sentence in sentences:
            if sentence in rep_text_to_idx:
                rep_idx = rep_text_to_idx[sentence]
                representative_labels[rep_idx] = label

    logger.info("Resolved labels for %d / %d representatives",
                len(representative_labels), len(representative_indices))

    # Choose propagation strategy
    probs_path = os.path.join(run_dir, "gmm_probs.npy")
    use_soft = getattr(args, "soft", False) and os.path.exists(probs_path)

    if use_soft:
        probs = np.load(probs_path)
        threshold = getattr(args, "confidence_threshold", 0.4)
        logger.info("Using SOFT propagation (threshold=%.2f)", threshold)
        all_labels = propagate_labels_soft(
            representative_labels, cluster_assignments, probs, n_documents, threshold,
        )
    else:
        logger.info("Using HARD propagation")
        all_labels = propagate_labels(
            representative_labels, cluster_assignments, n_documents,
        )

    # Build output
    full_classifications: dict[str, list[str]] = {}
    for doc_idx, label in enumerate(all_labels):
        full_classifications.setdefault(label, []).append(data_list[doc_idx]["input"])
    full_classifications = {k: v for k, v in full_classifications.items() if v}

    out_path = os.path.join(run_dir, "classifications_full.json")
    _write_json(out_path, full_classifications)

    for label, members in sorted(full_classifications.items(), key=lambda x: -len(x[1])):
        logger.info("  %-40s %d documents", label, len(members))

    total = sum(len(v) for v in full_classifications.values())
    logger.info("Propagation complete — %d / %d documents labelled", total, n_documents)


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GMM pre-clustering: compress the dataset to representative documents "
            "before running the LLM pipeline, then propagate labels back."
        ),
    )
    parser.add_argument("--data_path", type=str, default="./datasets/")
    parser.add_argument("--data", type=str, default="massive_scenario",
                        help="Dataset name (subfolder under data_path)")
    parser.add_argument("--runs_dir", type=str, default="./runs")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Existing run directory (for --propagate or to reuse embeddings)")
    parser.add_argument("--use_large", action="store_true")

    # GMM parameters
    parser.add_argument("--gmm_k", type=int, default=GMM_K,
                        help="Number of components. Set to 0 for auto-selection.")
    parser.add_argument("--gmm_k_min", type=int, default=10,
                        help="Min k for auto-selection range")
    parser.add_argument("--gmm_k_max", type=int, default=200,
                        help="Max k for auto-selection range")
    parser.add_argument("--selection_criterion", type=str, default="bic",
                        choices=["bic", "silhouette"],
                        help="Criterion for auto k selection")
    parser.add_argument("--covariance_type", type=str, default=GMM_COVARIANCE_TYPE,
                        choices=["full", "tied", "diag", "spherical"])
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # Propagation
    parser.add_argument("--propagate", action="store_true")
    parser.add_argument("--soft", action="store_true",
                        help="Use soft (probability-weighted) propagation instead of hard")
    parser.add_argument("--confidence_threshold", type=float, default=0.4,
                        help="Min posterior probability for soft propagation "
                             "(below → Unsuccessful)")

    return parser


def main(args) -> None:
    if args.propagate:
        if not args.run_dir:
            raise SystemExit("--run_dir is required when --propagate is set")
        propagate(args)
    else:
        precluster(args)


def main_cli() -> None:
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())


if __name__ == "__main__":
    main_cli()
