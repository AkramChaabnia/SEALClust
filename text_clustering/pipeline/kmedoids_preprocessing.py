"""
kmedoids_preprocessing.py — K-Medoids pre-clustering step for the pipeline.

This step sits **before** the existing LLM pipeline (Steps 0–3) and reduces
the number of documents that need to be processed by the LLM.

Workflow
--------
1. Load the full dataset.
2. Compute sentence embeddings for every document.
3. Run K-Medoids to select *k* representative medoid documents.
4. Save the clustering artefacts to the run directory:
     - ``kmedoids_metadata.json``   cluster assignments, medoid indices, config
     - ``medoid_documents.jsonl``    the medoid subset in the original JSONL format

After this step, the existing LLM pipeline (label generation → classification)
runs on ``medoid_documents.jsonl`` instead of the full dataset.  Once
classification is complete, a **label-propagation** post-step maps medoid
labels back to every document (see ``propagate`` sub-command).

Configuration
-------------
  KMEDOIDS_ENABLED=true      enable the pre-clustering step
  KMEDOIDS_K=100             number of medoids / clusters
  EMBEDDING_MODEL=all-MiniLM-L6-v2   sentence-transformers model

Usage
-----
  # Pre-cluster (creates run dir with medoid artefacts)
  tc-kmedoids --data massive_scenario

  # … run tc-label-gen / tc-classify on the medoid subset …

  # Propagate labels back to full dataset
  tc-kmedoids --data massive_scenario --run_dir ./runs/<run> --propagate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np

from text_clustering.config import EMBEDDING_MODEL, KMEDOIDS_K
from text_clustering.data import load_dataset
from text_clustering.embedding import compute_embeddings
from text_clustering.kmedoids import (
    build_cluster_map,
    get_medoid_documents,
    propagate_labels,
    run_kmedoids,
)
from text_clustering.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _make_run_dir(runs_dir: str, data: str, size: str) -> str:
    """Create and return a timestamped run directory."""
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
    """Run embedding + K-Medoids and write artefacts to a new run directory.

    If ``--run_dir`` is provided and already contains ``embeddings.npy``, the
    embedding step is skipped (checkpoint/resume).  Similarly, if
    ``kmedoids_metadata.json`` already exists the clustering step is skipped.

    Returns the path to the created (or reused) run directory.
    """
    size = "large" if args.use_large else "small"

    # Reuse existing run_dir if provided, otherwise create a new one
    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size)

    setup_logging(os.path.join(run_dir, "kmedoids_precluster.log"))

    logger.info("=== K-Medoids Pre-Clustering ===")
    logger.info("Dataset : %s  |  split: %s", args.data, size)
    logger.info("k       : %d", args.kmedoids_k)
    logger.info("Embed   : %s", args.embedding_model)
    logger.info("Run dir : %s", run_dir)
    start = time.time()

    # 1. Load dataset
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    texts = [item["input"] for item in data_list]
    logger.info("Loaded %d documents", len(texts))

    # 2. Compute embeddings (or load from checkpoint)
    emb_path = os.path.join(run_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        logger.info("[checkpoint] Loading cached embeddings from %s", emb_path)
        embeddings = np.load(emb_path)
        logger.info("[checkpoint] Loaded embeddings — shape: %s", embeddings.shape)
    else:
        embeddings = compute_embeddings(
            texts,
            model_name=args.embedding_model,
            batch_size=args.batch_size,
        )
        np.save(emb_path, embeddings)
        logger.info("Saved embeddings to %s", emb_path)

    # 3. Run K-Medoids (or load from checkpoint)
    meta_path = os.path.join(run_dir, "kmedoids_metadata.json")
    if os.path.exists(meta_path):
        logger.info("[checkpoint] Loading cached K-Medoids metadata from %s", meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        cluster_labels = np.array(meta["cluster_assignments"])
        medoid_indices = np.array(meta["medoid_indices"])
        logger.info(
            "[checkpoint] Loaded %d clusters, %d medoids",
            args.kmedoids_k, len(medoid_indices),
        )
    else:
        cluster_labels, medoid_indices = run_kmedoids(
            embeddings,
            k=args.kmedoids_k,
            random_state=args.seed,
        )

        # Save metadata
        metadata = {
            "dataset": args.data,
            "split": size,
            "n_documents": len(data_list),
            "kmedoids_k": args.kmedoids_k,
            "embedding_model": args.embedding_model,
            "random_state": args.seed,
            "n_medoids": len(medoid_indices),
            "medoid_indices": sorted(int(i) for i in medoid_indices),
            "cluster_assignments": [int(c) for c in cluster_labels],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(meta_path, metadata)

    # 4. Extract medoid documents
    medoid_docs = get_medoid_documents(data_list, medoid_indices)
    _write_jsonl(os.path.join(run_dir, "medoid_documents.jsonl"), medoid_docs)

    # Save cluster map for inspection
    cluster_map = build_cluster_map(cluster_labels, medoid_indices)
    cluster_sizes = {str(k): len(v) for k, v in sorted(cluster_map.items())}
    _write_json(os.path.join(run_dir, "cluster_sizes.json"), cluster_sizes)

    elapsed = time.time() - start
    logger.info("Pre-clustering complete in %.1fs", elapsed)
    logger.info("  %d documents → %d medoids (%.1fx reduction)",
                len(data_list), len(medoid_indices),
                len(data_list) / max(len(medoid_indices), 1))
    logger.info("  Run dir: %s", run_dir)
    logger.info("  Next: run tc-label-gen and tc-classify on medoid_documents.jsonl")

    return run_dir


# ── propagate sub-command ──────────────────────────────────────────────────

def propagate(args) -> None:
    """Propagate medoid-level classifications to the full dataset.

    Reads:
      - ``kmedoids_metadata.json``  (cluster assignments, medoid indices)
      - ``classifications.json``    (Step 2 output on medoid docs)
      - the original dataset

    Writes:
      - ``classifications_full.json``   (label → [sentences] for ALL documents)
    """
    run_dir = args.run_dir
    setup_logging(os.path.join(run_dir, "kmedoids_propagate.log"))

    logger.info("=== K-Medoids Label Propagation ===")
    logger.info("Run dir : %s", run_dir)

    # Load metadata
    meta_path = os.path.join(run_dir, "kmedoids_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    cluster_assignments = np.array(meta["cluster_assignments"])
    medoid_indices_sorted = sorted(meta["medoid_indices"])
    n_documents = meta["n_documents"]

    # Load the full dataset
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    assert len(data_list) == n_documents, (
        f"Dataset size mismatch: expected {n_documents}, got {len(data_list)}"
    )

    # Load the medoid-level classifications produced by Step 2
    class_path = os.path.join(run_dir, "classifications.json")
    with open(class_path) as f:
        classifications = json.load(f)
    logger.info("Loaded medoid classifications from %s", class_path)

    # Build medoid_index → label mapping
    # classifications is { label: [sentence, …] }
    # We need to find which medoid each sentence belongs to.
    medoid_docs_text = [data_list[i]["input"] for i in medoid_indices_sorted]
    medoid_text_to_idx = {text: idx for idx, text in zip(medoid_indices_sorted, medoid_docs_text)}

    medoid_labels: dict[int, str] = {}
    for label, sentences in classifications.items():
        for sentence in sentences:
            if sentence in medoid_text_to_idx:
                med_idx = medoid_text_to_idx[sentence]
                medoid_labels[med_idx] = label

    labelled_medoids = len(medoid_labels)
    logger.info("Resolved labels for %d / %d medoids", labelled_medoids, len(medoid_indices_sorted))

    # Propagate
    all_labels = propagate_labels(medoid_labels, cluster_assignments, n_documents)

    # Build output in the same format as classifications.json (label → [sentences])
    full_classifications: dict[str, list[str]] = {}
    for doc_idx, label in enumerate(all_labels):
        full_classifications.setdefault(label, []).append(data_list[doc_idx]["input"])

    # Drop empty buckets
    full_classifications = {k: v for k, v in full_classifications.items() if v}

    out_path = os.path.join(run_dir, "classifications_full.json")
    _write_json(out_path, full_classifications)

    # Summary
    for label, members in sorted(full_classifications.items(), key=lambda x: -len(x[1])):
        logger.info("  %-40s %d documents", label, len(members))

    total = sum(len(v) for v in full_classifications.values())
    logger.info("Propagation complete — %d / %d documents labelled", total, n_documents)


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "K-Medoids pre-clustering: compress the dataset to representative medoids "
            "before running the LLM pipeline, then propagate labels back."
        ),
    )
    parser.add_argument("--data_path", type=str, default="./datasets/")
    parser.add_argument("--data", type=str, default="massive_scenario",
                        help="Dataset name (subfolder under data_path)")
    parser.add_argument("--runs_dir", type=str, default="./runs",
                        help="Root directory for timestamped run folders")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Existing run directory (required for --propagate)")
    parser.add_argument("--use_large", action="store_true",
                        help="Use the large split instead of small")
    parser.add_argument("--kmedoids_k", type=int, default=KMEDOIDS_K,
                        help="Number of clusters / medoids (default: KMEDOIDS_K env var)")
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL,
                        help="sentence-transformers model name")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding computation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for K-Medoids")
    parser.add_argument("--propagate", action="store_true",
                        help="Run label propagation (requires --run_dir)")
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
