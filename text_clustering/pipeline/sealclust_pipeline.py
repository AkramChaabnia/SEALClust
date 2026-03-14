"""
sealclust_pipeline.py — SEAL-Clust full 7-step pipeline CLI.

Implements the complete SEAL-Clust framework as a single entry point
(``tc-sealclust``) with two sub-commands:

  1. **precluster** (default) — Steps 1–4 + auto-k (Step 6):
       Embed → t-SNE → Elbow auto-k → K-Medoids → Prototype extraction
  2. **propagate** — Step 7:
       Map prototype labels back to every document.

The LLM steps (5: label generation + classification) are run with the
existing ``tc-label-gen`` and ``tc-classify --medoid_mode`` commands
between precluster and propagate.

Auto-k with manual override
----------------------------
By default, ``tc-sealclust`` runs the Elbow method to find the best k.
If you want to **skip auto-k** and set k yourself, pass ``--sealclust_k <N>``.
When ``--sealclust_k 0`` (or omitted), the Elbow method runs automatically.

Workflow
--------
::

    # Step 1–4, 6: Embed + t-SNE + auto-k + K-Medoids + prototypes
    tc-sealclust --data massive_scenario

    # (optional) manually specify k instead of auto-k
    tc-sealclust --data massive_scenario --sealclust_k 80

    # Step 5a: LLM label generation
    tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>

    # (optional) re-merge labels
    python tools/remerge_labels.py ./runs/<run_dir> 18

    # Step 5b: LLM classification on prototypes only
    tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode

    # Step 7: propagate labels to full dataset
    tc-sealclust --data massive_scenario --run_dir ./runs/<run_dir> --propagate

    # Evaluate
    tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
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
    SEALCLUST_K,
    SEALCLUST_ELBOW_K_MIN,
    SEALCLUST_ELBOW_K_MAX,
    SEALCLUST_ELBOW_STEP,
    TSNE_N_COMPONENTS,
    TSNE_PERPLEXITY,
    TSNE_N_ITER,
    TSNE_METRIC,
)
from text_clustering.data import load_dataset
from text_clustering.embedding import compute_embeddings
from text_clustering.dimreduce import reduce_tsne
from text_clustering.sealclust import (
    elbow_select_k,
    find_elbow,
    get_prototypes,
    propagate_labels,
    run_sealclust_clustering,
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


# ── pre-cluster sub-command (Steps 1–4, 6) ────────────────────────────────

def precluster(args) -> str:
    """Run the full SEAL-Clust pre-clustering pipeline.

    Steps executed:
      1. Embedding generation (or cache reuse)
      2. t-SNE dimensionality reduction (or cache reuse)
      3+6. Auto-k via Elbow method (or manual k override)
      3. K-Medoids microcluster formation
      4. Prototype extraction → medoid_documents.jsonl

    Returns the run directory path.
    """
    size = "large" if args.use_large else "small"

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size)

    setup_logging(os.path.join(run_dir, "sealclust_precluster.log"))

    logger.info("=" * 60)
    logger.info("SEAL-Clust Pre-Clustering Pipeline")
    logger.info("=" * 60)
    logger.info("Dataset       : %s  |  split: %s", args.data, size)
    logger.info("Embedding     : %s", args.embedding_model)
    logger.info("t-SNE         : n_components=%d, perplexity=%.1f, metric=%s",
                args.tsne_n_components, args.tsne_perplexity, args.tsne_metric)
    if args.sealclust_k:
        logger.info("k (manual)    : %d", args.sealclust_k)
    else:
        logger.info("k (auto)      : Elbow method [%d–%d] step=%d",
                    args.elbow_k_min, args.elbow_k_max, args.elbow_step)
    logger.info("Run dir       : %s", run_dir)
    start = time.time()

    # ── Step 1: Load dataset ──
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    texts = [item["input"] for item in data_list]
    n_documents = len(texts)
    logger.info("Step 1: Loaded %d documents", n_documents)

    # ── Step 1: Compute embeddings (or reuse cache) ──
    emb_path = os.path.join(run_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        logger.info("[cache] Loading embeddings from %s", emb_path)
        embeddings = np.load(emb_path)
    else:
        embeddings = compute_embeddings(
            texts, model_name=args.embedding_model, batch_size=args.batch_size,
        )
        np.save(emb_path, embeddings)
        logger.info("Saved embeddings to %s  shape=%s", emb_path, embeddings.shape)

    # ── Step 2: t-SNE dimensionality reduction (or reuse cache) ──
    reduced_path = os.path.join(run_dir, "embeddings_reduced.npy")
    if os.path.exists(reduced_path):
        logger.info("[cache] Loading reduced embeddings from %s", reduced_path)
        embeddings_reduced = np.load(reduced_path)
    else:
        logger.info("Step 2: Running t-SNE dimensionality reduction …")
        embeddings_reduced = reduce_tsne(
            embeddings,
            n_components=args.tsne_n_components,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_n_iter,
            random_state=args.seed,
            metric=args.tsne_metric,
        )
        np.save(reduced_path, embeddings_reduced)
        logger.info("Saved reduced embeddings to %s  shape=%s",
                    reduced_path, embeddings_reduced.shape)

    # ── Step 6 (before 3): Determine k ──
    meta_path = os.path.join(run_dir, "sealclust_metadata.json")

    if os.path.exists(meta_path):
        # Resume from checkpoint
        logger.info("[cache] Loading SEAL-Clust metadata from %s", meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        k = meta["k"]
        cluster_labels = np.array(meta["cluster_assignments"])
        medoid_indices = np.array(meta["medoid_indices"])
        logger.info("[cache] k=%d, %d medoids", k, len(medoid_indices))
    else:
        if args.sealclust_k:
            # ── Manual k ──
            k = args.sealclust_k
            logger.info("Step 6: Using manual k=%d (Elbow skipped)", k)
            elbow_scores = {}
        else:
            # ── Auto-k via Elbow ──
            logger.info("Step 6: Running Elbow method for auto-k …")
            k, elbow_scores = elbow_select_k(
                embeddings_reduced,
                k_range=(args.elbow_k_min, args.elbow_k_max),
                step=args.elbow_step,
                random_state=args.seed,
            )
            logger.info("Step 6: Elbow selected k=%d", k)

            # Save elbow scores for inspection / plotting
            _write_json(os.path.join(run_dir, "elbow_scores.json"), {
                "best_k": k,
                "k_min": args.elbow_k_min,
                "k_max": args.elbow_k_max,
                "step": args.elbow_step,
                "scores": {str(kk): v for kk, v in elbow_scores.items()},
            })

        # ── Step 3: K-Medoids microcluster formation ──
        logger.info("Step 3: Running K-Medoids with k=%d …", k)
        cluster_labels, medoid_indices = run_sealclust_clustering(
            embeddings_reduced, k=k, random_state=args.seed,
        )

        # ── Save metadata ──
        metadata = {
            "dataset": args.data,
            "split": size,
            "n_documents": n_documents,
            "pipeline": "sealclust",
            "k": k,
            "k_method": "manual" if args.sealclust_k else "elbow",
            "embedding_model": args.embedding_model,
            "tsne_n_components": args.tsne_n_components,
            "tsne_perplexity": args.tsne_perplexity,
            "tsne_metric": args.tsne_metric,
            "tsne_n_iter": args.tsne_n_iter,
            "random_state": args.seed,
            "n_medoids": len(medoid_indices),
            "medoid_indices": sorted(int(i) for i in medoid_indices),
            "cluster_assignments": [int(c) for c in cluster_labels],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(meta_path, metadata)

    # ── Step 4: Prototype extraction ──
    logger.info("Step 4: Extracting prototypes …")
    prototype_docs = get_prototypes(data_list, medoid_indices)
    _write_jsonl(os.path.join(run_dir, "medoid_documents.jsonl"), prototype_docs)

    # Cluster sizes
    cluster_map = build_cluster_map(cluster_labels, medoid_indices)
    cluster_sizes = {str(cid): len(members) for cid, members in sorted(cluster_map.items())}
    _write_json(os.path.join(run_dir, "cluster_sizes.json"), cluster_sizes)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("SEAL-Clust pre-clustering complete in %.1fs", elapsed)
    logger.info("  %d documents → %d prototypes (%.1fx reduction)",
                n_documents, len(medoid_indices),
                n_documents / max(len(medoid_indices), 1))
    logger.info("  k=%d (%s)", k, "manual" if args.sealclust_k else "elbow auto")
    logger.info("  Run dir: %s", run_dir)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. tc-label-gen --data %s --run_dir %s", args.data, run_dir)
    logger.info("  2. tc-classify --data %s --run_dir %s --medoid_mode", args.data, run_dir)
    logger.info("  3. tc-sealclust --data %s --run_dir %s --propagate", args.data, run_dir)
    logger.info("  4. tc-evaluate --data %s --run_dir %s", args.data, run_dir)
    logger.info("=" * 60)

    return run_dir


# ── propagate sub-command (Step 7) ─────────────────────────────────────────

def propagate(args) -> None:
    """Step 7: Propagate prototype labels to the full dataset.

    Reads:
      - ``sealclust_metadata.json``
      - ``classifications.json`` (from tc-classify --medoid_mode)
      - the original dataset

    Writes:
      - ``classifications_full.json``
    """
    run_dir = args.run_dir
    setup_logging(os.path.join(run_dir, "sealclust_propagate.log"))

    logger.info("=" * 60)
    logger.info("SEAL-Clust Label Propagation (Step 7)")
    logger.info("=" * 60)
    logger.info("Run dir: %s", run_dir)

    # Load metadata
    meta_path = os.path.join(run_dir, "sealclust_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    cluster_assignments = np.array(meta["cluster_assignments"])
    medoid_indices_sorted = sorted(meta["medoid_indices"])
    n_documents = meta["n_documents"]
    k = meta["k"]

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
        "Resolved labels for %d / %d prototypes",
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
    unlabelled = total - sum(
        len(v) for k_, v in full_classifications.items() if k_ != "Unsuccessful"
    )
    logger.info("Propagation complete — %d / %d documents labelled", total, n_documents)
    if "Unsuccessful" in full_classifications:
        logger.warning(
            "  %d documents labelled 'Unsuccessful'",
            len(full_classifications["Unsuccessful"]),
        )


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SEAL-Clust: Scalable Efficient Autonomous LLM Clustering.\n\n"
            "Full 7-step framework: Embed → t-SNE → Elbow auto-k → K-Medoids → "
            "Prototype extraction → (LLM labelling) → Label propagation.\n\n"
            "Use --sealclust_k <N> to skip auto-k and set k manually."
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

    # t-SNE
    parser.add_argument("--tsne_n_components", type=int, default=TSNE_N_COMPONENTS,
                        help="t-SNE output dimensionality (default: 2)")
    parser.add_argument("--tsne_perplexity", type=float, default=TSNE_PERPLEXITY,
                        help="t-SNE perplexity (default: 30)")
    parser.add_argument("--tsne_n_iter", type=int, default=TSNE_N_ITER,
                        help="t-SNE optimisation iterations (default: 1000)")
    parser.add_argument("--tsne_metric", type=str, default=TSNE_METRIC,
                        help="t-SNE distance metric (default: cosine)")

    # k selection
    parser.add_argument("--sealclust_k", type=int, default=SEALCLUST_K,
                        help="Manual k. Set to 0 (default) for Elbow auto-selection.")
    parser.add_argument("--elbow_k_min", type=int, default=SEALCLUST_ELBOW_K_MIN,
                        help="Min k for Elbow search (default: 5)")
    parser.add_argument("--elbow_k_max", type=int, default=SEALCLUST_ELBOW_K_MAX,
                        help="Max k for Elbow search (default: 200)")
    parser.add_argument("--elbow_step", type=int, default=SEALCLUST_ELBOW_STEP,
                        help="Step between candidate k values (default: 5)")

    # General
    parser.add_argument("--seed", type=int, default=42)

    # Propagation
    parser.add_argument("--propagate", action="store_true",
                        help="Run label propagation (Step 7) — requires --run_dir")

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
