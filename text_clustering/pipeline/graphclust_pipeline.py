"""
graphclust_pipeline.py — Graph Community Clustering CLI.

A fundamentally different approach to text clustering: builds a k-NN
embedding similarity graph, discovers clusters via Louvain community
detection, and uses the LLM **only for post-hoc labelling** (~K calls).

The 3-Step Pipeline
-------------------
  1. **Build k-NN Graph** — Cosine similarity in embedding space
  2. **Community Detection** — Louvain modularity optimisation (binary
     search over resolution γ to hit target_k)
  3. **LLM Post-hoc Labelling** — Name each community (~K LLM calls)

Usage
-----
**Full pipeline (all 3 steps + evaluation)**::

    tc-graphclust --data massive_scenario --target_k 18 --full

**Auto-detect K (no target_k — Louvain decides)**::

    tc-graphclust --data massive_scenario --full

**Steps 1–2 only (clustering without labels)**::

    tc-graphclust --data massive_scenario --target_k 18

**Step 3 + evaluation only (requires existing run)**::

    tc-graphclust --data massive_scenario --run_dir ./runs/<dir> --label_only

**Using Make**::

    make run-graphclust-full data=massive_scenario target_k=18
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

def _make_run_dir(runs_dir: str, data: str, size: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"{data}_{size}_graphclust_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Wrote %s", path)


# ── Pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(args, label_only: bool = False) -> str:
    """Run the full or partial graph community clustering pipeline."""
    from text_clustering.graphclust import (
        build_classifications,
        step1_build_knn_graph,
        step2_detect_communities,
        step3_label_communities,
    )

    size = "large" if args.use_large else "small"

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size)

    setup_logging(os.path.join(run_dir, "graphclust_pipeline.log"))

    logger.info("=" * 70)
    logger.info("GRAPH COMMUNITY CLUSTERING — Louvain + LLM Post-hoc Labelling")
    logger.info("=" * 70)
    logger.info("Dataset       : %s  |  split: %s", args.data, size)
    logger.info("k-NN          : %d", args.knn)
    logger.info("Min similarity: %.2f", args.min_similarity)
    logger.info("Target K      : %s", args.target_k or "auto")
    logger.info("Resolution    : %.2f", args.resolution)
    logger.info("Run dir       : %s", run_dir)
    logger.info("-" * 70)
    full_start = time.time()

    # Load dataset
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    texts = [item["input"] for item in data_list]
    n_documents = len(texts)
    logger.info("Loaded %d documents", n_documents)

    # Save ground-truth labels
    true_labels = get_label_list(data_list)
    _write_json(os.path.join(run_dir, "labels_true.json"), true_labels)
    logger.info("Ground-truth K = %d", len(true_labels))

    if not label_only:
        # ── Embeddings (shared, cached) ──
        emb_path = os.path.join(run_dir, "embeddings.npy")

        if os.path.exists(emb_path):
            logger.info("[cache] Loading embeddings from %s", emb_path)
            embeddings = np.load(emb_path)
        else:
            from text_clustering.embedding import compute_embeddings
            embeddings = compute_embeddings(
                texts, model_name=args.embedding_model,
                batch_size=args.embed_batch_size,
            )
            np.save(emb_path, embeddings)

        # ── Step 1: Build k-NN Graph ──
        graph_path = os.path.join(run_dir, "graphclust_graph.npz")

        if os.path.exists(graph_path):
            logger.info("[cache] Loading graph from %s", graph_path)
            data = np.load(graph_path)
            rows, cols, weights = data["rows"], data["cols"], data["weights"]
        else:
            rows, cols, weights = step1_build_knn_graph(
                embeddings,
                knn=args.knn,
                min_similarity=args.min_similarity,
            )
            np.savez(graph_path, rows=rows, cols=cols, weights=weights)

        # ── Step 2: Community Detection ──
        community_path = os.path.join(run_dir, "graphclust_communities.json")

        if os.path.exists(community_path):
            logger.info("[cache] Loading communities from %s", community_path)
            with open(community_path) as f:
                comm_data = json.load(f)
            community_labels = np.array(comm_data["community_labels"])
            n_communities = comm_data["n_communities"]
            best_resolution = comm_data["resolution"]
        else:
            community_labels, n_communities, best_resolution = (
                step2_detect_communities(
                    n_nodes=n_documents,
                    row_indices=rows,
                    col_indices=cols,
                    weights=weights,
                    resolution=args.resolution,
                    target_k=args.target_k,
                )
            )
            _write_json(community_path, {
                "n_communities": n_communities,
                "resolution": best_resolution,
                "community_labels": [int(c) for c in community_labels],
                "community_sizes": {
                    str(c): int(s) for c, s in
                    zip(*np.unique(community_labels, return_counts=True))
                },
            })

        logger.info("Steps 1–2 complete: %d communities detected", n_communities)

        # Save metadata
        _write_json(os.path.join(run_dir, "graphclust_metadata.json"), {
            "pipeline": "graphclust_community",
            "dataset": args.data,
            "split": size,
            "n_documents": n_documents,
            "n_communities": n_communities,
            "resolution": best_resolution,
            "knn": args.knn,
            "min_similarity": args.min_similarity,
            "target_k": args.target_k,
            "embedding_model": args.embedding_model,
            "random_state": args.seed,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

        if not args.full and not label_only:
            elapsed = time.time() - full_start
            logger.info("=" * 70)
            logger.info("Steps 1–2 complete in %.1fs", elapsed)
            logger.info("  Communities: %d  |  Resolution: %.3f",
                        n_communities, best_resolution)
            logger.info("  Run dir: %s", run_dir)
            logger.info("=" * 70)
            return run_dir

    else:
        # label_only mode: load existing community data
        with open(os.path.join(run_dir, "graphclust_communities.json")) as f:
            comm_data = json.load(f)
        community_labels = np.array(comm_data["community_labels"])
        n_communities = comm_data["n_communities"]
        best_resolution = comm_data["resolution"]

    # ── Step 3: Post-hoc Labelling ──
    labels_path = os.path.join(run_dir, "graphclust_community_names.json")

    if os.path.exists(labels_path):
        logger.info("[cache] Loading community names from %s", labels_path)
        with open(labels_path) as f:
            community_names = {int(k): v for k, v in json.load(f).items()}
    else:
        from text_clustering.llm import ini_client
        client = ini_client()

        community_names = step3_label_communities(
            community_labels, texts, client,
            samples_per_community=args.samples_per_community,
            random_state=args.seed,
        )
        _write_json(labels_path, {str(k): v for k, v in community_names.items()})

    # Build classifications output
    classifications = build_classifications(
        community_labels, community_names, texts,
    )
    _write_json(os.path.join(run_dir, "classifications.json"), classifications)
    _write_json(os.path.join(run_dir, "classifications_full.json"), classifications)

    # Build labels_merged.json for evaluation compatibility
    merged_labels = list(community_names.values())
    _write_json(os.path.join(run_dir, "labels_merged.json"), merged_labels)

    # ── Evaluation ──
    if args.full or label_only:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Graph Community Clustering — Evaluation")
        logger.info("=" * 70)

        from text_clustering.pipeline.evaluation import (
            build_parser as eval_parser,
        )
        from text_clustering.pipeline.evaluation import (
            main as eval_main,
        )
        eval_args = eval_parser().parse_args([
            "--data", args.data,
            "--run_dir", run_dir,
        ])
        if args.use_large:
            eval_args.use_large = True
        eval_main(eval_args)

        # Summary
        results_path = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            logger.info("")
            logger.info("=" * 70)
            logger.info("GRAPH COMMUNITY CLUSTERING — COMPLETE")
            logger.info("=" * 70)
            logger.info("  Dataset       : %s", args.data)
            logger.info("  K (detected)  : %d", n_communities)
            logger.info("  K (true)      : %d", len(true_labels))
            logger.info("  ACC           : %.4f", results.get("ACC", 0))
            logger.info("  NMI           : %.4f", results.get("NMI", 0))
            logger.info("  ARI           : %.4f", results.get("ARI", 0))
            logger.info("  k-NN          : %d", args.knn)
            logger.info("  Min similarity: %.2f", args.min_similarity)
            logger.info("  Resolution    : %.3f", best_resolution)
            logger.info("  LLM calls     : ~%d (post-hoc labelling only)",
                        n_communities)
            logger.info("  Run dir       : %s", run_dir)
            logger.info("  Total time    : %.1fs", time.time() - full_start)
            logger.info("=" * 70)

    return run_dir


def run_single_step(args) -> None:
    """Run a single step of the pipeline (--step N)."""
    from text_clustering.graphclust import (
        build_classifications,
        step1_build_knn_graph,
        step2_detect_communities,
        step3_label_communities,
    )

    size = "large" if args.use_large else "small"

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size)

    setup_logging(os.path.join(run_dir, "graphclust_pipeline.log"))

    data_list = load_dataset(args.data_path, args.data, args.use_large)
    texts = [item["input"] for item in data_list]

    step = args.step

    if step == 1:
        from text_clustering.embedding import compute_embeddings

        emb_path = os.path.join(run_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
        else:
            embeddings = compute_embeddings(
                texts, model_name=args.embedding_model,
                batch_size=args.embed_batch_size,
            )
            np.save(emb_path, embeddings)

        rows, cols, weights = step1_build_knn_graph(
            embeddings, knn=args.knn, min_similarity=args.min_similarity,
        )
        np.savez(
            os.path.join(run_dir, "graphclust_graph.npz"),
            rows=rows, cols=cols, weights=weights,
        )
        logger.info("Step 1 complete: %d edges → %s", len(weights), run_dir)

    elif step == 2:
        data = np.load(os.path.join(run_dir, "graphclust_graph.npz"))
        rows, cols, weights = data["rows"], data["cols"], data["weights"]

        community_labels, n_communities, best_resolution = (
            step2_detect_communities(
                n_nodes=len(texts),
                row_indices=rows, col_indices=cols, weights=weights,
                resolution=args.resolution, target_k=args.target_k,
            )
        )
        _write_json(os.path.join(run_dir, "graphclust_communities.json"), {
            "n_communities": n_communities,
            "resolution": best_resolution,
            "community_labels": [int(c) for c in community_labels],
            "community_sizes": {
                str(c): int(s) for c, s in
                zip(*np.unique(community_labels, return_counts=True))
            },
        })
        logger.info("Step 2 complete: %d communities → %s",
                     n_communities, run_dir)

    elif step == 3:
        from text_clustering.llm import ini_client

        with open(os.path.join(run_dir, "graphclust_communities.json")) as f:
            comm_data = json.load(f)
        community_labels = np.array(comm_data["community_labels"])

        client = ini_client()
        community_names = step3_label_communities(
            community_labels, texts, client,
            samples_per_community=args.samples_per_community,
            random_state=args.seed,
        )
        _write_json(os.path.join(run_dir, "graphclust_community_names.json"), {
            str(k): v for k, v in community_names.items()
        })

        classifications = build_classifications(
            community_labels, community_names, texts,
        )
        _write_json(os.path.join(run_dir, "classifications.json"), classifications)
        _write_json(os.path.join(run_dir, "classifications_full.json"),
                     classifications)

        merged_labels = list(community_names.values())
        _write_json(os.path.join(run_dir, "labels_merged.json"), merged_labels)
        logger.info("Step 3 complete: %d communities labelled → %s",
                     len(community_names), run_dir)
    else:
        raise SystemExit(f"Invalid step: {step}. Must be 1–3.")


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Graph Community Clustering with LLM Post-hoc Labelling.\n\n"
            "Builds a k-NN embedding graph, discovers clusters via Louvain\n"
            "community detection, then names communities with the LLM.\n"
            "Fundamentally different from geometric partitioning (KMeans/GMM).\n\n"
            "3-Step pipeline:\n"
            "  1. Build k-NN Graph (embedding cosine similarity)\n"
            "  2. Community Detection (Louvain modularity optimisation)\n"
            "  3. LLM Post-hoc Labelling (~K calls)\n\n"
            "Default: runs Steps 1–2. Use --full for end-to-end."
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

    # Step 1: k-NN graph
    parser.add_argument("--knn", type=int, default=15,
                        help="k nearest neighbours per document (default: 15)")
    parser.add_argument("--min_similarity", type=float, default=0.3,
                        help="Min cosine similarity to create an edge (default: 0.3)")

    # Step 2: Community detection
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Louvain resolution γ (higher → more clusters)")
    parser.add_argument("--target_k", type=int, default=0,
                        help="Target communities (0 = auto via resolution)")

    # Step 3: Labelling
    parser.add_argument("--samples_per_community", type=int, default=8,
                        help="Docs to sample per community for labelling")

    # Embedding
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--embed_batch_size", type=int, default=64)

    # General
    parser.add_argument("--seed", type=int, default=42)

    # Execution mode
    parser.add_argument("--full", action="store_true",
                        help="Run all 3 steps + evaluation end-to-end")
    parser.add_argument("--label_only", action="store_true",
                        help="Run Step 3 + evaluation only (requires --run_dir)")
    parser.add_argument("--step", type=int, default=0,
                        help="Run a single step (1–3)")

    return parser


def main(args) -> None:
    if args.step:
        run_single_step(args)
    elif args.label_only:
        if not args.run_dir:
            raise SystemExit("--run_dir is required for --label_only")
        run_pipeline(args, label_only=True)
    else:
        run_pipeline(args)


def main_cli() -> None:
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())


if __name__ == "__main__":
    main_cli()
