"""
hybrid_pipeline.py — Hybrid LLM + Embedding clustering pipeline CLI.

Implements the 8-step hybrid pipeline that combines LLM-based label generation
with embedding-based optimisation and probabilistic clustering.

The 8-Step Pipeline
-------------------
  1. **LLM Label Generation (K0)** — batch documents → one-word labels
  2. **Embedding Computation** — sentence-transformers → dense vectors
  3. **LLM Label Reduction (K1)** — merge synonymous labels
  4. **KMeans Optimisation** — silhouette analysis → optimal K
  5. **LLM Label Alignment** — force exactly K* labels
  6. **GMM Overclustering** — p × n microclusters → medoid extraction
  7. **LLM Medoid Labelling** — assign labels to each medoid
  8. **Label Propagation** — propagate medoid labels → all documents

Usage
-----
**Full pipeline (all 8 steps + evaluation)**::

    tc-hybrid --data massive_scenario --target_k 18 --full

**Full pipeline with auto K* (silhouette-based)**::

    tc-hybrid --data massive_scenario --full

**Step-by-step (inspect intermediate results)**::

    # Steps 1–5: Label discovery + optimisation
    tc-hybrid --data massive_scenario --target_k 18

    # Steps 6–8 + evaluation: require --run_dir
    tc-hybrid --data massive_scenario --run_dir ./runs/<dir> --continue_from 6

**Custom overclustering proportion**::

    tc-hybrid --data massive_scenario --p 0.2 --full

**Individual steps**::

    tc-hybrid --data massive_scenario --step 1   # LLM label generation only
    tc-hybrid --data massive_scenario --step 4   # KMeans optimisation only
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


# ── Pipeline Steps ────────────────────────────────────────────────────────

def run_steps_1_to_5(args) -> str:
    """Run Steps 1–5: LLM label generation → alignment.

    Returns the run directory path.
    """
    from text_clustering.hybrid import (
        step1_generate_labels,
        step2_compute_embeddings,
        step3_reduce_labels,
        step4_optimise_k,
        step5_align_labels,
    )

    size = "large" if args.use_large else "small"

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size)

    setup_logging(os.path.join(run_dir, "hybrid_pipeline.log"))

    logger.info("=" * 70)
    logger.info("Hybrid Pipeline — Steps 1–5")
    logger.info("=" * 70)
    logger.info("Dataset       : %s  |  split: %s", args.data, size)
    logger.info("Target K      : %s", args.target_k or "auto")
    logger.info("Batch size    : %d (LLM)", args.llm_batch_size)
    logger.info("Embedding     : %s", args.embedding_model)
    logger.info("Run dir       : %s", run_dir)
    logger.info("-" * 70)
    start = time.time()

    # Load dataset
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    texts = [item["input"] for item in data_list]
    n_documents = len(texts)
    logger.info("Loaded %d documents", n_documents)

    # Save ground-truth labels
    true_labels = get_label_list(data_list)
    _write_json(os.path.join(run_dir, "labels_true.json"), true_labels)
    logger.info("Ground-truth K = %d", len(true_labels))

    # ── Step 1: LLM Label Generation (K0) ──
    labels_k0_path = os.path.join(run_dir, "hybrid_labels_k0.json")
    per_doc_path = os.path.join(run_dir, "hybrid_per_doc_labels.json")

    if os.path.exists(labels_k0_path):
        logger.info("[cache] Loading K0 labels from %s", labels_k0_path)
        with open(labels_k0_path) as f:
            unique_labels_k0 = json.load(f)
        with open(per_doc_path) as f:
            per_doc_labels = json.load(f)
    else:
        from text_clustering.llm import ini_client
        client = ini_client()

        per_doc_labels, unique_labels_k0 = step1_generate_labels(
            texts, client, batch_size=args.llm_batch_size,
        )
        _write_json(labels_k0_path, unique_labels_k0)
        _write_json(per_doc_path, per_doc_labels)

    logger.info("Step 1: K0 = %d unique labels", len(unique_labels_k0))

    # ── Step 2: Embedding Computation ──
    emb_path = os.path.join(run_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        logger.info("[cache] Loading embeddings from %s", emb_path)
        embeddings = np.load(emb_path)
    else:
        embeddings = step2_compute_embeddings(
            texts, model_name=args.embedding_model,
            batch_size=args.embed_batch_size,
        )
        np.save(emb_path, embeddings)

    # ── Step 3: LLM Label Reduction (K0 → K1) ──
    labels_k1_path = os.path.join(run_dir, "hybrid_labels_k1.json")
    if os.path.exists(labels_k1_path):
        logger.info("[cache] Loading K1 labels from %s", labels_k1_path)
        with open(labels_k1_path) as f:
            labels_k1 = json.load(f)
    else:
        if "client" not in dir():
            from text_clustering.llm import ini_client
            client = ini_client()

        labels_k1 = step3_reduce_labels(unique_labels_k0, client)
        _write_json(labels_k1_path, labels_k1)

    logger.info("Step 3: K1 = %d labels (reduced from K0=%d)",
                len(labels_k1), len(unique_labels_k0))

    # ── Step 4: KMeans + Silhouette Optimisation ──
    k_opt_path = os.path.join(run_dir, "hybrid_k_optimisation.json")
    if os.path.exists(k_opt_path):
        logger.info("[cache] Loading K optimisation from %s", k_opt_path)
        with open(k_opt_path) as f:
            k_opt_data = json.load(f)
        optimal_k = k_opt_data["optimal_k"]
    else:
        optimal_k, sil_scores = step4_optimise_k(
            embeddings,
            k1=len(labels_k1),
            k_min=args.k_min,
            k_max=args.k_max,
            random_state=args.seed,
        )
        _write_json(k_opt_path, {
            "optimal_k": optimal_k,
            "k1": len(labels_k1),
            "k_min": args.k_min,
            "k_max": args.k_max,
            "silhouette_scores": {str(k): v for k, v in sil_scores.items()},
        })

    logger.info("Step 4: Optimal K = %d", optimal_k)

    # ── Step 5: LLM Label Alignment ──
    target_k = args.target_k if args.target_k else optimal_k

    labels_aligned_path = os.path.join(run_dir, "labels_merged.json")
    if os.path.exists(labels_aligned_path):
        logger.info("[cache] Loading aligned labels from %s", labels_aligned_path)
        with open(labels_aligned_path) as f:
            final_labels = json.load(f)
    else:
        if "client" not in dir():
            from text_clustering.llm import ini_client
            client = ini_client()

        final_labels = step5_align_labels(labels_k1, target_k, client)
        _write_json(labels_aligned_path, final_labels)

    logger.info("Step 5: %d final labels (target was %d)", len(final_labels), target_k)

    # Save metadata
    meta_path = os.path.join(run_dir, "hybrid_metadata.json")
    metadata = {
        "dataset": args.data,
        "split": size,
        "pipeline": "hybrid",
        "n_documents": n_documents,
        "k0": len(unique_labels_k0),
        "k1": len(labels_k1),
        "optimal_k": optimal_k,
        "target_k": target_k,
        "n_final_labels": len(final_labels),
        "embedding_model": args.embedding_model,
        "llm_batch_size": args.llm_batch_size,
        "random_state": args.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(meta_path, metadata)

    elapsed = time.time() - start
    logger.info("=" * 70)
    logger.info("Steps 1–5 complete in %.1fs", elapsed)
    logger.info("  K0=%d → K1=%d → optimal_k=%d → target_k=%d → final=%d",
                len(unique_labels_k0), len(labels_k1), optimal_k,
                target_k, len(final_labels))
    logger.info("  Run dir: %s", run_dir)
    logger.info("=" * 70)

    return run_dir


def run_steps_6_to_8(args, run_dir: str) -> None:
    """Run Steps 6–8: GMM overclustering → medoid labelling → propagation."""
    from text_clustering.hybrid import (
        step6_gmm_overclustering,
        step7_label_medoids,
        step8_propagate_labels,
    )

    setup_logging(os.path.join(run_dir, "hybrid_pipeline.log"))

    logger.info("=" * 70)
    logger.info("Hybrid Pipeline — Steps 6–8")
    logger.info("=" * 70)
    logger.info("Run dir       : %s", run_dir)
    logger.info("p (microclust): %.2f", args.p)
    start = time.time()

    # Load data
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    n_documents = len(data_list)

    # Load embeddings
    emb_path = os.path.join(run_dir, "embeddings.npy")
    embeddings = np.load(emb_path)
    logger.info("Loaded embeddings shape=%s", embeddings.shape)

    # Load final labels
    labels_path = os.path.join(run_dir, "labels_merged.json")
    with open(labels_path) as f:
        final_labels = json.load(f)
    logger.info("Loaded %d final labels", len(final_labels))

    # ── Step 6: GMM Overclustering ──
    gmm_meta_path = os.path.join(run_dir, "hybrid_gmm_metadata.json")
    medoid_path = os.path.join(run_dir, "medoid_documents.jsonl")

    if os.path.exists(gmm_meta_path):
        logger.info("[cache] Loading GMM metadata from %s", gmm_meta_path)
        with open(gmm_meta_path) as f:
            gmm_meta = json.load(f)
        gmm_labels = np.array(gmm_meta["cluster_assignments"])
        medoid_indices = np.array(gmm_meta["medoid_indices"])
        medoid_docs = [data_list[int(i)] for i in medoid_indices]
    else:
        gmm_labels, gmm_probs, medoid_docs, medoid_indices = (
            step6_gmm_overclustering(
                data_list, embeddings,
                p=args.p,
                covariance_type=args.covariance_type,
                random_state=args.seed,
            )
        )

        _write_jsonl(medoid_path, medoid_docs)
        np.save(os.path.join(run_dir, "gmm_probs.npy"), gmm_probs)

        gmm_meta = {
            "n_documents": n_documents,
            "p": args.p,
            "n_microclusters": int(max(2, int(args.p * n_documents))),
            "n_medoids": len(medoid_indices),
            "medoid_indices": [int(i) for i in medoid_indices],
            "cluster_assignments": [int(c) for c in gmm_labels],
            "covariance_type": args.covariance_type,
        }
        _write_json(gmm_meta_path, gmm_meta)

    logger.info("Step 6: %d medoids from %d microclusters",
                len(medoid_indices), gmm_meta.get("n_microclusters", "?"))

    # ── Step 7: LLM Medoid Labelling ──
    medoid_labels_path = os.path.join(run_dir, "hybrid_medoid_labels.json")

    if os.path.exists(medoid_labels_path):
        logger.info("[cache] Loading medoid labels from %s", medoid_labels_path)
        with open(medoid_labels_path) as f:
            medoid_labels_raw = json.load(f)
        medoid_labels = {int(k): v for k, v in medoid_labels_raw.items()}
    else:
        from text_clustering.llm import ini_client
        client = ini_client()

        medoid_labels = step7_label_medoids(
            medoid_docs, medoid_indices, final_labels, client,
        )
        _write_json(medoid_labels_path, {
            str(k): v for k, v in medoid_labels.items()
        })

    # ── Step 8: Label Propagation ──
    all_labels = step8_propagate_labels(
        medoid_labels, gmm_labels, n_documents,
    )

    # Build classifications output (compatible with evaluation)
    classifications: dict[str, list[str]] = {}
    for doc_idx, label in enumerate(all_labels):
        classifications.setdefault(label, []).append(data_list[doc_idx]["input"])
    classifications = {k: v for k, v in classifications.items() if v}

    _write_json(os.path.join(run_dir, "classifications.json"), classifications)
    _write_json(os.path.join(run_dir, "classifications_full.json"), classifications)

    # Update metadata
    meta_path = os.path.join(run_dir, "hybrid_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    metadata.update({
        "p": args.p,
        "n_microclusters": gmm_meta.get("n_microclusters"),
        "n_medoids": len(medoid_indices),
        "covariance_type": args.covariance_type,
    })
    _write_json(meta_path, metadata)

    elapsed = time.time() - start
    logger.info("=" * 70)
    logger.info("Steps 6–8 complete in %.1fs", elapsed)
    logger.info("  Microclusters : %s (p=%.2f)",
                gmm_meta.get("n_microclusters", "?"), args.p)
    logger.info("  Medoids       : %d", len(medoid_indices))
    n_success = sum(
        1 for v in medoid_labels.values() if v != "Unsuccessful"
    )
    logger.info("  Labelled      : %d/%d medoids", n_success, len(medoid_labels))
    logger.info("=" * 70)


def run_full_pipeline(args) -> None:
    """Run the complete Hybrid pipeline (Steps 1–8 + evaluation)."""
    full_start = time.time()

    # Steps 1–5
    run_dir = run_steps_1_to_5(args)

    # Steps 6–8
    args.run_dir = run_dir
    run_steps_6_to_8(args, run_dir)

    # Evaluation
    logger.info("")
    logger.info("=" * 70)
    logger.info("Hybrid Pipeline — Evaluation")
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
        logger.info("HYBRID PIPELINE — COMPLETE")
        logger.info("=" * 70)
        logger.info("  Dataset       : %s", args.data)
        logger.info("  K* (final)    : %d", results.get("n_clusters_pred", "?"))
        logger.info("  ACC           : %.4f", results["ACC"])
        logger.info("  NMI           : %.4f", results["NMI"])
        logger.info("  ARI           : %.4f", results["ARI"])
        logger.info("  p             : %.2f", args.p)
        logger.info("  Run dir       : %s", run_dir)
        logger.info("  Total time    : %.1fs", time.time() - full_start)
        logger.info("=" * 70)


def run_single_step(args) -> None:
    """Run a single step of the pipeline (--step N)."""
    size = "large" if args.use_large else "small"

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size)

    setup_logging(os.path.join(run_dir, "hybrid_pipeline.log"))

    data_list = load_dataset(args.data_path, args.data, args.use_large)
    texts = [item["input"] for item in data_list]

    step = args.step

    if step == 1:
        from text_clustering.hybrid import step1_generate_labels
        from text_clustering.llm import ini_client
        client = ini_client()
        per_doc_labels, unique_labels_k0 = step1_generate_labels(
            texts, client, batch_size=args.llm_batch_size,
        )
        _write_json(os.path.join(run_dir, "hybrid_labels_k0.json"), unique_labels_k0)
        _write_json(os.path.join(run_dir, "hybrid_per_doc_labels.json"), per_doc_labels)
        logger.info("Step 1 complete: K0=%d labels → %s", len(unique_labels_k0), run_dir)

    elif step == 2:
        from text_clustering.hybrid import step2_compute_embeddings
        embeddings = step2_compute_embeddings(
            texts, model_name=args.embedding_model,
            batch_size=args.embed_batch_size,
        )
        np.save(os.path.join(run_dir, "embeddings.npy"), embeddings)
        logger.info("Step 2 complete: embeddings shape=%s → %s", embeddings.shape, run_dir)

    elif step == 3:
        from text_clustering.hybrid import step3_reduce_labels
        from text_clustering.llm import ini_client
        k0_path = os.path.join(run_dir, "hybrid_labels_k0.json")
        with open(k0_path) as f:
            labels_k0 = json.load(f)
        client = ini_client()
        labels_k1 = step3_reduce_labels(labels_k0, client)
        _write_json(os.path.join(run_dir, "hybrid_labels_k1.json"), labels_k1)
        logger.info("Step 3 complete: K0=%d → K1=%d → %s",
                     len(labels_k0), len(labels_k1), run_dir)

    elif step == 4:
        from text_clustering.hybrid import step4_optimise_k
        embeddings = np.load(os.path.join(run_dir, "embeddings.npy"))
        with open(os.path.join(run_dir, "hybrid_labels_k1.json")) as f:
            labels_k1 = json.load(f)
        optimal_k, sil_scores = step4_optimise_k(
            embeddings, k1=len(labels_k1),
            k_min=args.k_min, k_max=args.k_max,
            random_state=args.seed,
        )
        _write_json(os.path.join(run_dir, "hybrid_k_optimisation.json"), {
            "optimal_k": optimal_k,
            "silhouette_scores": {str(k): v for k, v in sil_scores.items()},
        })
        logger.info("Step 4 complete: optimal K=%d → %s", optimal_k, run_dir)

    elif step == 5:
        from text_clustering.hybrid import step5_align_labels
        from text_clustering.llm import ini_client
        with open(os.path.join(run_dir, "hybrid_labels_k1.json")) as f:
            labels_k1 = json.load(f)
        with open(os.path.join(run_dir, "hybrid_k_optimisation.json")) as f:
            k_opt = json.load(f)
        target_k = args.target_k if args.target_k else k_opt["optimal_k"]
        client = ini_client()
        final_labels = step5_align_labels(labels_k1, target_k, client)
        _write_json(os.path.join(run_dir, "labels_merged.json"), final_labels)
        logger.info("Step 5 complete: %d final labels → %s",
                     len(final_labels), run_dir)

    elif step == 6:
        from text_clustering.hybrid import step6_gmm_overclustering
        embeddings = np.load(os.path.join(run_dir, "embeddings.npy"))
        gmm_labels, gmm_probs, medoid_docs, medoid_indices = (
            step6_gmm_overclustering(
                data_list, embeddings,
                p=args.p,
                covariance_type=args.covariance_type,
                random_state=args.seed,
            )
        )
        _write_jsonl(os.path.join(run_dir, "medoid_documents.jsonl"), medoid_docs)
        np.save(os.path.join(run_dir, "gmm_probs.npy"), gmm_probs)
        _write_json(os.path.join(run_dir, "hybrid_gmm_metadata.json"), {
            "n_documents": len(data_list),
            "p": args.p,
            "n_microclusters": int(max(2, int(args.p * len(data_list)))),
            "n_medoids": len(medoid_indices),
            "medoid_indices": [int(i) for i in medoid_indices],
            "cluster_assignments": [int(c) for c in gmm_labels],
        })
        logger.info("Step 6 complete: %d medoids → %s",
                     len(medoid_indices), run_dir)

    elif step == 7:
        from text_clustering.hybrid import step7_label_medoids
        from text_clustering.llm import ini_client
        with open(os.path.join(run_dir, "labels_merged.json")) as f:
            final_labels = json.load(f)
        with open(os.path.join(run_dir, "hybrid_gmm_metadata.json")) as f:
            gmm_meta = json.load(f)
        medoid_indices = np.array(gmm_meta["medoid_indices"])
        medoid_docs = [data_list[int(i)] for i in medoid_indices]
        client = ini_client()
        medoid_labels = step7_label_medoids(
            medoid_docs, medoid_indices, final_labels, client,
        )
        _write_json(os.path.join(run_dir, "hybrid_medoid_labels.json"), {
            str(k): v for k, v in medoid_labels.items()
        })
        logger.info("Step 7 complete: labelled %d medoids → %s",
                     len(medoid_labels), run_dir)

    elif step == 8:
        from text_clustering.hybrid import step8_propagate_labels
        with open(os.path.join(run_dir, "hybrid_gmm_metadata.json")) as f:
            gmm_meta = json.load(f)
        with open(os.path.join(run_dir, "hybrid_medoid_labels.json")) as f:
            medoid_labels_raw = json.load(f)
        gmm_labels = np.array(gmm_meta["cluster_assignments"])
        medoid_labels = {int(k): v for k, v in medoid_labels_raw.items()}
        all_labels = step8_propagate_labels(
            medoid_labels, gmm_labels, len(data_list),
        )
        classifications: dict[str, list[str]] = {}
        for doc_idx, label in enumerate(all_labels):
            classifications.setdefault(label, []).append(
                data_list[doc_idx]["input"]
            )
        classifications = {k: v for k, v in classifications.items() if v}
        _write_json(os.path.join(run_dir, "classifications.json"), classifications)
        _write_json(
            os.path.join(run_dir, "classifications_full.json"), classifications,
        )
        logger.info("Step 8 complete: propagated to %d documents → %s",
                     len(data_list), run_dir)
    else:
        raise SystemExit(f"Invalid step: {step}. Must be 1–8.")


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid LLM + Embedding clustering pipeline.\n\n"
            "8-Step pipeline:\n"
            "  1. LLM Label Gen (K0)    2. Embed    3. LLM Label Reduce (K1)\n"
            "  4. KMeans Optimise K     5. LLM Label Align\n"
            "  6. GMM Overcluster       7. LLM Medoid Label    8. Propagate\n\n"
            "Default: runs Steps 1–5. Use --full for end-to-end."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Dataset
    parser.add_argument("--data_path", type=str, default="./datasets/")
    parser.add_argument("--data", type=str, default="massive_scenario",
                        help="Dataset name (subfolder under data_path)")
    parser.add_argument("--runs_dir", type=str, default="./runs")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Existing run directory (for cache reuse or --step)")
    parser.add_argument("--use_large", action="store_true")

    # LLM label generation
    parser.add_argument("--llm_batch_size", type=int, default=30,
                        help="Documents per LLM call in Step 1 (default: 30)")

    # Embedding
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--embed_batch_size", type=int, default=64,
                        help="Batch size for embedding computation")

    # KMeans optimisation
    parser.add_argument("--k_min", type=int, default=2,
                        help="Min K for silhouette search (default: 2)")
    parser.add_argument("--k_max", type=int, default=50,
                        help="Max K for silhouette search (default: 50)")

    # Target K
    parser.add_argument("--target_k", type=int, default=0,
                        help="Target number of categories. 0 = auto (from Step 4)")

    # GMM overclustering
    parser.add_argument("--p", type=float, default=0.1,
                        help="Overclustering proportion: n_micro = p × n (default: 0.1)")
    parser.add_argument("--covariance_type", type=str, default="tied",
                        choices=["full", "tied", "diag", "spherical"])

    # General
    parser.add_argument("--seed", type=int, default=42)

    # Execution mode
    parser.add_argument("--full", action="store_true",
                        help="Run all 8 steps + evaluation end-to-end")
    parser.add_argument("--continue_from", type=int, default=0,
                        help="Continue from step N (6–8), requires --run_dir")
    parser.add_argument("--step", type=int, default=0,
                        help="Run a single step (1–8)")

    return parser


def main(args) -> None:
    if args.step:
        run_single_step(args)
    elif args.full:
        run_full_pipeline(args)
    elif args.continue_from >= 6:
        if not args.run_dir:
            raise SystemExit("--run_dir is required for --continue_from")
        run_steps_6_to_8(args, args.run_dir)
    else:
        run_steps_1_to_5(args)


def main_cli() -> None:
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())


if __name__ == "__main__":
    main_cli()
