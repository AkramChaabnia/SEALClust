"""
sealclust_v3_pipeline.py — SEAL-Clust v3 9-stage pipeline CLI.

Mode Z — Full end-to-end pipeline (Stages 1–9 + evaluation)::

    tc-sealclust-v3 --data massive_scenario --k0 300 --full
    tc-sealclust-v3 --data massive_scenario --k0 300 --k_star 18 --full
    tc-sealclust-v3 --data massive_scenario --k0 300 --cluster_method gmm --full

Mode Y — Step-by-step (Stages 1–7, then 8–9 separately)::

    # Stages 1–7
    tc-sealclust-v3 --data massive_scenario --k0 300 --k_star 18

    # Stage 8: classify representatives
    tc-sealclust-v3 --data massive_scenario --run_dir ./runs/<dir> --classify

    # Stage 9: propagate labels
    tc-sealclust-v3 --data massive_scenario --run_dir ./runs/<dir> --propagate

    # Evaluate
    tc-evaluate --data massive_scenario --run_dir ./runs/<dir>

v3 Improvements over v2
-----------------------
  - Multiple clustering backends (K-Medoids, GMM, KMeans)
  - One-word label constraint (labels are single general words)
  - Iterative chunked label consolidation (handles 500+ candidates)
  - Batched representative classification (Stage 8)
  - Representative selection works for all clustering methods
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
    SEALCLUST_BIC_K_MAX,
    SEALCLUST_BIC_K_MIN,
    SEALCLUST_K,
    SEALCLUST_K0,
    SEALCLUST_K_METHOD,
    SEALCLUST_LABEL_CHUNK_SIZE,
    SEALCLUST_PCA_DIMS,
    SEALCLUST_V3_CLASSIFY_BATCH,
    SEALCLUST_V3_CLUSTER_METHOD,
)
from text_clustering.data import load_dataset
from text_clustering.dimreduce import reduce_pca
from text_clustering.embedding import compute_embeddings
from text_clustering.logging_config import setup_logging
from text_clustering.sealclust import estimate_k_star
from text_clustering.sealclust_v3 import (
    classify_representatives_v3,
    consolidate_labels_v3,
    discover_labels_v3,
    propagate_labels_v3,
    run_overclustering,
    select_representatives,
)

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _make_run_dir(runs_dir: str, data: str, size: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"{data}_{size}_v3_{ts}")
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


# ── Stages 1–7 ────────────────────────────────────────────────────────────

def run_pipeline(args) -> str:
    """Run SEAL-Clust v3 Stages 1–7.

    Returns the run directory path.
    """
    size = "large" if args.use_large else "small"

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(args.runs_dir, args.data, size)

    setup_logging(os.path.join(run_dir, "sealclust_v3_pipeline.log"))

    logger.info("=" * 70)
    logger.info("SEAL-Clust v3 — 9-Stage Pipeline (Stages 1–7)")
    logger.info("=" * 70)
    logger.info("Dataset       : %s  |  split: %s", args.data, size)
    logger.info("Embedding     : %s", args.embedding_model)
    logger.info("Reduction     : %s", args.reduction if args.reduction != "none" else "none (raw embeddings)")
    if args.reduction == "pca":
        logger.info("  PCA dims    : %d", args.pca_dims)
    logger.info("Clustering    : %s", args.cluster_method)
    logger.info("Label source  : %s", args.label_source)
    logger.info("K₀ (overclust): %d", args.k0)
    if args.k_star:
        logger.info("K* (manual)   : %d", args.k_star)
    else:
        logger.info(
            "K* (auto)     : %s [%d–%d]",
            args.k_method, args.bic_k_min, args.bic_k_max,
        )
    logger.info("Run dir       : %s", run_dir)
    logger.info("-" * 70)
    start = time.time()

    # ── Stage 1: Document Embedding ──
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

    # ── Stage 2: Dimensionality Reduction (optional) ──
    if args.reduction == "none":
        logger.info("Stage 2: Skipped — using raw %dD embeddings", embeddings.shape[1])
        embeddings_for_clustering = embeddings
    else:
        reduced_path = os.path.join(run_dir, "embeddings_reduced.npy")
        if os.path.exists(reduced_path):
            logger.info("[cache] Loading reduced embeddings from %s", reduced_path)
            embeddings_for_clustering = np.load(reduced_path)
        elif args.reduction == "pca":
            logger.info("Stage 2: PCA %dD → %dD", embeddings.shape[1], args.pca_dims)
            embeddings_for_clustering = reduce_pca(
                embeddings, n_components=args.pca_dims, random_state=args.seed,
            )
            np.save(reduced_path, embeddings_for_clustering)
            logger.info("Stage 2: Reduced %s → %s", embeddings.shape, embeddings_for_clustering.shape)
        elif args.reduction == "tsne":
            from text_clustering.dimreduce import reduce_tsne
            logger.info("Stage 2: t-SNE %dD → 2D", embeddings.shape[1])
            embeddings_for_clustering = reduce_tsne(
                embeddings, random_state=args.seed,
            )
            np.save(reduced_path, embeddings_for_clustering)
            logger.info("Stage 2: Reduced %s → %s", embeddings.shape, embeddings_for_clustering.shape)
        else:
            raise ValueError(f"Unknown reduction method: {args.reduction!r}")

    # ── Stage 3: Overclustering ──
    meta_path = os.path.join(run_dir, "sealclust_v3_metadata.json")

    if os.path.exists(meta_path):
        logger.info("[cache] Loading v3 metadata from %s", meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        k0 = meta["k0"]
        cluster_labels = np.array(meta["cluster_assignments"])
        rep_indices = meta["representative_indices"]
        cluster_method = meta.get("cluster_method", args.cluster_method)
        logger.info("[cache] K₀=%d, method=%s, %d reps", k0, cluster_method, len(rep_indices))
    else:
        k0 = min(args.k0, n_documents - 1)
        logger.info(
            "Stage 3: Overclustering K₀=%d via %s on %dD",
            k0, args.cluster_method, embeddings_for_clustering.shape[1],
        )
        cluster_labels, extra, cluster_method = run_overclustering(
            embeddings_for_clustering, k0=k0, method=args.cluster_method,
            random_state=args.seed,
        )

        # ── Stage 4: Representative Selection ──
        logger.info("Stage 4: Selecting representatives (%s) …", cluster_method)
        rep_docs, rep_indices = select_representatives(
            data_list, embeddings_for_clustering, cluster_labels,
            cluster_method, extra,
        )
        _write_jsonl(os.path.join(run_dir, "representative_documents.jsonl"), rep_docs)

    # ── Stages 5-7: Labels ──
    reuse_labels = getattr(args, "reuse_labels", False)
    label_cache_dir = (
        getattr(args, "label_cache_dir", None)
        or os.path.join(args.runs_dir, "label_cache")
    )

    # Resolve K* (Stage 6)
    if args.k_star:
        k_star = args.k_star
        logger.info("Stage 6: Manual K*=%d (estimation skipped)", k_star)
    else:
        logger.info(
            "Stage 6: Estimating K* via %s on %d embeddings (dim=%d)",
            args.k_method, embeddings_for_clustering.shape[0],
            embeddings_for_clustering.shape[1],
        )
        k_star, k_details = estimate_k_star(
            embeddings_for_clustering,
            k_min=args.bic_k_min,
            k_max=args.bic_k_max,
            method=args.k_method,
            random_state=args.seed,
        )
        _write_json(os.path.join(run_dir, "k_estimation.json"), {
            "k_star": k_star,
            "method": args.k_method,
            "k_min": args.bic_k_min,
            "k_max": args.bic_k_max,
        })

    logger.info("K* = %d", k_star)

    # Label reuse check
    _labels_from_cache = False
    if reuse_labels:
        from text_clustering.label_cache import list_cached as _lc_list
        from text_clustering.label_cache import load_labels as _lc_load

        cached = _lc_load(label_cache_dir, args.data, size, n_labels=k_star)
        if cached is not None:
            final_labels = cached
            candidate_labels = cached
            _write_json(os.path.join(run_dir, "labels_merged.json"), final_labels)
            logger.info(
                "[label-reuse] Loaded %d cached labels for K*=%d",
                len(final_labels), k_star,
            )
            _labels_from_cache = True
        else:
            available = _lc_list(label_cache_dir, args.data, size)
            if available:
                logger.info(
                    "[label-reuse] No exact match for K*=%d; available: %s",
                    k_star, available,
                )

    if not _labels_from_cache:
        # ── Stage 5: Label Discovery ──
        # v3 sends ALL documents (not just representatives) in chunks.
        # Chunked LLM calls are cheap and give much better label coverage.
        labels_proposed_path = os.path.join(run_dir, "labels_proposed.json")
        _need_discovery = True

        if os.path.exists(labels_proposed_path):
            logger.info("[cache] Loading proposed labels from %s", labels_proposed_path)
            with open(labels_proposed_path) as f:
                candidate_labels = json.load(f)

            # If the cache has fewer labels than K*, invalidate and re-run
            if k_star and len(candidate_labels) < k_star:
                logger.warning(
                    "Stage 5 (v3): Cached labels_proposed.json has %d labels "
                    "but K*=%d — re-running label discovery",
                    len(candidate_labels), k_star,
                )
                _need_discovery = True
            else:
                _need_discovery = False

        if _need_discovery:
            if "client" not in dir():
                from text_clustering.llm import ini_client
                client = ini_client()

            # Pre-seed with existing labels if any (partial cache)
            existing = []
            if os.path.exists(labels_proposed_path):
                with open(labels_proposed_path) as f:
                    existing = json.load(f)

            # Choose which texts to send for label discovery
            if args.label_source == "representatives":
                # Only representative documents (K₀ texts — faster)
                discovery_texts = [texts[i] for i in rep_indices]
                logger.info(
                    "Stage 5 (v3): Discovering labels from %d REPRESENTATIVE texts "
                    "(--label_source representatives)",
                    len(discovery_texts),
                )
            else:
                # ALL documents (better label coverage)
                discovery_texts = texts
                logger.info(
                    "Stage 5 (v3): Discovering labels from ALL %d documents "
                    "(--label_source all)",
                    len(discovery_texts),
                )

            candidate_labels = discover_labels_v3(
                discovery_texts, client, chunk_size=args.label_chunk_size,
                run_dir=run_dir,
                min_labels=k_star if k_star else 0,
            )

            # Merge in any previously cached labels we hadn't found this time
            for lbl in existing:
                if lbl not in candidate_labels:
                    candidate_labels.append(lbl)

            _write_json(labels_proposed_path, candidate_labels)

        # ── Stage 7: Label Consolidation ──
        labels_merged_path = os.path.join(run_dir, "labels_merged.json")
        if os.path.exists(labels_merged_path):
            logger.info("[cache] Loading merged labels from %s", labels_merged_path)
            with open(labels_merged_path) as f:
                final_labels = json.load(f)
        else:
            if "client" not in dir():
                from text_clustering.llm import ini_client
                client = ini_client()

            final_labels = consolidate_labels_v3(
                candidate_labels, k_star, client,
            )
            _write_json(labels_merged_path, final_labels)

        if reuse_labels:
            from text_clustering.label_cache import save_labels as _lc_save
            _lc_save(label_cache_dir, args.data, size, final_labels)

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
            "pipeline": "sealclust_v3",
            "cluster_method": cluster_method,
            "reduction": args.reduction,
            "reduction_dims": embeddings_for_clustering.shape[1],
            "label_source": args.label_source,
            "k0": k0,
            "k_star": k_star,
            "k_star_method": "manual" if args.k_star else args.k_method,
            "n_candidate_labels": len(candidate_labels),
            "n_final_labels": len(final_labels),
            "embedding_model": args.embedding_model,
            "random_state": args.seed,
            "n_representatives": len(rep_indices),
            "representative_indices": [int(i) for i in rep_indices],
            "cluster_assignments": [int(c) for c in cluster_labels],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(meta_path, metadata)
    else:
        with open(meta_path) as f:
            metadata = json.load(f)
        metadata.update({
            "k_star": k_star,
            "k_star_method": "manual" if args.k_star else args.k_method,
            "n_candidate_labels": len(candidate_labels),
            "n_final_labels": len(final_labels),
        })
        _write_json(meta_path, metadata)

    elapsed = time.time() - start
    logger.info("=" * 70)
    logger.info("SEAL-Clust v3 Stages 1–7 complete in %.1fs", elapsed)
    logger.info("  Documents     : %d", n_documents)
    logger.info("  Clustering    : %s", cluster_method)
    logger.info(
        "  K₀ (overcl.)  : %d  (%.1f× compression)",
        k0, n_documents / max(k0, 1),
    )
    logger.info(
        "  K* (optimal)  : %d  (%s)",
        k_star, "manual" if args.k_star else args.k_method,
    )
    logger.info(
        "  Candidate labs: %d → Final labs: %d",
        len(candidate_labels), len(final_labels),
    )
    logger.info("  Run dir       : %s", run_dir)
    logger.info("")
    logger.info("Next steps:")
    logger.info(
        "  Stage 8: tc-sealclust-v3 --data %s --run_dir %s --classify",
        args.data, run_dir,
    )
    logger.info(
        "  Stage 9: tc-sealclust-v3 --data %s --run_dir %s --propagate",
        args.data, run_dir,
    )
    logger.info("  Evaluate: tc-evaluate --data %s --run_dir %s", args.data, run_dir)
    logger.info("=" * 70)

    return run_dir


# ── Stage 8 sub-command ───────────────────────────────────────────────────

def classify_reps(args) -> None:
    """Stage 8: Classify representatives into K* labels via LLM."""
    run_dir = args.run_dir
    setup_logging(os.path.join(run_dir, "sealclust_v3_classify.log"))

    logger.info("=" * 70)
    logger.info("SEAL-Clust v3 — Stage 8: Representative Classification")
    logger.info("=" * 70)

    # Load metadata
    meta_path = os.path.join(run_dir, "sealclust_v3_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Load final labels
    with open(os.path.join(run_dir, "labels_merged.json")) as f:
        final_labels = json.load(f)

    # Load representative texts
    rep_texts = []
    rep_docs_path = os.path.join(run_dir, "representative_documents.jsonl")
    with open(rep_docs_path) as f:
        for line in f:
            rep_texts.append(json.loads(line)["input"])

    logger.info("  Representatives: %d", len(rep_texts))
    logger.info("  Final labels:    %d", len(final_labels))

    from text_clustering.llm import ini_client
    client = ini_client()

    rep_labels = classify_representatives_v3(
        rep_texts, final_labels, client,
        batch_size=args.classify_batch_size,
        run_dir=run_dir,
    )

    # Save as classifications.json (compatible with tc-evaluate)
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    # IMPORTANT: use the original order from metadata (cluster_id order for
    # GMM/KMeans, sorted-doc-index for K-Medoids).  This matches the order
    # of rep_texts loaded from representative_documents.jsonl.
    rep_indices = meta["representative_indices"]

    classifications: dict[str, list[str]] = {}
    for order, rep_idx in enumerate(rep_indices):
        label = rep_labels.get(order, "Unsuccessful")
        classifications.setdefault(label, []).append(data_list[rep_idx]["input"])

    _write_json(os.path.join(run_dir, "classifications.json"), classifications)

    for label, members in sorted(classifications.items(), key=lambda x: -len(x[1])):
        logger.info("  %-30s %d reps", label, len(members))

    logger.info("Stage 8 complete — %d representatives classified", len(rep_texts))


# ── Stage 9 sub-command ───────────────────────────────────────────────────

def propagate(args) -> None:
    """Stage 9: Propagate representative labels to all documents."""
    run_dir = args.run_dir
    setup_logging(os.path.join(run_dir, "sealclust_v3_propagate.log"))

    logger.info("=" * 70)
    logger.info("SEAL-Clust v3 — Stage 9: Label Propagation")
    logger.info("=" * 70)

    # Load metadata
    meta_path = os.path.join(run_dir, "sealclust_v3_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    cluster_assignments = np.array(meta["cluster_assignments"])
    # IMPORTANT: preserve original order (cluster_id order for GMM/KMeans).
    rep_indices = meta["representative_indices"]
    n_documents = meta["n_documents"]

    # Load dataset
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    assert len(data_list) == n_documents, (
        f"Dataset size mismatch: expected {n_documents}, got {len(data_list)}"
    )

    # Load classifications.json
    class_path = os.path.join(run_dir, "classifications.json")
    with open(class_path) as f:
        classifications = json.load(f)

    # Build rep_index → label mapping
    rep_labels: dict[int, str] = {}
    rep_text_to_idx = {data_list[i]["input"]: i for i in rep_indices}

    for label, sentences in classifications.items():
        for sentence in sentences:
            if sentence in rep_text_to_idx:
                rep_idx = rep_text_to_idx[sentence]
                # Find order in rep_indices (original metadata order)
                order = rep_indices.index(rep_idx)
                rep_labels[order] = label

    logger.info(
        "Resolved labels for %d / %d representatives",
        len(rep_labels), len(rep_indices),
    )

    # Propagate
    all_labels = propagate_labels_v3(
        rep_labels, rep_indices, cluster_assignments, n_documents,
    )

    # Build output
    full_classifications: dict[str, list[str]] = {}
    for doc_idx, label in enumerate(all_labels):
        full_classifications.setdefault(label, []).append(data_list[doc_idx]["input"])
    full_classifications = {k: v for k, v in full_classifications.items() if v}

    _write_json(os.path.join(run_dir, "classifications_full.json"), full_classifications)

    for label, members in sorted(full_classifications.items(), key=lambda x: -len(x[1])):
        logger.info("  %-30s %d documents", label, len(members))

    total = sum(len(v) for v in full_classifications.values())
    logger.info("Propagation complete — %d / %d documents labelled", total, n_documents)


# ── Full pipeline (Stages 1–9 + evaluation) ──────────────────────────────

def run_full_pipeline(args) -> None:
    """Mode Z: Run the complete v3 pipeline end-to-end."""
    full_start = time.time()

    # Stages 1–7
    run_dir = run_pipeline(args)

    # Stage 8: Classify representatives
    logger.info("")
    logger.info("=" * 70)
    logger.info("SEAL-Clust v3 — Stage 8: Classify Representatives")
    logger.info("=" * 70)

    class_path = os.path.join(run_dir, "classifications.json")
    if os.path.exists(class_path):
        logger.info("[cache] classifications.json exists — skipping Stage 8")
    else:
        args.run_dir = run_dir
        classify_reps(args)

    # Stage 9: Propagate
    logger.info("")
    full_class_path = os.path.join(run_dir, "classifications_full.json")
    if os.path.exists(full_class_path):
        logger.info("[cache] classifications_full.json exists — skipping Stage 9")
    else:
        args.run_dir = run_dir
        propagate(args)

    # Evaluation
    logger.info("")
    logger.info("=" * 70)
    logger.info("SEAL-Clust v3 — Evaluation")
    logger.info("=" * 70)

    from text_clustering.pipeline.evaluation import build_parser as eval_parser
    from text_clustering.pipeline.evaluation import main as eval_main
    eval_args = eval_parser().parse_args([
        "--data", args.data,
        "--run_dir", run_dir,
    ])
    if args.use_large:
        eval_args.use_large = True
    eval_main(eval_args)

    # Visualization
    logger.info("")
    try:
        from text_clustering.visualization import generate_all_visualizations
        generate_all_visualizations(
            run_dir=run_dir,
            data_path=args.data_path,
            data_name=args.data,
            use_large=args.use_large,
        )
    except Exception as exc:
        logger.warning("Visualization failed (non-fatal): %s", exc)

    # Summary
    results_path = os.path.join(run_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        logger.info("")
        logger.info("=" * 70)
        logger.info("SEAL-Clust v3 — FULL PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info("  Dataset       : %s", args.data)
        logger.info("  Clustering    : %s", args.cluster_method)
        logger.info("  K₀ (overcl.)  : %d", args.k0)
        logger.info(
            "  K* (final)    : %d  (%s)",
            results.get("n_clusters_pred", "?"),
            "manual" if args.k_star else args.k_method,
        )
        logger.info("  ACC           : %.4f", results["ACC"])
        logger.info("  NMI           : %.4f", results["NMI"])
        logger.info("  ARI           : %.4f", results["ARI"])
        logger.info("  Run dir       : %s", run_dir)
        logger.info("  Total time    : %.1fs", time.time() - full_start)
        logger.info("=" * 70)


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SEAL-Clust v3 — Improved 9-Stage LLM Clustering Pipeline.\n\n"
            "v3 improvements: multi-method clustering (kmedoids/gmm/kmeans),\n"
            "one-word label constraint, iterative consolidation, batched\n"
            "classification.\n\n"
            "Modes:\n"
            "  Default      : Stages 1–7\n"
            "  --full       : Stages 1–9 + evaluation (Mode Z)\n"
            "  --classify   : Stage 8 only (requires --run_dir)\n"
            "  --propagate  : Stage 9 only (requires --run_dir)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Dataset
    parser.add_argument("--data_path", type=str, default="./datasets/")
    parser.add_argument(
        "--data", type=str, default="massive_scenario",
        help="Dataset name (subfolder under data_path)",
    )
    parser.add_argument("--runs_dir", type=str, default="./runs")
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Existing run directory (for --classify/--propagate or cache reuse)",
    )
    parser.add_argument("--use_large", action="store_true")

    # Embedding
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--batch_size", type=int, default=64)

    # Dimensionality reduction
    parser.add_argument(
        "--reduction", type=str, default="none",
        choices=["none", "pca", "tsne"],
        help="Dimensionality reduction method (default: none — raw embeddings)",
    )
    parser.add_argument(
        "--pca_dims", type=int, default=SEALCLUST_PCA_DIMS,
        help="PCA output dimensions, only used when --reduction pca (default: 50)",
    )

    # Overclustering
    parser.add_argument(
        "--k0", type=int, default=SEALCLUST_K0,
        help="Overclustering size K₀ (default: 300)",
    )
    parser.add_argument(
        "--cluster_method", type=str, default=SEALCLUST_V3_CLUSTER_METHOD,
        choices=["kmedoids", "gmm", "kmeans"],
        help="Clustering algorithm for Stage 3 (default: kmedoids)",
    )

    # K* estimation
    parser.add_argument(
        "--k_star", type=int, default=SEALCLUST_K if SEALCLUST_K > 0 else 0,
        help="Manual K* override. 0 = auto via selected method (default).",
    )
    parser.add_argument(
        "--k_method", type=str, default=SEALCLUST_K_METHOD,
        choices=["silhouette", "calinski", "bic", "ensemble"],
        help="K* estimation method (default: silhouette)",
    )
    parser.add_argument("--bic_k_min", type=int, default=SEALCLUST_BIC_K_MIN)
    parser.add_argument("--bic_k_max", type=int, default=SEALCLUST_BIC_K_MAX)

    # Label discovery
    parser.add_argument(
        "--label_chunk_size", type=int, default=SEALCLUST_LABEL_CHUNK_SIZE,
        help="Documents per LLM call for label discovery (default: 30)",
    )
    parser.add_argument(
        "--label_source", type=str, default="all",
        choices=["all", "representatives"],
        help=(
            "Which texts to send for label discovery: "
            "'all' = every document (better coverage), "
            "'representatives' = only K₀ representative docs (faster, fewer LLM calls). "
            "Default: all."
        ),
    )

    # Classification
    parser.add_argument(
        "--classify_batch_size", type=int, default=SEALCLUST_V3_CLASSIFY_BATCH,
        help="Representatives per LLM classification call (default: 20)",
    )

    # General
    parser.add_argument("--seed", type=int, default=42)

    # Sub-commands
    parser.add_argument(
        "--classify", action="store_true",
        help="Run Stage 8 only — classify representatives (requires --run_dir)",
    )
    parser.add_argument(
        "--propagate", action="store_true",
        help="Run Stage 9 only — propagate labels (requires --run_dir)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full pipeline end-to-end: Stages 1-9 + evaluation (Mode Z)",
    )

    # Label reuse
    parser.add_argument(
        "--reuse_labels", action="store_true", default=False,
        help="Enable label caching (reuse labels from previous runs).",
    )
    parser.add_argument(
        "--label_cache_dir", type=str, default=None,
        help="Directory for the shared label cache (default: <runs_dir>/label_cache).",
    )

    return parser


def main(args) -> None:
    if args.classify:
        if not args.run_dir:
            raise SystemExit("--run_dir is required when --classify is set")
        classify_reps(args)
    elif args.propagate:
        if not args.run_dir:
            raise SystemExit("--run_dir is required when --propagate is set")
        propagate(args)
    elif args.full:
        run_full_pipeline(args)
    else:
        run_pipeline(args)


def main_cli() -> None:
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())


if __name__ == "__main__":
    main_cli()
