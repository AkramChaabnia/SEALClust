"""
visualization.py — Post-run visualization module for the clustering pipeline.

Generates and saves multiple plots to ``<run_dir>/assets/``:

  1. **Confusion matrix** — Hungarian-aligned predicted vs true labels
  2. **Dimensionality-reduction scatter plots** — UMAP, PCA, t-SNE
     (two versions each: colored by predicted cluster and by true label)
  3. **Cluster comparison** — side-by-side predicted vs true scatter plots
  4. **Cluster distribution** — bar chart and histogram of cluster sizes

All functions are idempotent: re-running overwrites existing plots.

Usage
-----
**Standalone CLI** (on an existing run)::

    tc-visualize --data massive_scenario --run_dir ./runs/massive_scenario_small_20260313_141759

**Programmatic** (from within the pipeline)::

    from text_clustering.visualization import generate_all_visualizations
    generate_all_visualizations(run_dir, data_path, data_name, use_large)

Integration
-----------
Called automatically at the end of:
  - ``sealclust_pipeline.run_full_pipeline()``  (after evaluation)
  - ``evaluation.main()``  (after metrics are computed)
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

ASSETS_DIR = "assets"
DPI = 150
SCATTER_POINT_SIZE = 8
SCATTER_ALPHA = 0.6
MAX_LABELS_IN_LEGEND = 30  # hide legend if too many classes (clutters the plot)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: load run data
# ═══════════════════════════════════════════════════════════════════════════

def _load_run_data(run_dir: str, data_path: str, data_name: str, use_large: bool) -> dict:
    """Load all artefacts needed for visualization from a completed run directory.

    Returns a dict with keys:
        texts, true_labels, pred_labels, embeddings, embeddings_reduced,
        true_label_ids, pred_label_ids, true_unique, pred_unique
    """
    from text_clustering.data import load_dataset

    # ── Dataset ──
    data_list = load_dataset(data_path, data_name, use_large)
    texts = [item["input"] for item in data_list]
    true_labels_raw = [item["label"] for item in data_list]

    # ── Predicted labels ──
    classifications = _load_classifications(run_dir)

    # Build text → predicted label mapping
    text_to_pred: dict[str, str] = {}
    for label, sentences in classifications.items():
        for s in sentences:
            text_to_pred[s] = label

    # Align: keep only texts that appear in both dataset and predictions
    true_labels: list[str] = []
    pred_labels: list[str] = []
    indices: list[int] = []
    for i, text in enumerate(texts):
        if text in text_to_pred:
            true_labels.append(true_labels_raw[i])
            pred_labels.append(text_to_pred[text])
            indices.append(i)

    n = len(true_labels)
    logger.info("Matched %d / %d documents with predictions", n, len(texts))

    # ── Numerical IDs ──
    true_unique = sorted(set(true_labels))
    pred_unique = sorted(set(pred_labels))
    true_map = {label: idx for idx, label in enumerate(true_unique)}
    pred_map = {label: idx for idx, label in enumerate(pred_unique)}
    true_ids = np.array([true_map[lab] for lab in true_labels])
    pred_ids = np.array([pred_map[lab] for lab in pred_labels])

    # ── Embeddings ──
    emb_path = os.path.join(run_dir, "embeddings.npy")
    embeddings = np.load(emb_path) if os.path.exists(emb_path) else None

    emb_red_path = os.path.join(run_dir, "embeddings_reduced.npy")
    embeddings_reduced = np.load(emb_red_path) if os.path.exists(emb_red_path) else None

    # Sub-select embeddings to matched indices
    if embeddings is not None:
        embeddings = embeddings[indices]
    if embeddings_reduced is not None:
        embeddings_reduced = embeddings_reduced[indices]

    return {
        "texts": [texts[i] for i in indices],
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "true_ids": true_ids,
        "pred_ids": pred_ids,
        "true_unique": true_unique,
        "pred_unique": pred_unique,
        "embeddings": embeddings,
        "embeddings_reduced": embeddings_reduced,
        "n_samples": n,
    }


def _load_classifications(run_dir: str) -> dict:
    """Load predicted classifications — prefer propagated file."""
    full_path = os.path.join(run_dir, "classifications_full.json")
    base_path = os.path.join(run_dir, "classifications.json")
    path = full_path if os.path.exists(full_path) else base_path
    with open(path) as f:
        return json.load(f)


def _ensure_assets_dir(run_dir: str) -> str:
    assets = os.path.join(run_dir, ASSETS_DIR)
    os.makedirs(assets, exist_ok=True)
    return assets


# ═══════════════════════════════════════════════════════════════════════════
# 1. Confusion Matrix (Hungarian-aligned)
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
    true_unique: list[str],
    pred_unique: list[str],
    assets_dir: str,
) -> str:
    """Generate a Hungarian-aligned confusion matrix and save it.

    Returns the path to the saved image.
    """
    from scipy.optimize import linear_sum_assignment

    n_true = len(true_unique)
    n_pred = len(pred_unique)
    D = max(n_true, n_pred)

    # Build the cost matrix
    w = np.zeros((D, D), dtype=int)
    for t, p in zip(true_ids, pred_ids):
        w[p, t] += 1

    # Hungarian alignment (maximize → flip sign)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # Reorder pred to match true via the alignment
    pred_order = [None] * D
    for r, c in zip(row_ind, col_ind):
        pred_order[c] = r

    # Build the aligned matrix (true on y-axis, pred on x-axis)
    cm = np.zeros((n_true, n_pred), dtype=int)
    for t, p in zip(true_ids, pred_ids):
        cm[t, p] += 1

    # Reorder columns (pred) by alignment
    aligned_cols = [c for c in col_ind if c < n_pred]
    remaining = [i for i in range(n_pred) if i not in aligned_cols]
    col_order = aligned_cols + remaining
    cm_aligned = cm[:, col_order]
    pred_labels_ordered = [pred_unique[i] for i in col_order]

    # ── Plot ──
    fig_height = max(6, n_true * 0.4)
    fig_width = max(8, n_pred * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(cm_aligned, cmap="Blues", aspect="auto")

    ax.set_xticks(range(n_pred))
    ax.set_xticklabels(pred_labels_ordered, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_true))
    ax.set_yticklabels(true_unique, fontsize=7)
    ax.set_xlabel("Predicted Cluster", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title("Confusion Matrix (Hungarian-Aligned)", fontsize=12, fontweight="bold")

    # Annotate cells (skip if matrix is too large)
    if n_true * n_pred <= 900:
        for i in range(n_true):
            for j in range(n_pred):
                val = cm_aligned[i, j]
                if val > 0:
                    color = "white" if val > cm_aligned.max() / 2 else "black"
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()

    path = os.path.join(assets_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 2. Dimensionality Reduction Scatter Plots
# ═══════════════════════════════════════════════════════════════════════════

def _compute_2d_projection(embeddings: np.ndarray, method: str, seed: int = 42) -> np.ndarray:
    """Compute a 2D projection using the specified method.

    Parameters
    ----------
    embeddings : shape (n_samples, dim)
    method : "pca", "tsne", or "umap"

    Returns
    -------
    np.ndarray, shape (n_samples, 2)
    """
    n_samples = embeddings.shape[0]

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=seed)
        return reducer.fit_transform(embeddings)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30.0, max(5.0, n_samples / 4.0))
        reducer = TSNE(
            n_components=2, perplexity=perplexity,
            max_iter=1000, random_state=seed,
            init="pca", metric="cosine",
        )
        return reducer.fit_transform(embeddings)

    elif method == "umap":
        import umap
        n_neighbors = min(15, max(2, n_samples - 1))
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=0.1, metric="cosine", random_state=seed,
        )
        return reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Unknown projection method: {method}")


def _scatter_plot(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    unique_labels: list[str],
    title: str,
    save_path: str,
    cmap_name: str = "tab20",
) -> str:
    """Generic 2D scatter colored by label IDs."""
    n_classes = len(unique_labels)
    cmap = plt.get_cmap(cmap_name, max(n_classes, 1))

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=labels, cmap=cmap, s=SCATTER_POINT_SIZE, alpha=SCATTER_ALPHA,
        edgecolors="none",
    )

    if n_classes <= MAX_LABELS_IN_LEGEND:
        handles = []
        for i, name in enumerate(unique_labels):
            h = plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=cmap(i / max(n_classes - 1, 1)),
                markersize=6, label=name,
            )
            handles.append(h)
        ax.legend(
            handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=6, frameon=True, ncol=1 + n_classes // 25,
        )
    else:
        fig.colorbar(scatter, ax=ax, shrink=0.6, label="Cluster ID")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimension 1", fontsize=9)
    ax.set_ylabel("Dimension 2", fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()

    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter plot → %s", save_path)
    return save_path


def plot_projections(
    embeddings: np.ndarray,
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
    true_unique: list[str],
    pred_unique: list[str],
    assets_dir: str,
    methods: list[str] | None = None,
    seed: int = 42,
) -> list[str]:
    """Generate 2D projection scatter plots for each method × {pred, true}.

    Returns list of saved file paths.
    """
    if methods is None:
        methods = ["pca", "tsne", "umap"]

    saved: list[str] = []

    for method in methods:
        logger.info("Computing %s 2D projection …", method.upper())
        try:
            coords = _compute_2d_projection(embeddings, method, seed)
        except Exception as exc:
            logger.warning("Skipping %s projection: %s", method.upper(), exc)
            continue

        # Predicted clusters
        path = _scatter_plot(
            coords, pred_ids, pred_unique,
            title=f"{method.upper()} — Predicted Clusters",
            save_path=os.path.join(assets_dir, f"{method}_predicted.png"),
        )
        saved.append(path)

        # True labels
        path = _scatter_plot(
            coords, true_ids, true_unique,
            title=f"{method.upper()} — True Labels",
            save_path=os.path.join(assets_dir, f"{method}_true.png"),
        )
        saved.append(path)

    return saved


# ═══════════════════════════════════════════════════════════════════════════
# 3. Side-by-Side Comparison (Predicted vs True)
# ═══════════════════════════════════════════════════════════════════════════

def plot_side_by_side(
    embeddings: np.ndarray,
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
    true_unique: list[str],
    pred_unique: list[str],
    assets_dir: str,
    method: str = "pca",
    seed: int = 42,
) -> str:
    """Side-by-side scatter: predicted clusters vs true labels on the same projection."""
    logger.info("Computing %s projection for side-by-side comparison …", method.upper())
    try:
        coords = _compute_2d_projection(embeddings, method, seed)
    except Exception as exc:
        logger.warning("Cannot create side-by-side plot: %s", exc)
        return ""

    fig, (ax_pred, ax_true) = plt.subplots(1, 2, figsize=(18, 7))

    n_pred = len(pred_unique)
    n_true = len(true_unique)
    cmap_pred = plt.get_cmap("tab20", max(n_pred, 1))
    cmap_true = plt.get_cmap("tab20", max(n_true, 1))

    # Left: predicted
    ax_pred.scatter(
        coords[:, 0], coords[:, 1],
        c=pred_ids, cmap=cmap_pred, s=SCATTER_POINT_SIZE,
        alpha=SCATTER_ALPHA, edgecolors="none",
    )
    ax_pred.set_title(f"Predicted Clusters (n={n_pred})", fontsize=11, fontweight="bold")
    ax_pred.set_xlabel("Dimension 1", fontsize=9)
    ax_pred.set_ylabel("Dimension 2", fontsize=9)
    ax_pred.tick_params(labelsize=7)

    # Right: true
    ax_true.scatter(
        coords[:, 0], coords[:, 1],
        c=true_ids, cmap=cmap_true, s=SCATTER_POINT_SIZE,
        alpha=SCATTER_ALPHA, edgecolors="none",
    )
    ax_true.set_title(f"True Labels (n={n_true})", fontsize=11, fontweight="bold")
    ax_true.set_xlabel("Dimension 1", fontsize=9)
    ax_true.set_ylabel("Dimension 2", fontsize=9)
    ax_true.tick_params(labelsize=7)

    fig.suptitle(
        f"Cluster Comparison — {method.upper()} Projection",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(assets_dir, f"comparison_{method}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved side-by-side comparison → %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 4. Cluster Distribution
# ═══════════════════════════════════════════════════════════════════════════

def plot_cluster_distribution(
    true_labels: list[str],
    pred_labels: list[str],
    assets_dir: str,
) -> list[str]:
    """Generate cluster size bar chart + histogram for predicted and true labels."""
    saved: list[str] = []

    for name, labels in [("predicted", pred_labels), ("true", true_labels)]:
        # Count per cluster
        from collections import Counter
        counts = Counter(labels)
        sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        names = [item[0] for item in sorted_items]
        sizes = [item[1] for item in sorted_items]

        # ── Bar chart ──
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.35), 6))
        bars = ax.bar(range(len(names)), sizes, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Number of Documents", fontsize=10)
        ax.set_title(f"Cluster Size Distribution — {name.title()} Labels (n={len(names)})",
                      fontsize=11, fontweight="bold")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Annotate bar values
        for bar, val in zip(bars, sizes):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=5,
            )

        plt.tight_layout()
        path = os.path.join(assets_dir, f"distribution_{name}.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved distribution bar chart → %s", path)
        saved.append(path)

    # ── Combined histogram (overlay) ──
    from collections import Counter
    pred_sizes = list(Counter(pred_labels).values())
    true_sizes = list(Counter(true_labels).values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(true_sizes, bins=20, alpha=0.6, label="True", color="forestgreen", edgecolor="white")
    ax.hist(pred_sizes, bins=20, alpha=0.6, label="Predicted", color="steelblue", edgecolor="white")
    ax.set_xlabel("Cluster Size", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Cluster Size Histogram — True vs Predicted", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()

    path = os.path.join(assets_dir, "histogram_cluster_sizes.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cluster size histogram → %s", path)
    saved.append(path)

    return saved


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_visualizations(
    run_dir: str,
    data_path: str = "./datasets/",
    data_name: str | None = None,
    use_large: bool = False,
    projection_methods: list[str] | None = None,
    seed: int = 42,
) -> list[str]:
    """Generate all visualizations for a completed run.

    Parameters
    ----------
    run_dir : str
        Path to the run directory (e.g. ``./runs/massive_scenario_small_20260313``).
    data_path : str
        Root path to datasets.
    data_name : str or None
        Dataset name. If None, inferred from ``sealclust_metadata.json`` or
        ``results.json`` in the run directory.
    use_large : bool
        Whether the large split was used.
    projection_methods : list[str] or None
        Which 2D projections to compute. Default: ["pca", "tsne", "umap"].
    seed : int
        Random state for projections.

    Returns
    -------
    list[str]
        Paths to all saved images.
    """
    from text_clustering.logging_config import setup_logging
    setup_logging(os.path.join(run_dir, "visualization.log"))

    logger.info("=" * 70)
    logger.info("Visualization — Generating plots")
    logger.info("=" * 70)
    logger.info("Run dir: %s", run_dir)

    # ── Infer dataset name if not provided ──
    if data_name is None:
        data_name = _infer_dataset_name(run_dir)
    if data_name is None:
        raise ValueError(
            "Cannot infer dataset name from run directory. "
            "Please pass --data explicitly."
        )
    logger.info("Dataset: %s", data_name)

    # ── Load all data ──
    data = _load_run_data(run_dir, data_path, data_name, use_large)
    assets_dir = _ensure_assets_dir(run_dir)

    saved: list[str] = []

    # ── 1. Confusion matrix ──
    logger.info("─" * 40)
    logger.info("1/4  Confusion Matrix")
    saved.append(plot_confusion_matrix(
        data["true_ids"], data["pred_ids"],
        data["true_unique"], data["pred_unique"],
        assets_dir,
    ))

    # ── 2. Projection scatter plots ──
    embeddings = data["embeddings"]
    if embeddings is None:
        logger.warning("No embeddings.npy found — skipping projection plots.")
    else:
        logger.info("─" * 40)
        logger.info("2/4  Projection Scatter Plots")

        # Determine which methods to use
        methods = projection_methods or ["pca", "tsne", "umap"]

        # Check if umap is available
        if "umap" in methods:
            try:
                import umap  # noqa: F401
            except ImportError:
                logger.warning("umap-learn not installed — skipping UMAP projection.")
                methods = [m for m in methods if m != "umap"]

        saved.extend(plot_projections(
            embeddings, data["true_ids"], data["pred_ids"],
            data["true_unique"], data["pred_unique"],
            assets_dir, methods=methods, seed=seed,
        ))

        # ── 3. Side-by-side comparison ──
        logger.info("─" * 40)
        logger.info("3/4  Side-by-Side Comparison")
        # Use PCA for the comparison (fastest and most deterministic)
        path = plot_side_by_side(
            embeddings, data["true_ids"], data["pred_ids"],
            data["true_unique"], data["pred_unique"],
            assets_dir, method="pca", seed=seed,
        )
        if path:
            saved.append(path)

    # ── 4. Cluster distribution ──
    logger.info("─" * 40)
    logger.info("4/4  Cluster Distribution")
    saved.extend(plot_cluster_distribution(
        data["true_labels"], data["pred_labels"], assets_dir,
    ))

    logger.info("=" * 70)
    logger.info("Visualization complete — %d plots saved to %s", len(saved), assets_dir)
    logger.info("=" * 70)

    return saved


def _infer_dataset_name(run_dir: str) -> str | None:
    """Try to infer the dataset name from metadata or results files."""
    for fname in ("sealclust_metadata.json", "results.json"):
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                meta = json.load(f)
            if "dataset" in meta:
                return meta["dataset"]
    # Fallback: parse from directory name (e.g. "massive_scenario_small_20260313_141759")
    dirname = os.path.basename(run_dir.rstrip("/"))
    parts = dirname.rsplit("_", 2)  # remove timestamp parts
    if len(parts) >= 3:
        # e.g. "massive_scenario_small" → remove size suffix
        candidate = "_".join(parts[:-2])
        # Remove size suffix (small/large)
        for suffix in ("_small", "_large"):
            if candidate.endswith(suffix):
                candidate = candidate[: -len(suffix)]
        return candidate if candidate else None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate post-run visualizations for a clustering experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the completed run directory.",
    )
    parser.add_argument(
        "--data_path", type=str, default="./datasets/",
        help="Root path to datasets (default: ./datasets/).",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Dataset name. If omitted, inferred from run metadata.",
    )
    parser.add_argument(
        "--use_large", action="store_true",
        help="Use large split (default: small).",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=None,
        choices=["pca", "tsne", "umap"],
        help="Projection methods to use (default: pca tsne umap).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for projections.",
    )
    return parser


def main(args=None) -> None:
    if args is None:
        args = build_parser().parse_args()

    generate_all_visualizations(
        run_dir=args.run_dir,
        data_path=args.data_path,
        data_name=args.data,
        use_large=args.use_large,
        projection_methods=args.methods,
        seed=args.seed,
    )


def main_cli() -> None:
    """Console-script entry point (registered in pyproject.toml)."""
    main()


if __name__ == "__main__":
    main_cli()
