"""
dataset_analysis.py — Automated dataset profiling and statistics.

Produces structured JSON reports for every dataset inside ``datasets/``.
Reports are written to ``assets/<dataset_name>_stats.json`` and contain
per-split statistics (sample count, label distribution, text length stats,
imbalance indicators) as well as a cross-split summary.

Public API
----------
analyze_split(records)
    Compute statistics for a single list of JSONL records.

analyze_dataset(dataset_dir)
    Analyze both splits of a dataset directory and return the full report.

analyze_all(datasets_dir, assets_dir)
    Iterate over every sub-directory in *datasets_dir*, analyze each
    dataset, and write one JSON file per dataset into *assets_dir*.

Usage
-----
    # As a library
    from text_clustering.dataset_analysis import analyze_all
    analyze_all("datasets", "assets")

    # As a CLI (via entry-point registered in pyproject.toml)
    tc-analyze-datasets
    tc-analyze-datasets --datasets datasets --assets assets --dataset arxiv_fine
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
_SPLITS = ("small", "large")
_IMBALANCE_RATIO_WARN = 10.0  # max/min ≥ 10× → flag as imbalanced


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping blank / malformed lines."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning("%s:%d — skipped malformed JSON line", path, lineno)
    return records


def _safe_div(a: float, b: float) -> float:
    """Return a/b or 0.0 when b is zero."""
    return a / b if b else 0.0


def _entropy(counts: list[int]) -> float:
    """Shannon entropy (bits) over a list of counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return round(ent, 4)


def _gini_impurity(counts: list[int]) -> float:
    """Gini impurity index ∈ [0, 1)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return round(1.0 - sum((c / total) ** 2 for c in counts), 4)


# ── Core analysis ─────────────────────────────────────────────────────────

def analyze_split(records: list[dict]) -> dict:
    """Compute comprehensive statistics for a single JSONL split.

    Parameters
    ----------
    records : list[dict]
        Each dict must have at least ``"input"`` and ``"label"`` keys.

    Returns
    -------
    dict
        Statistics dictionary with the following top-level keys:

        - ``n_samples``
        - ``n_labels``
        - ``labels`` — sorted list of unique label strings
        - ``label_counts`` — {label: count} sorted descending
        - ``label_pct`` — {label: pct} sorted descending
        - ``text_length`` — min / max / mean / median / std / total_chars
        - ``imbalance`` — ratio, entropy, gini, warnings
    """
    if not records:
        return {"n_samples": 0, "n_labels": 0}

    # Label distribution
    label_counter: Counter[str] = Counter()
    text_lengths: list[int] = []

    for rec in records:
        label = rec.get("label", "<MISSING>")
        label_counter[label] += 1
        text = rec.get("input", "")
        text_lengths.append(len(text))

    n_samples = len(records)
    n_labels = len(label_counter)

    # Sort labels by count descending, then alphabetically for ties
    sorted_labels = sorted(label_counter.items(), key=lambda kv: (-kv[1], kv[0]))

    label_counts = {lab: cnt for lab, cnt in sorted_labels}
    label_pct = {lab: round(cnt / n_samples * 100, 2) for lab, cnt in sorted_labels}

    # Text length stats
    tl_mean = round(statistics.mean(text_lengths), 1) if text_lengths else 0.0
    tl_median = round(statistics.median(text_lengths), 1) if text_lengths else 0.0
    tl_std = round(statistics.stdev(text_lengths), 1) if len(text_lengths) > 1 else 0.0

    text_length_stats = {
        "min": min(text_lengths) if text_lengths else 0,
        "max": max(text_lengths) if text_lengths else 0,
        "mean": tl_mean,
        "median": tl_median,
        "stdev": tl_std,
        "total_chars": sum(text_lengths),
    }

    # Imbalance indicators
    counts_list = list(label_counter.values())
    max_count = max(counts_list)
    min_count = min(counts_list)
    imbalance_ratio = round(_safe_div(max_count, min_count), 2)

    warnings: list[str] = []
    if imbalance_ratio >= _IMBALANCE_RATIO_WARN:
        warnings.append(
            f"High class imbalance: largest class is {imbalance_ratio}× "
            f"the smallest ({max_count} vs {min_count} samples)."
        )

    # Singleton labels
    singletons = [lab for lab, cnt in label_counter.items() if cnt == 1]
    if singletons:
        warnings.append(
            f"{len(singletons)} singleton label(s) (count=1): "
            f"{singletons[:5]}{'...' if len(singletons) > 5 else ''}"
        )

    avg_per_label = round(n_samples / n_labels, 1) if n_labels else 0.0

    imbalance = {
        "ratio_max_min": imbalance_ratio,
        "entropy_bits": _entropy(counts_list),
        "max_entropy_bits": round(math.log2(n_labels), 4) if n_labels > 1 else 0.0,
        "gini_impurity": _gini_impurity(counts_list),
        "avg_samples_per_label": avg_per_label,
        "largest_class": sorted_labels[0][0] if sorted_labels else None,
        "largest_class_count": max_count,
        "smallest_class": sorted_labels[-1][0] if sorted_labels else None,
        "smallest_class_count": min_count,
        "warnings": warnings,
    }

    return {
        "n_samples": n_samples,
        "n_labels": n_labels,
        "labels": sorted(label_counter.keys()),
        "label_counts": label_counts,
        "label_pct": label_pct,
        "text_length": text_length_stats,
        "imbalance": imbalance,
    }


def analyze_dataset(dataset_dir: str | Path) -> dict:
    """Analyze both splits of a single dataset and produce a full report.

    Parameters
    ----------
    dataset_dir : Path
        Directory that contains ``small.jsonl`` and/or ``large.jsonl``.

    Returns
    -------
    dict
        ``{"dataset_name": ..., "splits": {"small": {...}, "large": {...}}, "summary": {...}}``
    """
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name

    splits: dict[str, dict] = {}
    for split in _SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            logger.warning("Missing split: %s", path)
            splits[split] = {"error": f"{split}.jsonl not found"}
            continue
        records = _load_jsonl(path)
        logger.info("Analyzing %s/%s.jsonl — %d records", dataset_name, split, len(records))
        splits[split] = analyze_split(records)

    # Cross-split summary
    summary = _build_summary(dataset_name, splits)

    return {
        "dataset_name": dataset_name,
        "splits": splits,
        "summary": summary,
    }


def _build_summary(dataset_name: str, splits: dict[str, dict]) -> dict:
    """Build a cross-split summary comparing small vs large."""
    small = splits.get("small", {})
    large = splits.get("large", {})

    s_n = small.get("n_samples", 0)
    l_n = large.get("n_samples", 0)
    s_k = small.get("n_labels", 0)
    l_k = large.get("n_labels", 0)

    # Label overlap between splits
    s_labels = set(small.get("labels", []))
    l_labels = set(large.get("labels", []))
    shared = sorted(s_labels & l_labels)
    only_small = sorted(s_labels - l_labels)
    only_large = sorted(l_labels - s_labels)

    return {
        "dataset_name": dataset_name,
        "total_samples": s_n + l_n,
        "small_samples": s_n,
        "large_samples": l_n,
        "size_ratio_large_to_small": round(_safe_div(l_n, s_n), 2),
        "small_n_labels": s_k,
        "large_n_labels": l_k,
        "labels_shared": len(shared),
        "labels_only_in_small": only_small if only_small else None,
        "labels_only_in_large": only_large if only_large else None,
        "label_sets_identical": s_labels == l_labels and len(s_labels) > 0,
    }


def analyze_all(
    datasets_dir: str | Path = "datasets",
    assets_dir: str | Path = "assets",
    *,
    dataset_filter: str | None = None,
) -> list[Path]:
    """Iterate over every dataset directory, analyze, and write JSON reports.

    Parameters
    ----------
    datasets_dir : str | Path
        Root directory containing dataset sub-directories.
    assets_dir : str | Path
        Directory where ``<name>_stats.json`` files will be written.
    dataset_filter : str | None
        If set, only process this single dataset name.

    Returns
    -------
    list[Path]
        Paths of all generated JSON report files.
    """
    datasets_dir = Path(datasets_dir)
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not datasets_dir.is_dir():
        logger.error("Datasets directory does not exist: %s", datasets_dir)
        return []

    # Discover dataset directories (any subdir that contains at least one JSONL)
    candidates = sorted(
        d for d in datasets_dir.iterdir()
        if d.is_dir() and any(d.glob("*.jsonl"))
    )

    if dataset_filter:
        candidates = [d for d in candidates if d.name == dataset_filter]
        if not candidates:
            logger.error("Dataset '%s' not found in %s", dataset_filter, datasets_dir)
            return []

    generated: list[Path] = []
    for dataset_dir in candidates:
        logger.info("─── Analyzing: %s ───", dataset_dir.name)
        report = analyze_dataset(dataset_dir)
        out_path = assets_dir / f"{dataset_dir.name}_stats.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("  → %s", out_path)
        generated.append(out_path)

    logger.info("Done — %d dataset(s) analyzed, reports in %s/", len(generated), assets_dir)
    return generated


# ── CLI entry-point ───────────────────────────────────────────────────────

def main_cli() -> None:
    """Entry-point for ``tc-analyze-datasets``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze all datasets and generate structured JSON reports.",
    )
    parser.add_argument(
        "--datasets",
        default="datasets",
        help="Root directory containing dataset sub-folders (default: datasets)",
    )
    parser.add_argument(
        "--assets",
        default="assets",
        help="Output directory for JSON reports (default: assets)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Analyze only this single dataset (e.g. arxiv_fine)",
    )
    args = parser.parse_args()

    # Basic logging to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    generated = analyze_all(
        datasets_dir=args.datasets,
        assets_dir=args.assets,
        dataset_filter=args.dataset,
    )

    if not generated:
        logger.warning("No reports generated.")
        raise SystemExit(1)

    # Print a quick summary table
    print("\n" + "=" * 72)
    print(f"  {'Dataset':<22} {'Small':>8} {'Large':>8} {'Labels':>7}  Balanced?")
    print("-" * 72)
    for path in generated:
        with open(path) as fh:
            rpt = json.load(fh)
        s = rpt["splits"].get("small", {})
        lg = rpt["splits"].get("large", {})
        n_s = s.get("n_samples", "—")
        n_l = lg.get("n_samples", "—")
        n_k = s.get("n_labels", lg.get("n_labels", "—"))
        warnings = s.get("imbalance", {}).get("warnings", [])
        balanced = "⚠️" if warnings else "✓"
        print(f"  {rpt['dataset_name']:<22} {n_s:>8} {n_l:>8} {n_k:>7}  {balanced}")
    print("=" * 72)


if __name__ == "__main__":
    main_cli()
