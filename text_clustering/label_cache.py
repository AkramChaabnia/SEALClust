"""
label_cache.py — Shared label cache for cross-run label reuse.

When ``--reuse_labels`` is enabled, the pipeline saves its final merged
labels to a shared cache directory (``runs/label_cache/``) keyed by
**dataset + split + label count**.  Subsequent runs on the same dataset
with the same number of clusters can skip all LLM-based label generation
and merge steps, loading labels directly from the cache.

Cache Layout
------------
::

    runs/label_cache/
    └── massive_scenario_small_k18.json     ← 18-label set for massive_scenario (small)
    └── massive_scenario_small_k19.json     ← 19-label set (different K*)
    └── clinc_domain_small_k10.json
    └── ...

Each cache file is a plain JSON array of label strings — identical in
format to ``labels_merged.json``.

Cache Key
---------
``{dataset}_{split}_k{n_labels}``

The cache does **not** include the model name or timestamp in the key,
because:

* Labels are semantic — they describe topics, not model artifacts.
* Two different models producing K=18 labels for ``massive_scenario``
  should not overwrite each other by default.  Instead the user explicitly
  decides when to save (first run) and when to load (subsequent runs).
* The ``--reuse_labels`` flag must be explicitly passed — default behaviour
  is unchanged (generate every run).

Functions
---------
cache_key(dataset, split, n_labels)
    Build the cache key string.

cache_path(cache_dir, dataset, split, n_labels)
    Full filesystem path for a cached label file.

save_labels(cache_dir, dataset, split, labels)
    Save a label list to the cache.  n_labels = len(labels).

load_labels(cache_dir, dataset, split, n_labels)
    Load labels from cache.  Returns None if cache miss.

list_cached(cache_dir, dataset, split)
    List all cached label counts for a dataset+split.

has_cached(cache_dir, dataset, split, n_labels)
    Check if a specific cache entry exists.

find_best_cached(cache_dir, dataset, split, target_k)
    Find the cached entry closest to target_k, or None.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join("runs", "label_cache")


def cache_key(dataset: str, split: str, n_labels: int) -> str:
    """Build a cache key string: ``{dataset}_{split}_k{n_labels}``."""
    return f"{dataset}_{split}_k{n_labels}"


def cache_path(cache_dir: str, dataset: str, split: str, n_labels: int) -> str:
    """Full filesystem path for a cached label file."""
    return os.path.join(cache_dir, f"{cache_key(dataset, split, n_labels)}.json")


def save_labels(
    cache_dir: str,
    dataset: str,
    split: str,
    labels: list[str],
) -> str:
    """Save a label list to the shared cache.

    Parameters
    ----------
    cache_dir : str
        Root cache directory (e.g. ``runs/label_cache``).
    dataset : str
        Dataset name (e.g. ``massive_scenario``).
    split : str
        ``"small"`` or ``"large"``.
    labels : list[str]
        The final merged label list.

    Returns
    -------
    str
        Path where the cache file was written.
    """
    os.makedirs(cache_dir, exist_ok=True)
    n_labels = len(labels)
    path = cache_path(cache_dir, dataset, split, n_labels)
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)
    logger.info(
        "[label-cache] Saved %d labels → %s  (key=%s)",
        n_labels, path, cache_key(dataset, split, n_labels),
    )
    return path


def load_labels(
    cache_dir: str,
    dataset: str,
    split: str,
    n_labels: int | None = None,
) -> list[str] | None:
    """Load labels from cache.

    Parameters
    ----------
    cache_dir : str
        Root cache directory.
    dataset : str
        Dataset name.
    split : str
        ``"small"`` or ``"large"``.
    n_labels : int | None
        Exact label count to look for.  If *None*, loads the **most recent**
        (largest-K) cached set for this dataset+split.

    Returns
    -------
    list[str] | None
        Cached labels, or *None* on cache miss.
    """
    if n_labels is not None:
        path = cache_path(cache_dir, dataset, split, n_labels)
        if not os.path.exists(path):
            logger.info(
                "[label-cache] Miss — no cache for %s (looked for %s)",
                cache_key(dataset, split, n_labels), path,
            )
            return None
        with open(path) as f:
            labels = json.load(f)
        logger.info(
            "[label-cache] Hit — loaded %d labels from %s",
            len(labels), path,
        )
        return labels

    # No specific n_labels requested — find any cached entry
    available = list_cached(cache_dir, dataset, split)
    if not available:
        logger.info(
            "[label-cache] Miss — no cached labels for %s_%s",
            dataset, split,
        )
        return None

    # Pick the largest K available (most labels → most flexible)
    best_k = max(available)
    return load_labels(cache_dir, dataset, split, best_k)


def has_cached(cache_dir: str, dataset: str, split: str, n_labels: int) -> bool:
    """Check if a specific cache entry exists."""
    return os.path.exists(cache_path(cache_dir, dataset, split, n_labels))


def list_cached(cache_dir: str, dataset: str, split: str) -> list[int]:
    """List all cached label counts for a dataset+split.

    Returns
    -------
    list[int]
        Sorted list of K values available in cache (e.g. ``[10, 18, 25]``).
    """
    if not os.path.isdir(cache_dir):
        return []

    prefix = f"{dataset}_{split}_k"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.json$")
    result: list[int] = []
    for fname in os.listdir(cache_dir):
        m = pattern.match(fname)
        if m:
            result.append(int(m.group(1)))
    return sorted(result)


def find_best_cached(
    cache_dir: str,
    dataset: str,
    split: str,
    target_k: int,
) -> int | None:
    """Find the cached K closest to *target_k*, or *None* if nothing cached.

    An exact match is always preferred.  If no exact match, returns the
    closest K.  If there's a tie, prefers the lower K.
    """
    available = list_cached(cache_dir, dataset, split)
    if not available:
        return None
    if target_k in available:
        return target_k
    # Find closest
    return min(available, key=lambda k: (abs(k - target_k), k))


def describe_cache(cache_dir: str) -> str:
    """Return a human-readable summary of all cached label sets."""
    if not os.path.isdir(cache_dir):
        return "Label cache is empty (directory does not exist)."

    entries: list[str] = []
    pattern = re.compile(r"^(.+)_k(\d+)\.json$")
    for fname in sorted(os.listdir(cache_dir)):
        m = pattern.match(fname)
        if m:
            key = m.group(1)
            k = int(m.group(2))
            path = os.path.join(cache_dir, fname)
            size = os.path.getsize(path)
            entries.append(f"  {key:40s}  K={k:<4d}  ({size:,d} bytes)")

    if not entries:
        return "Label cache is empty."
    return "Cached label sets:\n" + "\n".join(entries)
