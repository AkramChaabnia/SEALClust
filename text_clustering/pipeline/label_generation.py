"""
label_generation.py — Step 1 of the clustering pipeline.

For each dataset, the script:
  1. Loads the dataset and shuffles it.
  2. Reads the seed labels produced by seed_labels.py (chosen_labels.json).
  3. Iterates over the texts in chunks of --chunk_size (default 15).
     For each chunk, it asks the LLM to propose new label names for texts
     that don't fit any existing label.
  4. Merges the resulting label set with a second LLM call that deduplicates
     near-synonymous labels.
  5. Writes all outputs to a timestamped run directory under --runs_dir:
       runs/<data>_<size>_<timestamp>/
         labels_true.json           ground-truth label list
         labels_proposed.json       all labels before merge
         labels_merged.json         merged/deduplicated labels  ← used by Step 2

Run directory path is printed on start so Step 2 can be pointed at it via --run_dir.

The LLM is configured entirely through environment variables (see .env.example).
Model selection, temperature, and token limits are read at startup — no code
change is needed to switch models.

Original source: ECNU-Text-Computing/Text-Clustering-via-LLM
Modifications:
  - ini_client() wired to text_clustering.client.make_client()
  - response_format disabled by default (LLM_FORCE_JSON_MODE=false)
  - _strip_fenced_json() strips markdown fences before eval()
  - chat() retries up to 5 times with backoff on 429 errors
  - model read from LLM_MODEL env var
  - outputs written to timestamped run directories under ./runs/
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime

from text_clustering.data import get_label_list, load_dataset
from text_clustering.llm import chat, ini_client
from text_clustering.logging_config import setup_logging
from text_clustering.prompts import prompt_construct_generate_label, prompt_construct_merge_label

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint_labelgen.json"


def make_run_dir(runs_dir: str, data: str, size: str) -> str:
    """Create and return a timestamped run directory path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"{data}_{size}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote %s", path)


def _save_checkpoint(run_dir: str, processed_chunks: int, all_labels: list[str]) -> None:
    """Save label-generation progress to a checkpoint file."""
    ckpt_path = os.path.join(run_dir, CHECKPOINT_FILE)
    with open(ckpt_path, "w") as f:
        json.dump({"processed_chunks": processed_chunks, "all_labels": all_labels}, f, indent=2)
    logger.info("[checkpoint] Saved label-gen progress: %d chunks, %d labels", processed_chunks, len(all_labels))


def _load_checkpoint(run_dir: str) -> tuple[int, list[str]] | None:
    """Load label-generation checkpoint if it exists.

    Returns ``(processed_chunks, all_labels)`` or *None*.
    """
    ckpt_path = os.path.join(run_dir, CHECKPOINT_FILE)
    if not os.path.exists(ckpt_path):
        return None
    try:
        with open(ckpt_path) as f:
            data = json.load(f)
        processed = data["processed_chunks"]
        labels = data["all_labels"]
        logger.info("[checkpoint] Resuming label-gen from chunk %d (%d labels accumulated)", processed, len(labels))
        return processed, labels
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("[checkpoint] Corrupt checkpoint, starting fresh: %s", exc)
        return None


def _remove_checkpoint(run_dir: str) -> None:
    """Remove the checkpoint file after successful completion."""
    ckpt_path = os.path.join(run_dir, CHECKPOINT_FILE)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info("[checkpoint] Removed label-gen checkpoint (step complete)")


def get_sentences(sentence_list):
    return [item["input"] for item in sentence_list]


def label_generation(args, client, data_list, chunk_size, run_dir: str | None = None):
    with open(args.given_label_path, "r") as f:
        given_labels = json.load(f)

    all_labels = list(given_labels.get(args.data, []))
    count = 0

    # ── Checkpoint resume ──
    start_chunk = 0
    if run_dir:
        ckpt = _load_checkpoint(run_dir)
        if ckpt is not None:
            start_chunk, all_labels = ckpt
            count = start_chunk

    total_chunks = (len(data_list) + chunk_size - 1) // chunk_size
    ckpt_interval = max(5, total_chunks // 20)  # ~5% granularity, at least every 5 chunks

    for chunk_idx, i in enumerate(range(0, len(data_list), chunk_size)):
        # Skip already-processed chunks
        if chunk_idx < start_chunk:
            continue

        chunk = data_list[i : i + chunk_size]
        sentences = get_sentences(chunk)
        prompt = prompt_construct_generate_label(sentences, given_labels[args.data])
        raw = chat(prompt, client)
        if raw is None:
            count += 1
            continue
        count += 1
        try:
            response = eval(raw)  # noqa: S307
        except Exception:
            continue

        first_key = list(response.keys())[0]
        if isinstance(response[first_key], list):
            for label in response[first_key]:
                if "unknown_topic" in label or "new_label" in label:
                    continue
                if label not in all_labels:
                    all_labels.append(label)
        else:
            all_labels.append(response[first_key])

        if args.print_details:
            print(f"Prompt:\n{prompt}")
            print(f"Raw response: {raw}")
            print(f"Parsed: {response}")
            print(f"Label count so far: {len(all_labels)}")
            if count >= args.test_num:
                break
        elif count % 10 == 0:
            logger.debug("chunk %d — label count so far: %d", count, len(all_labels))

        # ── Checkpoint save ──
        if run_dir and (chunk_idx + 1) % ckpt_interval == 0:
            _save_checkpoint(run_dir, chunk_idx + 1, all_labels)

    # Final checkpoint save before returning
    if run_dir:
        _save_checkpoint(run_dir, total_chunks, all_labels)

    return all_labels


def _parse_merge_response(response: str) -> list[str] | None:
    """
    Parse the LLM merge response into a flat list of label strings.

    Handles two response shapes:
      - Dict:  {"merged_labels": ["a", "b", ...]}  (expected)
      - List:  ["a", "b", ...]                      (some models return flat list)

    Returns None if parsing fails entirely.
    """
    try:
        parsed = eval(response)  # noqa: S307
    except Exception:
        return None

    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        return parsed

    if isinstance(parsed, dict):
        merged = []
        for val in parsed.values():
            if isinstance(val, list):
                merged.extend(val)
        if merged:
            return merged

    return None


def merge_labels(args, all_labels, client, target_k: int | None = None):
    max_attempts = 3 if target_k else 1
    best_result = None
    best_distance = float("inf")

    for attempt in range(1, max_attempts + 1):
        prompt = prompt_construct_merge_label(
            all_labels, target_k=target_k,
        )
        response = chat(prompt, client, max_tokens=4096)
        parsed = _parse_merge_response(response)
        if parsed is None:
            logger.warning(
                "merge_labels: attempt %d/%d — could not parse response",
                attempt, max_attempts,
            )
            continue

        if target_k is None:
            return parsed

        distance = abs(len(parsed) - target_k)
        if distance < best_distance:
            best_distance = distance
            best_result = parsed

        if len(parsed) == target_k:
            logger.info(
                "merge_labels: hit target K=%d on attempt %d",
                target_k, attempt,
            )
            return parsed

        logger.info(
            "merge_labels: attempt %d/%d — got %d labels "
            "(target %d, off by %d)",
            attempt, max_attempts, len(parsed),
            target_k, distance,
        )

    if best_result is not None:
        logger.info(
            "merge_labels: returning best result with %d labels "
            "(target was %d)",
            len(best_result), target_k,
        )
        return best_result

    logger.warning(
        "merge_labels: all %d attempts failed — returning unmerged list",
        max_attempts,
    )
    return all_labels


def main(args):
    size = "large" if args.use_large else "small"

    # If an explicit --run_dir was provided (e.g. from the pre-clustering step),
    # reuse that directory instead of creating a new timestamped one.
    if getattr(args, "run_dir", None):
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = make_run_dir(args.runs_dir, args.data, size)

    setup_logging(os.path.join(run_dir, "step1_label_gen.log"))

    logger.info("=== Step 1 — Label Generation ===")
    logger.info("Dataset : %s  |  split: %s", args.data, size)
    logger.info("Target K: %s", args.target_k or "auto (no target)")
    logger.info("Run dir : %s", run_dir)
    logger.info("  (pass --run_dir %s to Step 2)", run_dir)
    start = time.time()

    reuse_labels = getattr(args, "reuse_labels", False)
    label_cache_dir = getattr(args, "label_cache_dir", None) or os.path.join(args.runs_dir, "label_cache")
    forced_k = args.target_k if hasattr(args, "target_k") and args.target_k is not None else None

    # ── Label reuse: try loading from shared cache ──
    if reuse_labels:
        from text_clustering.label_cache import list_cached, load_labels

        cached = load_labels(label_cache_dir, args.data, size, n_labels=forced_k)
        if cached is not None:
            final_labels = cached
            write_json(os.path.join(run_dir, "labels_merged.json"), final_labels)
            logger.info("[label-reuse] Loaded %d cached labels — skipping LLM generation + merge", len(final_labels))

            # Still write ground-truth labels for evaluation
            data_list = load_dataset(args.data_path, args.data, args.use_large)
            true_labels = get_label_list(data_list)
            write_json(os.path.join(run_dir, "labels_true.json"), true_labels)

            logger.info("Done in %.1fs (label reuse — 0 LLM calls)", time.time() - start)
            return
        else:
            available = list_cached(label_cache_dir, args.data, size)
            if available:
                logger.info(
                    "[label-reuse] No exact match for K=%s; available: %s — generating new labels",
                    forced_k or "any", available,
                )
            else:
                logger.info("[label-reuse] No cached labels for %s_%s — generating new labels", args.data, size)

    client = ini_client()
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    # Use a fixed seed for shuffle so checkpoint resume sees the same order.
    random.seed(42)
    random.shuffle(data_list)

    true_labels = get_label_list(data_list)
    logger.info("True cluster count: %d", len(true_labels))
    write_json(os.path.join(run_dir, "labels_true.json"), true_labels)

    all_labels = label_generation(args, client, data_list, args.chunk_size, run_dir=run_dir)
    logger.info("Labels proposed (before merge): %d", len(all_labels))
    write_json(os.path.join(run_dir, "labels_proposed.json"), all_labels)

    # target_k: only pass when explicitly requested via --target_k.
    # The paper does NOT use a target — capable models (e.g. gemini-2.0-flash) should
    # consolidate naturally.  Forcing k fills slots with spurious labels.
    final_labels = merge_labels(args, all_labels, client, target_k=forced_k)
    write_json(os.path.join(run_dir, "labels_merged.json"), final_labels)
    logger.info("Labels after merge: %d", len(final_labels))

    # Remove checkpoint once step completes successfully
    _remove_checkpoint(run_dir)

    # ── Label reuse: save to shared cache for future runs ──
    if reuse_labels:
        from text_clustering.label_cache import save_labels
        save_labels(label_cache_dir, args.data, size, final_labels)

    ratio = len(final_labels) / len(true_labels)
    if ratio > 2:
        logger.warning(
            "Merged label count (%d) is %.1fx the true class count (%d). "
            "Classification results will not be comparable to the paper baseline. "
            "Consider re-running Step 1 before proceeding to Step 2.",
            len(final_labels), ratio, len(true_labels),
        )

    logger.info("Done in %.1fs", time.time() - start)


def build_parser():
    parser = argparse.ArgumentParser(description="Step 1: LLM label generation and merge.")
    parser.add_argument("--data_path", type=str, default="./datasets/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--runs_dir", type=str, default="./runs",
                        help="Root directory where timestamped run folders are created")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Reuse an existing run directory (e.g. from K-Medoids pre-clustering). "
                             "If set, --runs_dir is ignored and no new timestamp dir is created.")
    parser.add_argument("--given_label_path", type=str, default="./runs/chosen_labels.json")
    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--print_details", type=bool, default=False)
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=15)
    parser.add_argument(
        "--target_k", type=int, default=None,
        help=(
            "Target number of labels after the merge step.  "
            "When set, the LLM is instructed to consolidate "
            "labels to exactly this count (e.g. --target_k 18 "
            "for massive_scenario).  If the first merge attempt "
            "does not hit the target, up to 3 retries are made.  "
            "Default: None (let the model consolidate naturally)."
        ),
    )
    # --api_key kept for backward compatibility but ignored; key comes from .env
    parser.add_argument("--api_key", type=str, default="", help="ignored — use OPENAI_API_KEY in .env")
    # ── Label reuse ──
    parser.add_argument(
        "--reuse_labels", action="store_true", default=False,
        help=(
            "Enable label caching.  On the first run for a dataset+split+K, "
            "generated labels are saved to a shared cache under runs/label_cache/. "
            "On subsequent runs with the same key, cached labels are loaded "
            "instead of calling the LLM again."
        ),
    )
    parser.add_argument(
        "--label_cache_dir", type=str, default=None,
        help="Directory for the shared label cache (default: <runs_dir>/label_cache).",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())


def main_cli():
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())
