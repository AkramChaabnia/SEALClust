"""
classification.py — Step 2 of the clustering pipeline.

Given the merged label set produced by label_generation.py, this script:
  1. Loads the dataset.
  2. Reads labels_merged.json from the run directory produced by Step 1.
  3. For each text, asks the LLM to pick the best matching label.
  4. Saves results to classifications.json in the same run directory.

Checkpoint / resume
-------------------
Progress is saved to checkpoint.json in the run directory every 200 samples.
If the run is interrupted, resume from where it stopped by passing the same
--run_dir flag again — the script detects the checkpoint automatically:

    tc-classify --run_dir ./runs/massive_intent_small_20260220_143012

The checkpoint file contains:
  {
    "processed": 1400,          <- number of samples already done
    "answer":    { ... }        <- accumulated classifications so far
  }

K-Medoids medoid mode
---------------------
When ``--medoid_mode`` is passed, the script loads documents from
``medoid_documents.jsonl`` inside the run directory instead of the full
dataset.  This allows the LLM to classify only the representative medoid
documents (typically ~100 instead of ~3000).

Original source: ECNU-Text-Computing/Text-Clustering-via-LLM
Modifications:
  - ini_client() wired to text_clustering.client.make_client()
  - fixed ini_client() call signature (original passed api_key arg incorrectly)
  - response_format disabled by default (LLM_FORCE_JSON_MODE=false)
  - _strip_fenced_json() strips markdown fences before eval()
  - chat() retries up to 5 times with backoff on 429 errors
  - model read from LLM_MODEL env var
  - outputs written to timestamped run directories under ./runs/
  - checkpoint/resume added (checkpoint.json every 200 samples)
  - added --medoid_mode for K-Medoids pre-clustering integration
"""

import argparse
import json
import logging
import os
import time

from text_clustering.data import load_dataset
from text_clustering.llm import chat, ini_client
from text_clustering.logging_config import setup_logging
from text_clustering.prompts import (
    prompt_construct_classify,
    prompt_construct_classify_batch,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint_classify.json"
CLASSIFICATIONS_FILE = "classifications.json"


def _load_medoid_documents(run_dir: str) -> list[dict]:
    """Load the medoid document subset from ``medoid_documents.jsonl`` in the run directory."""
    path = os.path.join(run_dir, "medoid_documents.jsonl")
    with open(path, "r") as f:
        docs = [json.loads(line) for line in f]
    logger.info("Loaded %d medoid documents from %s", len(docs), path)
    return docs


def _load_representative_documents(run_dir: str) -> list[dict]:
    """Load the GMM representative document subset from ``representative_documents.jsonl``."""
    path = os.path.join(run_dir, "representative_documents.jsonl")
    with open(path, "r") as f:
        docs = [json.loads(line) for line in f]
    logger.info("Loaded %d representative documents from %s", len(docs), path)
    return docs


def get_merged_labels(run_dir: str) -> list[str]:
    path = os.path.join(run_dir, "labels_merged.json")
    with open(path, "r") as f:
        data_list = json.load(f)
    return list(set(data_list))


def load_checkpoint(run_dir: str) -> tuple[int, dict] | tuple[None, None]:
    """Return (processed_count, answer_dict) from checkpoint, or (None, None) if none exists."""
    path = os.path.join(run_dir, CHECKPOINT_FILE)
    # Backward compatibility: also check the old name
    legacy_path = os.path.join(run_dir, "checkpoint.json")
    if not os.path.exists(path) and os.path.exists(legacy_path):
        path = legacy_path
    if not os.path.exists(path):
        return None, None
    with open(path, "r") as f:
        ckpt = json.load(f)
    logger.info("[checkpoint] Resuming from sample %d (loaded from %s)", ckpt["processed"], path)
    return ckpt["processed"], ckpt["answer"]


def save_checkpoint(run_dir: str, processed: int, answer: dict) -> None:
    path = os.path.join(run_dir, CHECKPOINT_FILE)
    with open(path, "w") as f:
        json.dump({"processed": processed, "answer": answer}, f)


def write_classifications(run_dir: str, answer: dict) -> None:
    path = os.path.join(run_dir, CLASSIFICATIONS_FILE)
    with open(path, "w") as f:
        json.dump(answer, f, indent=2)
    logger.info("Wrote %s", path)


def answer_process(response, label_list):
    label = "Unsuccessful"
    try:
        parsed = eval(response)  # noqa: S307
    except Exception:
        parsed = str(response)

    if isinstance(parsed, dict):
        for value in parsed.values():
            if value in label_list:
                return value
    else:
        for candidate in label_list:
            if candidate in parsed:
                return candidate

    return label


def _parse_batch_response(response: str, batch_size: int, label_list: list[str]) -> dict[int, str]:
    """Parse a batched classification response into {index: label} mapping.

    Expected format: {"1": "label_a", "2": "label_b", ...}

    Returns a dict mapping 1-based indices to validated labels.
    Missing or invalid entries are omitted so the caller can retry them individually.
    """
    results: dict[int, str] = {}
    if response is None:
        return results

    try:
        parsed = eval(response)  # noqa: S307
    except Exception:
        return results

    if not isinstance(parsed, dict):
        return results

    for key, value in parsed.items():
        try:
            idx = int(key)
        except (ValueError, TypeError):
            continue
        if idx < 1 or idx > batch_size:
            continue

        # Validate the label — exact match first, then substring fallback
        if isinstance(value, str):
            if value in label_list:
                results[idx] = value
            else:
                # Try substring match (e.g. "Weather" matches "Weather Inquiries")
                for candidate in label_list:
                    if value.lower() in candidate.lower() or candidate.lower() in value.lower():
                        results[idx] = candidate
                        break

    return results


def known_label_categorize_batched(args, client, data_list, label_list, run_dir, batch_size: int = 10):
    """Batched classification — sends multiple sentences per LLM call.

    For each batch of ``batch_size`` sentences, constructs a single prompt
    asking the LLM to classify all of them at once.  The response is parsed
    as an indexed JSON object.  Any sentences that fail to parse are retried
    individually as a fallback.

    This reduces LLM calls from N to approximately N/batch_size + failures.

    Parameters
    ----------
    batch_size : int
        Number of sentences per LLM call (default: 10).
    """
    # Resume from checkpoint if available
    start_from, answer = load_checkpoint(run_dir)
    if answer is None:
        answer = {"Unsuccessful": []}
        for label in label_list:
            answer[label] = []
        start_from = 0

    length = args.test_num if args.print_details else len(data_list)

    # Checkpoint interval — batch-aligned
    medoid_mode = getattr(args, "medoid_mode", False)
    representative_mode = getattr(args, "representative_mode", False)
    ckpt_interval = 1 if (medoid_mode or representative_mode) else max(batch_size * 10, 100)

    llm_calls = 0
    fallback_calls = 0

    i = start_from
    while i < length:
        batch_end = min(i + batch_size, length)
        batch_items = data_list[i:batch_end]
        batch_sentences = [item["input"] for item in batch_items]
        actual_batch_size = len(batch_sentences)

        # --- Batched LLM call ---
        prompt = prompt_construct_classify_batch(label_list, batch_sentences)
        response = chat(prompt, client)
        llm_calls += 1

        # Parse batch response
        batch_results = _parse_batch_response(response, actual_batch_size, label_list)

        # Process results and identify failures for individual retry
        failed_indices: list[int] = []  # 0-based within this batch

        for j in range(actual_batch_size):
            one_based = j + 1
            if one_based in batch_results:
                predicted = batch_results[one_based]
                sentence = batch_sentences[j]
                answer[predicted].append(sentence)
            else:
                failed_indices.append(j)

        # --- Fallback: retry failed sentences individually ---
        for j in failed_indices:
            sentence = batch_sentences[j]
            individual_prompt = prompt_construct_classify(label_list, sentence)
            individual_response = chat(individual_prompt, client)
            llm_calls += 1
            fallback_calls += 1

            if individual_response is None:
                predicted = "Unsuccessful"
            else:
                predicted = answer_process(individual_response, label_list)

            if predicted in label_list:
                answer[predicted].append(sentence)
            else:
                answer["Unsuccessful"].append(sentence)

        if args.print_details:
            print(f"--- Batch {i // batch_size + 1} (samples {i + 1}–{batch_end}) ---")
            print(f"  Parsed: {len(batch_results)}/{actual_batch_size}")
            print(f"  Failed: {len(failed_indices)} (retried individually)")

        # Advance pointer
        i = batch_end

        # Checkpoint
        if (i - start_from) % ckpt_interval == 0 or i >= length:
            logger.info("Progress: %d/%d  (LLM calls: %d, fallbacks: %d)", i, length, llm_calls, fallback_calls)
            save_checkpoint(run_dir, i, answer)
            write_classifications(run_dir, answer)

    logger.info(
        "Batched classification complete — %d LLM calls total "
        "(%d batched + %d individual fallbacks) for %d documents",
        llm_calls, llm_calls - fallback_calls, fallback_calls, length,
    )

    return answer


def known_label_categorize(args, client, data_list, label_list, run_dir):
    # Resume from checkpoint if available
    start_from, answer = load_checkpoint(run_dir)
    if answer is None:
        answer = {"Unsuccessful": []}
        for label in label_list:
            answer[label] = []
        start_from = 0

    length = args.test_num if args.print_details else len(data_list)

    # In medoid/representative mode (~100 docs, each call is expensive) → checkpoint every sample.
    # In normal mode (~3000 docs) → checkpoint every 200 to avoid I/O overhead.
    medoid_mode = getattr(args, "medoid_mode", False)
    representative_mode = getattr(args, "representative_mode", False)
    ckpt_interval = 1 if (medoid_mode or representative_mode) else 200

    for i in range(start_from, length):
        sentence = data_list[i]["input"]
        prompt = prompt_construct_classify(label_list, sentence)
        response = chat(prompt, client)

        if response is None:
            predicted = "Unsuccessful"
        else:
            predicted = answer_process(response, label_list)

        if predicted in label_list:
            answer[predicted].append(sentence)
        else:
            answer["Unsuccessful"].append(sentence)

        if args.print_details:
            print(f"--- Sample {i + 1} ---")
            print(f"Input    : {sentence}")
            print(f"Response : {response}")
            print(f"Predicted: {predicted}")

        if (i - start_from) % ckpt_interval == 0:
            logger.info("Progress: %d/%d", i + 1, length)
            save_checkpoint(run_dir, i + 1, answer)
            write_classifications(run_dir, answer)

    return answer


def describe_final_output(answer):
    for key, values in answer.items():
        logger.info("  %s: %d", key, len(values))


def main(args):
    size = "large" if args.use_large else "small"
    setup_logging(os.path.join(args.run_dir, "step2_classification.log"))

    medoid_mode = getattr(args, "medoid_mode", False)
    representative_mode = getattr(args, "representative_mode", False)

    logger.info("=== Step 2 — Classification ===")
    logger.info("Dataset : %s  |  split: %s", args.data, size)
    logger.info("Run dir : %s", args.run_dir)
    if medoid_mode:
        logger.info("Mode    : MEDOID (classifying medoid subset only)")
    elif representative_mode:
        logger.info("Mode    : REPRESENTATIVE (classifying GMM representative subset only)")
    start = time.time()

    client = ini_client()

    # Load documents — either full dataset or a compressed subset
    if medoid_mode:
        data_list = _load_medoid_documents(args.run_dir)
    elif representative_mode:
        data_list = _load_representative_documents(args.run_dir)
    else:
        data_list = load_dataset(args.data_path, args.data, args.use_large)

    label_list = get_merged_labels(args.run_dir)
    logger.info("Labels to classify into: %d", len(label_list))
    logger.info("Documents to classify  : %d", len(data_list))

    batch_size = getattr(args, "batch_size", 1)
    if batch_size > 1:
        logger.info("Classification mode    : BATCHED (batch_size=%d)", batch_size)
        logger.info("Expected LLM calls    : ~%d (vs %d unbatched)",
                     (len(data_list) + batch_size - 1) // batch_size, len(data_list))
        answer = known_label_categorize_batched(
            args, client, data_list, label_list, args.run_dir,
            batch_size=batch_size,
        )
    else:
        logger.info("Classification mode    : INDIVIDUAL (1 sentence per call)")
        answer = known_label_categorize(args, client, data_list, label_list, args.run_dir)
    answer = {k: v for k, v in answer.items() if v}  # drop empty buckets
    write_classifications(args.run_dir, answer)

    # Remove checkpoint once run completes successfully
    ckpt_path = os.path.join(args.run_dir, CHECKPOINT_FILE)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info("[checkpoint] Removed checkpoint (run complete)")

    logger.info("Classification results:")
    describe_final_output(answer)
    logger.info("Done in %.1fs", time.time() - start)


def build_parser():
    parser = argparse.ArgumentParser(description="Step 2: Classify texts into generated labels.")
    parser.add_argument("--data_path", type=str, default="./datasets/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory produced by Step 1 (e.g. ./runs/massive_intent_small_20260220_143012)",
    )
    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--print_details", type=bool, default=False)
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument(
        "--medoid_mode", action="store_true",
        help=(
            "Classify only the medoid documents (from medoid_documents.jsonl in run_dir) "
            "instead of the full dataset. Used with the K-Medoids pre-clustering step."
        ),
    )
    parser.add_argument(
        "--representative_mode", action="store_true",
        help=(
            "Classify only the GMM representative documents (from representative_documents.jsonl) "
            "instead of the full dataset. Used with the GMM pre-clustering step."
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help=(
            "Number of sentences to classify per LLM call (default: 1 = original behaviour). "
            "Set to 10-20 for batched classification which reduces LLM calls by that factor. "
            "Failed batch parses are retried individually as fallback."
        ),
    )
    # --api_key kept for backward compatibility but ignored; key comes from .env
    parser.add_argument("--api_key", type=str, default="", help="ignored — use OPENAI_API_KEY in .env")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())


def main_cli():
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())
