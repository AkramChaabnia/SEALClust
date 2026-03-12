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
from text_clustering.prompts import prompt_construct_classify

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.json"
CLASSIFICATIONS_FILE = "classifications.json"


def _load_medoid_documents(run_dir: str) -> list[dict]:
    """Load the medoid document subset from ``medoid_documents.jsonl`` in the run directory."""
    path = os.path.join(run_dir, "medoid_documents.jsonl")
    with open(path, "r") as f:
        docs = [json.loads(line) for line in f]
    logger.info("Loaded %d medoid documents from %s", len(docs), path)
    return docs


def get_merged_labels(run_dir: str) -> list[str]:
    path = os.path.join(run_dir, "labels_merged.json")
    with open(path, "r") as f:
        data_list = json.load(f)
    return list(set(data_list))


def load_checkpoint(run_dir: str) -> tuple[int, dict] | tuple[None, None]:
    """Return (processed_count, answer_dict) from checkpoint, or (None, None) if none exists."""
    path = os.path.join(run_dir, CHECKPOINT_FILE)
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


def known_label_categorize(args, client, data_list, label_list, run_dir):
    # Resume from checkpoint if available
    start_from, answer = load_checkpoint(run_dir)
    if answer is None:
        answer = {"Unsuccessful": []}
        for label in label_list:
            answer[label] = []
        start_from = 0

    length = args.test_num if args.print_details else len(data_list)

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

        if i % 200 == 0:
            logger.info("Progress: %d/%d", i, length)
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

    logger.info("=== Step 2 — Classification ===")
    logger.info("Dataset : %s  |  split: %s", args.data, size)
    logger.info("Run dir : %s", args.run_dir)
    if medoid_mode:
        logger.info("Mode    : MEDOID (classifying medoid subset only)")
    start = time.time()

    client = ini_client()

    # Load documents — either full dataset or medoid subset
    if medoid_mode:
        data_list = _load_medoid_documents(args.run_dir)
    else:
        data_list = load_dataset(args.data_path, args.data, args.use_large)

    label_list = get_merged_labels(args.run_dir)
    logger.info("Labels to classify into: %d", len(label_list))
    logger.info("Documents to classify  : %d", len(data_list))

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
    parser.add_argument("--data_path", type=str, default="./dataset/")
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
    # --api_key kept for backward compatibility but ignored; key comes from .env
    parser.add_argument("--api_key", type=str, default="", help="ignored — use OPENAI_API_KEY in .env")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())


def main_cli():
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())
