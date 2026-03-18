"""
evaluation.py — Step 3 of the clustering pipeline.

Computes ACC, NMI and ARI clustering metrics using Hungarian alignment between
the ground-truth labels and the LLM-predicted labels.

Reads from and writes to the run directory produced by Steps 1 and 2:

  Input  (run_dir):
    labels_true.json        ground-truth label list  (written by Step 1)
    classifications.json    LLM predictions          (written by Step 2)

  Output (run_dir):
    results.json            { "ACC": ..., "NMI": ..., "ARI": ...,
                              "n_samples": ..., "n_clusters_true": ...,
                              "n_clusters_pred": ..., "dataset": ...,
                              "split": ..., "model": ..., "timestamp": ... }

Original source: ECNU-Text-Computing/Text-Clustering-via-LLM
Modifications:
  - moved to text_clustering.pipeline package
  - results saved to results.json (previously only printed to stdout)
  - reads from / writes to timestamped run directories
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from text_clustering.config import MODEL
from text_clustering.logging_config import setup_logging

logger = logging.getLogger(__name__)


def load_data(data_path, data, use_large):
    data_file = (
        os.path.join(data_path, data, "large.jsonl")
        if use_large
        else os.path.join(data_path, data, "small.jsonl")
    )
    with open(data_file, "r") as f:
        data_list = []
        for line in f:
            json_object = json.loads(line)
            data_list.append(json_object)
    return data_list


def load_predict_data(run_dir):
    """Load predicted classifications — prefer the propagated file if it exists."""
    full_path = os.path.join(run_dir, "classifications_full.json")
    base_path = os.path.join(run_dir, "classifications.json")
    if os.path.exists(full_path):
        logger.info("Using propagated classifications: %s", full_path)
        path = full_path
    else:
        path = base_path
    with open(path, "r") as f:
        return json.load(f)


def get_labels(data_list):
    labels = []
    for data in data_list:
        labels.append(data["label"])
    return labels


def get_predict_labels(label_data_list, predict_data_dict):
    predict_labels = []
    for label_data in label_data_list:
        sentence = label_data["input"]
        for predict_label, sentence_list in predict_data_dict.items():
            if sentence in sentence_list:
                predict_labels.append(predict_label)
                break
    return predict_labels


def convert_label_to_ids(labels):
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}  # noqa: E741
    label_ids = [label_map[l] for l in labels]  # noqa: E741
    logger.info("  samples: %d  |  clusters: %d", len(labels), n_clusters)
    return np.asarray(label_ids), n_clusters


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc


def clustering_score(y_true, y_pred):
    return {
        "ACC": clustering_accuracy_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
    }


def main(args):
    size = "large" if args.use_large else "small"
    setup_logging(os.path.join(args.run_dir, "step3_evaluation.log"))

    logger.info("=== Step 3 — Evaluation ===")
    logger.info("Dataset : %s  |  split: %s", args.data, size)
    logger.info("Run dir : %s", args.run_dir)

    label_data_list = load_data(args.data_path, args.data, args.use_large)
    labels = get_labels(label_data_list)
    logger.info("Total samples: %d", len(labels))

    predict_data_dict = load_predict_data(args.run_dir)
    predict_labels = get_predict_labels(label_data_list, predict_data_dict)

    labels = labels[: len(predict_labels)]

    logger.info("Ground truth labels:")
    y_true, n_true = convert_label_to_ids(labels=labels)
    logger.info("Predicted labels:")
    y_pred, n_pred = convert_label_to_ids(labels=predict_labels)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    score = clustering_score(y_true=y_true, y_pred=y_pred)

    logger.info("Results:")
    logger.info("  ACC : %.4f", score["ACC"])
    logger.info("  NMI : %.4f", score["NMI"])
    logger.info("  ARI : %.4f", score["ARI"])

    results = {
        "dataset": args.data,
        "split": size,
        "model": MODEL,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(labels),
        "n_clusters_true": n_true,
        "n_clusters_pred": n_pred,
        "ACC": round(float(score["ACC"]), 6),
        "NMI": round(float(score["NMI"]), 6),
        "ARI": round(float(score["ARI"]), 6),
    }

    results_path = os.path.join(args.run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", results_path)

    # ── Auto-generate visualizations ──
    try:
        from text_clustering.visualization import generate_all_visualizations
        generate_all_visualizations(
            run_dir=args.run_dir,
            data_path=args.data_path,
            data_name=args.data,
            use_large=args.use_large,
        )
    except Exception as exc:
        logger.warning("Visualization failed (non-fatal): %s", exc)


def build_parser():
    parser = argparse.ArgumentParser(description="Step 3: Evaluate clustering metrics.")
    parser.add_argument("--data_path", type=str, default="./datasets/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory produced by Steps 1 and 2 (e.g. ./runs/massive_intent_small_20260220_143012)",
    )
    parser.add_argument(
        "--use_large",
        action="store_true",
        help="Use large split if set, otherwise use small split",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())


def main_cli():
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())
