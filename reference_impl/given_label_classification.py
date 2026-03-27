"""
given_label_classification.py — Step 2 entry-point shim.

All logic lives in text_clustering.pipeline.classification.
This file is kept for backward compatibility with the original paper's
invocation style (``python given_label_classification.py --data foo``).
"""

from text_clustering.pipeline.classification import build_parser, main

if __name__ == "__main__":
    main(build_parser().parse_args())
