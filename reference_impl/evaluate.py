"""
evaluate.py — Step 3 entry-point shim.

All logic lives in text_clustering.pipeline.evaluation.
This file is kept for backward compatibility with the original paper's
invocation style (``python evaluate.py --data foo --predict_file ...``).
"""

from text_clustering.pipeline.evaluation import build_parser, main

if __name__ == "__main__":
    main(build_parser().parse_args())
