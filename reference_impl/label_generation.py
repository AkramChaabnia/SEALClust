"""
label_generation.py — Step 1 entry-point shim.

All logic lives in text_clustering.pipeline.label_generation.
This file is kept for backward compatibility with the original paper's
invocation style (``python label_generation.py --data foo``).
"""

from text_clustering.pipeline.label_generation import build_parser, main

if __name__ == "__main__":
    main(build_parser().parse_args())
