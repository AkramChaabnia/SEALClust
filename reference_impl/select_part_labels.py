"""
select_part_labels.py — Step 0 entry-point shim.

All logic lives in text_clustering.pipeline.seed_labels.
This file is kept for backward compatibility with the original paper's
invocation style (``python select_part_labels.py``).
"""

from text_clustering.pipeline.seed_labels import main

if __name__ == "__main__":
    main()
