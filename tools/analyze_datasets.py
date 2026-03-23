#!/usr/bin/env python3
"""
analyze_datasets.py — Generate structured JSON reports for every dataset.

Wrapper around ``text_clustering.dataset_analysis.analyze_all`` that can
be invoked directly from the repository root::

    python tools/analyze_datasets.py
    python tools/analyze_datasets.py --dataset arxiv_fine
    python tools/analyze_datasets.py --datasets datasets --assets assets

Equivalent entry-point (after ``pip install -e .``)::

    tc-analyze-datasets
    tc-analyze-datasets --dataset massive_scenario
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — ensure repo root is on sys.path so the import works even
# when invoked as `python tools/analyze_datasets.py` without pip-install.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from text_clustering.dataset_analysis import main_cli  # noqa: E402

if __name__ == "__main__":
    main_cli()
