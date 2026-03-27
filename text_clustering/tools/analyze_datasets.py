#!/usr/bin/env python3
"""
analyze_datasets.py — Generate structured JSON reports for every dataset.

Wrapper around ``text_clustering.dataset_analysis.analyze_all`` that can
be invoked directly from the repository root::

    python -m text_clustering.tools.analyze_datasets
    python -m text_clustering.tools.analyze_datasets --dataset arxiv_fine
    python -m text_clustering.tools.analyze_datasets --datasets datasets --assets assets

Equivalent entry-point (after ``pip install -e .``)::

    tc-analyze-datasets
    tc-analyze-datasets --dataset massive_scenario
"""

from __future__ import annotations

from text_clustering.dataset_analysis import main_cli  # noqa: E402

if __name__ == "__main__":
    main_cli()
