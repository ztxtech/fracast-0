"""Template for a new pipeline flow.

Copy this file to ``pipeline/<name>.py`` and implement ``run(config)``. Keep
the single-function interface and do not add ``argparse`` here; configuration
files loaded by ``main.py`` are the only source of flow parameters.

After implementing the flow, register it in ``pipeline.pipeline.KINDS`` and in
the dispatch table in ``Pipeline.run``. Keep orchestration in this file, model
construction in ``model/``, reusable components in ``module/``, and data access
in ``dataport/``.
"""
from __future__ import annotations


def run(config: dict) -> dict:
    """Execute the flow once and return metrics for the experiment record."""
    raise NotImplementedError("Implement run(config) in a concrete flow module")
