"""Configuration-driven pipeline dispatch for corpus preparation and pretraining.

Public entry point: ``from pipeline import Pipeline`` followed by
``Pipeline(config).run()``. Model code lives in ``model/``, reusable
components in ``module/``, and data access in ``dataport/``.
"""
from pipeline.pipeline import Pipeline

__all__ = ["Pipeline"]
