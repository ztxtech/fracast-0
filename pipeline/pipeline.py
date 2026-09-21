"""Configuration-driven pipeline dispatch.

The only command-line entry point is ``main.py``. It loads a configuration,
expands its grid, and calls ``Pipeline(config).run()``. The pipeline then
dispatches to a flow implementation based solely on the configuration.

Pipeline flows:

- ``build_corpus``: convert raw Arrow or Parquet data into the fast corpus
  format used by training.
- ``train``: pretrain FracCast.

Model definitions belong in ``model/``, reusable components in ``module/``, data
access and conversion in ``dataport/``, and operational tooling in ``script/``.
"""
from __future__ import annotations

from typing import Any

KINDS = ("build_corpus", "train")
_ALIASES = {"train_eval": "train"}


class Pipeline:
    """Execute one configured pipeline flow."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    @property
    def kind(self) -> str:
        """Return the resolved flow type."""
        run_meta = self.config.get("_run") or {}
        raw = str(
            run_meta.get("kind")
            or (self.config.get("pipeline") or {}).get("name")
            or "train"
        )
        return _ALIASES.get(raw, raw)

    def flow_config(self) -> dict[str, Any]:
        """Return the configuration without runner metadata."""
        from util.config import strip_meta

        return strip_meta(self.config)

    def build_corpus(self) -> Any:
        """Run the corpus conversion flow."""
        from pipeline.build_corpus import run

        return run(self.flow_config())

    def train(self) -> Any:
        """Run the training flow."""
        from pipeline.train import run

        return run(self.flow_config())

    def run(self) -> Any:
        """Dispatch to the configured flow."""
        method = {
            "build_corpus": self.build_corpus,
            "train": self.train,
        }.get(self.kind)
        if method is None:
            raise NotImplementedError(
                f"Pipeline kind {self.kind!r} is not implemented; "
                f"expected one of {KINDS}"
            )
        return method()
