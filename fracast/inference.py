"""Standalone inference API for the Fracast-0 release."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fracast._core.model import build_from_cfg
from fracast._core.safetensors_io import load_w8_safetensors

_HF_FILES = (
    "config.json",
    "model.safetensors",
    "model.int8.safetensors",
    "w8_manifest.json",
)
_MIN_FINITE = 8
_PACKAGE_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_WEIGHT_FORMATS = ("fp32", "w8")


def _resolve_model_dir(source: str | Path | None) -> Path:
    """Resolve bundled weights, a local directory, or a Hugging Face repo."""
    if source is None:
        return _PACKAGE_WEIGHTS_DIR
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve()
    if local.exists() or source in (".", ".."):
        raise NotADirectoryError(
            f"model source must be a directory or repo id: {source}"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download remote model weights"
        ) from exc

    downloaded = snapshot_download(repo_id=str(source), allow_patterns=list(_HF_FILES))
    return Path(downloaded).resolve()


def load_fracast(
    model_dir: str | Path, weights: str = "w8", device: str | torch.device | None = None
):
    """Load a strictly matched ``(core, head)`` pair from a release directory."""
    directory = Path(model_dir)
    if weights not in _WEIGHT_FORMATS:
        raise ValueError(f"weights must be one of {_WEIGHT_FORMATS}, got {weights!r}")
    if weights == "fp32":
        from safetensors.torch import load_file

        state = load_file(str(directory / "model.safetensors"), device="cpu")
    else:
        state = load_w8_safetensors(
            directory / "model.int8.safetensors", directory / "w8_manifest.json"
        )

    cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    core, head = build_from_cfg({key: cfg[key] for key in ("model", "pyramid")})
    core_state = {
        name.removeprefix("core."): value
        for name, value in state.items()
        if name.startswith("core.")
    }
    head_state = {
        name.removeprefix("head."): value
        for name, value in state.items()
        if name.startswith("head.")
    }
    if set(core_state) != set(core.state_dict()) or set(head_state) != set(
        head.state_dict()
    ):
        raise RuntimeError("release weights do not match the model config")
    core.load_state_dict(core_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    if device is not None:
        core.to(device=device)
        head.to(device=device)
    return core.eval(), head.eval()


def load_fracast_from_hf(
    model_dir: str | Path, weights: str = "w8", device: str | torch.device | None = None
):
    """Compatibility alias for the original release loader."""
    return load_fracast(model_dir, weights=weights, device=device)


class FracastModel:
    """Lightweight inference wrapper for Fracast-0."""

    def __init__(
        self,
        core: torch.nn.Module,
        head: torch.nn.Module,
        cfg: dict[str, Any],
        device: str | torch.device = "cpu",
    ):
        self.core = core.to(device=device).eval()
        self.head = head.to(device=device).eval()
        self.device = torch.device(device)
        model_cfg = cfg["model"]
        self.context_length = int(model_cfg["W"])
        self.horizon = int(model_cfg["horizon"])
        self.quantiles = [float(value) for value in model_cfg["quantiles"]]
        if getattr(self.head, "horizon", self.horizon) != self.horizon:
            raise ValueError("config horizon does not match the forecast head")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path | None = None,
        *,
        weights: str = "w8",
        device: str | torch.device = "cpu",
    ) -> "FracastModel":
        """Load bundled, local, or Hugging Face weights.

        Calling ``from_pretrained()`` without arguments uses the checkpoints
        packaged with ``fracast``.  A local directory path or Hugging Face
        repository id can be supplied instead.
        """
        model_dir = _resolve_model_dir(model_id)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        core, head = load_fracast(model_dir, weights=weights, device=device)
        return cls(core, head, cfg, device=device)

    def _predict_rows(self, rows: np.ndarray) -> np.ndarray:
        """Predict the native 48-step block from normalized context rows."""
        if rows.ndim != 2:
            raise ValueError(f"context must be [channels, time], got {rows.shape}")
        if rows.shape[0] == 0:
            raise ValueError("context must contain at least one channel")
        if rows.shape[1] < _MIN_FINITE:
            raise ValueError(f"each series needs at least {_MIN_FINITE} observations")

        tail = np.asarray(rows[:, -self.context_length:], dtype=np.float32)
        mask = np.isfinite(tail)
        if any(int(count) < _MIN_FINITE for count in mask.sum(axis=1)):
            raise ValueError(f"each series needs at least {_MIN_FINITE} finite observations")

        finite = np.where(mask, tail, np.float32(np.nan))
        loc = np.nanmin(finite, axis=1, keepdims=True).astype(np.float64)
        span = np.maximum(
            np.nanmax(finite, axis=1, keepdims=True) - loc, 1e-5
        ).astype(np.float64)
        normalized_tail = np.where(mask, (tail - loc) / span, np.float32(0.0))

        normalized = np.zeros((rows.shape[0], self.context_length), dtype=np.float32)
        observed = np.zeros((rows.shape[0], self.context_length), dtype=bool)
        start = self.context_length - tail.shape[1]
        normalized[:, start:] = normalized_tail.astype(np.float32)
        observed[:, start:] = mask

        values = torch.from_numpy(normalized[:, None, :]).to(self.device)
        valid = torch.from_numpy(observed[:, None, :]).to(self.device)
        coverage = valid.to(torch.float32)
        with torch.inference_mode():
            hidden = self.core(values, valid, coverage)
            quantiles = self.head.forward_horizon(
                hidden,
                last_obs=values[:, 0, -1:],
                ctx=values[:, 0],
                ctx_mask=valid[:, 0],
            )
        forecast = quantiles.cpu().numpy()
        expected = (rows.shape[0], self.horizon, len(self.quantiles))
        if forecast.shape != expected or not np.isfinite(forecast).all():
            raise RuntimeError(f"model returned an unexpected output: {forecast.shape}")
        return forecast * span[:, None, :] + loc[:, None, :]

    def _forecast_rows(self, rows: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast one or more blocks, re-normalizing at each block boundary."""
        if horizon is None:
            horizon = self.horizon
        if isinstance(horizon, (bool, np.bool_)) or not isinstance(
            horizon, (int, np.integer)
        ):
            raise TypeError(f"horizon must be an integer, got {type(horizon).__name__}")
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        block = self._predict_rows(rows)
        if horizon <= self.horizon:
            return block[:, :horizon]

        forecast = np.empty(
            (rows.shape[0], horizon, len(self.quantiles)), dtype=np.float32
        )
        forecast[:, : self.horizon] = block
        source = rows
        median_index = int(np.argmin(np.abs(np.asarray(self.quantiles) - 0.5)))
        produced = self.horizon
        while produced < horizon:
            median = block[:, :, median_index]
            source = np.concatenate(
                (source[:, -self.context_length :], median.astype(np.float32)),
                axis=1,
            )
            block = self._predict_rows(source)
            next_end = min(horizon, produced + self.horizon)
            forecast[:, produced:next_end] = block[:, : next_end - produced]
            produced = next_end
        return forecast

    def forecast(
        self, context: np.ndarray, *, horizon: int | None = None
    ) -> np.ndarray:
        """Forecast ``[T]`` or ``[V, T]``; channels are independent."""
        values = np.asarray(context, dtype=np.float32)
        if values.ndim == 1:
            return self._forecast_rows(values[None, :], horizon)[0]
        if values.ndim == 2:
            return self._forecast_rows(values, horizon)
        raise ValueError(f"context supports [T] or [V,T], got {values.shape}")

    def forecast_batch(
        self, context: np.ndarray, *, horizon: int | None = None
    ) -> np.ndarray:
        """Forecast an independent ``[batch, time]`` tensor as one batch."""
        values = np.asarray(context, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(
                f"forecast_batch supports [batch,time], got {values.shape}"
            )
        return self._forecast_rows(values, horizon)
