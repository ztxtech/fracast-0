"""Python inference wrapper for the Fracast-0 release."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from module.fracast.safetensors_io import load_w8_safetensors

_DEFAULT_MODEL_ID = "ztxtech/fracast-0"
_HF_FILES = (
    "config.json",
    "model.safetensors",
    "model.int8.safetensors",
    "w8_manifest.json",
)
_MIN_FINITE = 8
_WEIGHT_FORMATS = ("fp32", "w8")


def _resolve_model_dir(source: str | Path) -> Path:
    """Resolve a local directory or a Hugging Face model repository id."""
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve()
    if local.exists() or source in (".", ".."):
        raise NotADirectoryError(f"model source must be a directory or repo id: {source}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download models: "
            "pip install -r requirements.txt"
        ) from exc

    downloaded = snapshot_download(
        repo_id=str(source),
        allow_patterns=list(_HF_FILES),
    )
    return Path(downloaded).resolve()


def load_fracast_from_hf(
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

    from model.fracast.model import build_from_cfg

    cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    cfg = {key: cfg[key] for key in ("model", "pyramid")}
    core, head = build_from_cfg(cfg)
    core_state = {
        name.removeprefix("core."): value for name, value in state.items()
        if name.startswith("core.")
    }
    head_state = {
        name.removeprefix("head."): value for name, value in state.items()
        if name.startswith("head.")
    }
    if set(core_state) != set(core.state_dict()) or set(head_state) != set(head.state_dict()):
        raise RuntimeError("release weights do not match the model defined by config.json")
    core.load_state_dict(core_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    if device is not None:
        core.to(device=device)
        head.to(device=device)
    return core.eval(), head.eval()


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
        model_id: str | Path = _DEFAULT_MODEL_ID,
        *,
        weights: str = "w8",
        device: str | torch.device = "cpu",
    ) -> "FracastModel":
        """Load FP32 or W8 weights from a local directory or Hugging Face."""
        model_dir = _resolve_model_dir(model_id)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        core, head = load_fracast_from_hf(model_dir, weights=weights, device=device)
        return cls(core, head, cfg, device=device)

    def _predict_rows(self, rows: np.ndarray) -> np.ndarray:
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

    def forecast(self, context: np.ndarray) -> np.ndarray:
        """Forecast `[T]` or `[V, T]`; channels are processed independently."""
        values = np.asarray(context, dtype=np.float32)
        if values.ndim == 1:
            return self._predict_rows(values[None, :])[0]
        if values.ndim == 2:
            return self._predict_rows(values)
        raise ValueError(f"context supports [T] or [V,T], got {values.shape}")

    def forecast_batch(self, context: np.ndarray) -> np.ndarray:
        """Forecast an independent `[batch, time]` tensor as one GPU/CPU batch."""
        values = np.asarray(context, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"forecast_batch supports [batch,time], got {values.shape}")
        return self._predict_rows(values)
