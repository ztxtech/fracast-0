"""FracCast encoder with one shared filter across a geometric scale ladder.

This file defines block ordering only.  Numerical operations live in
``module/fraccast``.  A single full-resolution stream is projected, processed
at dilations 1, 2, 4, ..., optionally conditioned on the continuous scale
coordinate, and normalized.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from module.fraccast.gather_head import GatherQuantileHead
from module.fraccast.self_similar_block import ScaleCondition, SelfSimilarBlock
from module.periodic.encoder import PeriodicPhaseEncoder
from module.periodic.official_encoding import N_RECENCY_CHANNELS, _recency_encoding


class FraccastCore(nn.Module):
    """Encode a full-resolution context stream into ``[B, L, D]``."""

    def __init__(self, cfg: dict):
        super().__init__()
        m, p = cfg["model"], cfg["pyramid"]
        self.d = int(m["d_model"])
        self.W = int(m["W"])
        self.n_stages = int(m.get("n_stages", 10))
        self.dilation_base = int(m.get("dilation_base", 2))
        self.share_stages = bool(m.get("share_stages", True))
        self.scale_cond_kind = str(m.get("scale_cond", "film"))
        self.use_recency = bool(m.get("recency_encoding", True))
        self.periodic_phase = bool(m.get("periodic_phase", True))
        # Multi-scale structure is represented by the dilation ladder.
        ratios = list(p.get("ratios", []) or [])
        self.level_widths = [int(w) for w in (m.get("level_widths") or [self.W])]
        if ratios:
            raise ValueError(
                "FracCast requires a single full-resolution context; "
                f"pyramid.ratios must be empty, got {ratios}"
            )
        if len(self.level_widths) != 1:
            raise ValueError(
                f"level_widths must contain one full-resolution level; "
                f"got {self.level_widths}"
            )
        if self.level_widths[0] != self.W:
            raise ValueError(
                f"level_widths[0]={self.level_widths[0]} must equal "
                f"model.W={self.W}"
            )

        n_value_ch = 2 + (N_RECENCY_CHANNELS if self.use_recency else 0)
        self.in_proj = nn.Linear(n_value_ch, self.d)
        # Optional zero-parameter periodogram phase features.
        self.periodic = PeriodicPhaseEncoder(cfg) if self.periodic_phase else None
        kernel = int(m.get("kernel_size", 3))
        ffn_mult = float(m.get("ffn_mult", 1.5))
        causal = bool(m.get("causal", True))
        separable = bool(m.get("separable_conv", True))
        # Optional parameter sharing across stages.
        if self.share_stages:
            self.block = SelfSimilarBlock(self.d, kernel, ffn_mult, causal, separable)
            self.blocks = None
        else:
            self.block = None
            self.blocks = nn.ModuleList([
                SelfSimilarBlock(self.d, kernel, ffn_mult, causal, separable,
                                 dilation=self.dilations()[i])
                for i in range(self.n_stages)])
        # Optional per-scale FiLM conditioning.
        self.scale = (ScaleCondition(self.d, int(m.get("scale_cond_dim", 8)))
                      if self.scale_cond_kind == "film" else None)
        self.norm = nn.RMSNorm(self.d)

    def dilations(self) -> list[int]:
        """Return the geometric dilation ladder."""
        return [self.dilation_base ** i for i in range(self.n_stages)]

    def forward(self, values: torch.Tensor, mask: torch.Tensor, cov: torch.Tensor,
                **kw) -> torch.Tensor:
        """Map ``values/mask/cov`` from ``[B, 1, W]`` to ``[B, W, D]``."""
        del kw
        if values.dim() != 3 or values.shape[1] != 1:
            raise ValueError(
                f"FracCast expects a single level [B,1,W], got {tuple(values.shape)}"
            )
        if values.shape[2] != self.W:
            raise ValueError(
                f"context length {values.shape[2]} does not match model.W={self.W}"
            )
        v = torch.nan_to_num(values[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
        m = mask[:, 0].bool()
        c = cov[:, 0].to(v.dtype)
        v = torch.where(m, v, torch.zeros_like(v))
        B, L = v.shape
        # Input features are value, coverage, and optional recency channels.
        feats = [v.unsqueeze(-1), c.unsqueeze(-1)]
        if self.use_recency:
            pos = torch.arange(L, device=v.device).unsqueeze(0).expand(B, -1)
            feats.append(_recency_encoding(pos, L).to(v.dtype))
        x = self.in_proj(torch.cat(feats, dim=-1))
        if self.periodic is not None:
            x = x + self.periodic(v, m, self.level_widths)
        for i, dil in enumerate(self.dilations()):
            xi = x if self.scale is None else self.scale(x, dil)
            x = self.block(xi, dil) if self.share_stages else self.blocks[i](xi, dil)
        return self.norm(x)


class Model(nn.Module):
    """Public model wrapper that receives all parameters in one config object."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        cfg = getattr(configs, "cfg", None) or {"model": {}, "pyramid": {}}
        self.core = FraccastCore(cfg)
        self.head = build_head(cfg)

    def forward(self, values, mask, cov, **kw):
        return self.core(values, mask, cov, **kw)


def build_head(cfg: dict) -> nn.Module:
    """Build the forecast head selected by the configuration."""
    m = cfg["model"]
    # The future-convolution head requires one real time series per sample.
    if bool(m.get("head_future_conv", False)) and list(
            (cfg.get("pyramid") or {}).get("ratios") or []):
        raise ValueError(
            "model.head_future_conv requires a single full-resolution context "
            "(FracCast with empty pyramid.ratios); flattened pyramid tokens are "
            "not a time series"
        )
    kind = str(m.get("head_kind", "gather"))
    if kind != "gather":
        raise ValueError(
            f"unsupported model.head_kind: {kind!r}; "
            "this release ships only the gather head")
    args = (int(m["d_model"]), list(m["quantiles"]), int(m["horizon"]),
            int(m["future_queries"]), int(m["future_patch"]))
    return GatherQuantileHead(
        *args, int(m["head_d_ff"]), int(m.get("head_layers", 1)),
        int(m.get("head_fc_layers", 2)), int(m.get("head_fc_kernel", 3)),
        str(m.get("head_output_mode", "direct")),
        future_conv=bool(m.get("head_future_conv", False)),
        future_conv_layers=int(m.get("head_future_conv_layers", 6)),
        future_conv_seed=int(m.get("head_future_conv_seed", 128)),
        phase_bins=int(m.get("head_phase_bins", 16)),
        period_topk=int(m.get("periodic_topk", 4)),
        period_min=int(m.get("periodic_min_period", 2)),
        period_alpha=float(m.get("periodic_alpha", 0.05)),
        ffn_mult=float(m.get("ffn_mult", 1.5)),
        kernel=int(m.get("kernel_size", 3)),
        dilation_base=int(m.get("dilation_base", 2)),
        seasonal_fill_mode=str(m.get("head_seasonal_fill_mode", "phase_mean")))


def as_configs(cfg: dict):
    from types import SimpleNamespace

    ns = SimpleNamespace()
    for key, value in cfg.items():
        setattr(ns, key, value)
    ns.cfg = cfg
    return ns


def build_from_cfg(cfg: dict):
    """Pipeline entry point returning ``(core, head)``."""
    model = Model(as_configs(cfg))
    return model.core, model.head
