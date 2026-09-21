"""CPU unit tests for FracCast structure and numerical behavior.

The central claim is that one set of block weights serves every scale. The
tests check parameter sharing, shapes, causality, conditioning initialization,
gradients, parameter counts, and context extension before GPU training.

Run with: env -u PYTHONPATH .venv/bin/python script/tests/test_fraccast_unit.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.fraccast.model import build_from_cfg            # noqa: E402
from util.config import load_config, strip_meta            # noqa: E402

CFG_PATH = ROOT / "config" / "fraccast" / "pretrain_base.yaml"
B, W, H, Q = 2, 2048, 48, 9
fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  {mark} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def make_cfg(**model_over):
    cfg = strip_meta(load_config(CFG_PATH))
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(model_over)
    return cfg


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def build(**model_over):
    cfg = make_cfg(**model_over)
    torch.manual_seed(0)
    core, head = build_from_cfg(cfg)
    return cfg, core.eval(), head.eval()


def rand_input(b: int = B, w: int = W, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(b, 1, w, generator=g)
    mask = torch.ones(b, 1, w, dtype=torch.bool)
    mask[0, 0, : w // 8] = False  # Include an unobserved prefix.
    cov = mask.to(torch.float32)
    return values, mask, cov


def main() -> int:
    print("1. structure: shared weights versus per-level weights")
    _, core_t, _ = build(share_stages=True, scale_cond="film")
    _, core_u, _ = build(share_stages=False)
    keys_t = {k.split(".")[0] for k in core_t.state_dict()}
    keys_u = {k.split(".")[0] for k in core_u.state_dict()}
    check("shared core has one block, not a blocks module", "block" in keys_t and "blocks" not in keys_t)
    check("unshared core has indexed blocks", "blocks" in keys_u and "block" not in keys_u)
    check("shared core has fewer parameters than the unshared core",
          n_params(core_t) < n_params(core_u),
          f"{n_params(core_t):,} < {n_params(core_u):,}")
    n_block = n_params(core_t.block)
    check("unshared parameter count equals shared count plus extra block copies",
          abs(n_params(core_u) - (n_params(core_t) + 9 * n_block)) <= 2 * 64,
          f"diff={n_params(core_u) - n_params(core_t) - 9 * n_block}")

    print("2. forward shapes and numerical health")
    values, mask, cov = rand_input()
    _, _, head_t = build(share_stages=True, scale_cond="film",
                         head_future_conv=False)
    with torch.no_grad():
        h = core_t(values, mask, cov)
        qh = head_t.forward_horizon(h)
    check("core output has shape [B, W, D]", tuple(h.shape) == (B, W, 64), str(tuple(h.shape)))
    check("head output has shape [B, H, Q]", tuple(qh.shape) == (B, H, Q), str(tuple(qh.shape)))
    check("outputs contain no NaN or Inf", bool(torch.isfinite(h).all() and torch.isfinite(qh).all()))

    print("3. causality: changing the last point does not change earlier outputs")
    v2 = values.clone()
    v2[:, 0, -1] += 10.0
    with torch.no_grad():
        h2 = core_t(v2, mask, cov)
    d_early = (h2[:, :-1] - h[:, :-1]).abs().max().item()
    d_last = (h2[:, -1] - h[:, -1]).abs().max().item()
    check("earlier positions remain bit-for-bit unchanged", d_early == 0.0, f"max|delta|={d_early:.3e}")
    check("the final position changes, so the probe is active", d_last > 1e-6, f"max|delta|={d_last:.3e}")

    print("4. scale conditioning is exactly the identity at initialization")
    cond = core_t.scale
    x = torch.randn(2, 7, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        for i in (0, 3, 9):
            y = cond(x, core_t.dilations()[i])
            check(f"stage {i} identity", torch.equal(y, x),
                  f"max|delta|={(y - x).abs().max().item():.3e}")
        gb = cond.gamma_beta(core_t.dilations())
    check("gamma and beta initialize to zero", bool((gb == 0).all()))

    print("5. gradients: the shared block receives gradients from every level")
    cfg = make_cfg(share_stages=True, scale_cond="film")
    torch.manual_seed(0)
    core, head = build_from_cfg(cfg)
    values, mask, cov = rand_input()
    loss = core(values, mask, cov).pow(2).mean()
    loss.backward()
    check("shared convolution weights receive gradients", core.block.dw_weight.grad is not None
          and float(core.block.dw_weight.grad.abs().sum()) > 0)
    check("scale-conditioning projection receives gradients", core.scale.to_gb.weight.grad is not None
          and float(core.scale.to_gb.weight.grad.abs().sum()) > 0)
    check("input projection receives gradients", core.in_proj.weight.grad is not None
          and float(core.in_proj.weight.grad.abs().sum()) > 0)

    print("6. parameter accounting for published tables")
    rows = []
    for name, over in (("tied_d64", {"share_stages": True, "scale_cond": "film"}),
                       ("tied_d64_nofilm", {"share_stages": True, "scale_cond": "none"}),
                       ("untied_d64", {"share_stages": False}),
                       ("tied_d32", {"share_stages": True, "scale_cond": "film",
                                     "d_model": 32, "head_d_ff": 48})):
        _, c, hd = build(**over)
        rows.append((name, n_params(c), n_params(hd), n_params(c) + n_params(hd)))
    for name, nc, nh, tot in rows:
        print(f"      {name:<16} core={nc:>7,}  head={nh:>6,}  total={tot:>7,}")
    tied_tot = dict((r[0], r[3]) for r in rows)["tied_d64"]
    untied_tot = dict((r[0], r[3]) for r in rows)["untied_d64"]
    check("shared model is no larger than the TinyCast 146,505 parameters", tied_tot <= 146_505,
          f"{tied_tot:,} vs 146,505")
    check("untying scales substantially and confirms the parameter saving", untied_tot > tied_tot * 2,
          f"{untied_tot:,} vs {tied_tot:,}")

    print("7. context extrapolation: adding levels adds no parameters")
    cfg_x = make_cfg(share_stages=True, scale_cond="film", n_stages=14)
    torch.manual_seed(0)
    core_x, _ = build_from_cfg(cfg_x)
    check("14 levels have the same parameter count as 10 levels", n_params(core_x) == n_params(core_t),
          f"{n_params(core_x):,} vs {n_params(core_t):,}")
    check("14 levels produce the expected dilation ladder",
          len(core_x.dilations()) == 14 and max(core_x.dilations()) == 8192,
          f"RF={1 + (core_x.block.kernel - 1) * sum(core_x.dilations())}")
    check("out-of-training scales remain finite and initialized as identity",
          bool(torch.isfinite(core_x.scale.scale_feat(2 ** 20)).all() and
               torch.equal(core_x.scale(x, 2 ** 20), x)),
          "scale_feat is finite and conditioning is the identity at tau=20")
    with torch.no_grad():
        hx = core_x(values, mask, cov)
    check("14 levels preserve the forward-output shape", tuple(hx.shape) == (B, W, 64))
    check("14 levels produce no NaN", bool(torch.isfinite(hx).all()))

    print("8. configuration guard: a non-empty pyramid is rejected")
    cfg_bad = make_cfg()
    cfg_bad["pyramid"]["ratios"] = [4, 4]
    try:
        build_from_cfg(cfg_bad)
        check("non-empty ratios raise ValueError", False, "no exception was raised")
    except ValueError as exc:
        check("non-empty ratios raise ValueError", "single full-resolution" in str(exc), str(exc)[:40])

    print(f"\nResult: {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
