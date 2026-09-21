"""CPU tests for forecast-head shape paths and future-state conditioning.

These tests guard three structural properties before GPU training: seasonal
fill matches the official implementation, future states are causal, and the
optional future convolution is bit-for-bit equivalent when disabled or
initialized but untrained. A wrong implementation can still reduce training
loss, so these checks are required before experiments.

Run with: env -u PYTHONPATH .venv/bin/python script/tests/test_head_future_conv.py
"""


from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.fraccast.model import build_from_cfg                    # noqa: E402
from module.fraccast.future_conv import FutureConvStates           # noqa: E402
from module.fraccast.gather_head import GatherQuantileHead         # noqa: E402
from module.periodic.seasonal_fill import (detect_periods,         # noqa: E402
                                           folded_seasonal_fill,   # noqa: E402
                                           last_period_fill)       # noqa: E402
from util.config import load_config, strip_meta                    # noqa: E402

CFG_PATH = ROOT / "config" / "fraccast" / "pretrain_base.yaml"
B, W, H, Q = 2, 2048, 48, 9
fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'} {name}{(': ' + detail) if detail else '|'}")
    if not cond:
        fails.append(name)


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def make_cfg(**model_over):
    cfg = copy.deepcopy(strip_meta(load_config(CFG_PATH)))
    cfg["model"].update(model_over)
    return cfg


def periodic_ctx(b: int, w: int, period: int, amp: float = 1.0,
                 offset: float = 0.0, noise: float = 0.0, seed: int = 0):
    """Build a deterministic periodic series with optional additive noise."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(w).float().view(1, w)
    y = offset + amp * torch.sin(2 * torch.pi * t / period)
    if noise > 0:
        y = y + noise * torch.randn(b, w, generator=g)
    return y.expand(b, w).clone()


# Part A: seasonal folded fill
def test_part_a() -> None:
    print("[A] folded seasonal fill (parameter-free)")
    try:
        from tinycast.backbone import DilatedConvBackbone
    except ImportError:
        print("  SKIP: tinycast is not installed; official comparison omitted")
        DilatedConvBackbone = None

    # Bind the official implementation to a shell with only phase_bins. This compares
    # against the actual upstream source rather than a copied implementation.
    if DilatedConvBackbone is not None:
        for nb in (16, 8, 4):
            shell = types.SimpleNamespace(phase_bins=nb)
            ref = types.MethodType(DilatedConvBackbone._seasonal_naive, shell)
            x = periodic_ctx(B, W, 7, amp=3.0, offset=10.0, noise=0.05, seed=1)
            fut = torch.arange(W, W + H).view(1, H).expand(B, H)
            periods = torch.tensor([[7, 24, 0, 0]] * B, dtype=torch.long)
            ours = folded_seasonal_fill(x, fut, periods, phase_bins=nb)
            theirs = ref(x, fut, periods)
            check(f"A1 official implementation matches exactly with phase_bins={nb}", torch.equal(ours, theirs),
              f"max|delta|={float((ours - theirs).abs().max()):.3e}")

    # The fill must carry shape: a period-seven input correlates at lag seven.
    x = periodic_ctx(1, W, 7, amp=3.0, offset=10.0)
    fut = torch.arange(W, W + H).view(1, H)
    periods = torch.tensor([[7, 0, 0, 0]], dtype=torch.long)
    fill = folded_seasonal_fill(x, fut, periods, phase_bins=16)
    amp_in = float(x[:, -10 * 7:].std())
    amp_fill = float(fill.std())
    # Compare shifted tails directly; torch.roll would wrap around at the boundary.
    a, b = fill[0, :-7], fill[0, 7:]
    corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    check("A2 fill preserves the period-seven amplitude", amp_fill > 0.25 * amp_in,
          f"input std={amp_in:.3f}; fill std={amp_fill:.3f}")
    check("A3 fill correlates exactly at lag seven", corr > 0.999, f"corr={corr:.4f}")

    # Masking invalid observations is a deliberate extension of the official helper.
    xm = x.clone()
    xm[:, :W // 2] = 999.0  # The first half is intentionally invalid.
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :W // 2] = False
    fill_m = folded_seasonal_fill(x, fut, periods, phase_bins=16, mask=mask)
    check("A4 masked and valid-only fills agree", torch.allclose(fill, fill_m))
    fill_bad = folded_seasonal_fill(xm, fut, periods, phase_bins=16)
    check("A5 invalid points pollute an unmasked fill", not torch.allclose(fill, fill_bad))

    # A clean period-seven signal should report seven as the dominant period.
    p = detect_periods(x[:, -512:], top_k=4, min_period=2, alpha=0.05)
    check("A6 periodogram detects dominant period seven", int(p[0, 0]) == 7, f"detected={p[0].tolist()}")


# Part B: future states
def test_part_b() -> None:
    print("[B] FutureConvStates (causal dilated convolution)")
    torch.manual_seed(0)
    d, seed = 64, 128
    m = FutureConvStates(d, n_layers=4, seed=seed, kernel=3, ffn_mult=1.5).eval()
    h = torch.randn(B, W, d)
    fill = periodic_ctx(B, H, 7, amp=1.0, offset=0.0)
    pe = torch.randn(B, H, 5)
    with torch.no_grad():
        st = m(h, fill, pe)
    check("B1 state has shape [B,H,D]", tuple(st.shape) == (B, H, d), str(tuple(st.shape)))
    check("B2 zero-initialized output projection leaves states at the baseline",
          float(st.abs().max()) == 0.0)

    # After randomizing out_proj, changing fill[t] must not affect states[:t].
    nn_lin = torch.nn.Linear(d, d)
    with torch.no_grad():
        m.out_proj.weight.copy_(nn_lin.weight)
        m.out_proj.bias.copy_(nn_lin.bias)
        base = m(h, fill, pe)
        t0 = 20
        fill2 = fill.clone()
        fill2[:, t0] += 10.0
        pert = m(h, fill2, pe)
    d_pre = (pert[:, :t0] - base[:, :t0]).abs().max()
    d_post = (pert[:, t0:] - base[:, t0:]).abs().max()
    check("B3 changing fill at t does not affect earlier future positions", float(d_pre) == 0.0,
          f"t<{t0} max|delta|={float(d_pre):.3e}")
    check("B4 the perturbation reaches current and later positions", float(d_post) > 0.0,
          f"t>={t0} max|delta|={float(d_post):.3e}")
    check("B5 states vary along the horizon and are not constant",
          float(base.std(dim=1).mean()) > 0.0,
          f"per-step std={float(base.std(dim=1).mean()):.4f}")


# Part C: model integration
def _head(fc: bool):
    m = dict(d_model=64, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             horizon=H, future_queries=3, future_patch=16, head_d_ff=96,
             head_layers=1, head_fc_layers=2, head_fc_kernel=3, head_kind="gather")
    m["head_future_conv"] = fc
    torch.manual_seed(0)
    _, head = build_from_cfg(make_cfg(**m))
    return head.eval()


def test_part_c() -> None:
    print("[C] GatherQuantileHead integration")
    off, on = _head(False), _head(True)
    check("C1 disabled switch creates no future_conv module", off.future_conv is None)
    check("C2 enabled switch creates the future_conv module", on.future_conv is not None)
    check("C3 parameter count increases",
          n_params(on) > n_params(off),
          f"{n_params(off)} -> {n_params(on)} (+{n_params(on) - n_params(off)})")

    # Copy shared weights to prove that an enabled but untrained head equals the disabled head.
    sd_off = off.state_dict()
    sd_on = on.state_dict()
    shared = {k: v for k, v in sd_off.items() if k in sd_on}
    sd_on.update(shared)
    on.load_state_dict(sd_on)

    h = torch.randn(B, W, 64)
    ctx = periodic_ctx(B, W, 7, amp=3.0, offset=10.0, noise=0.05, seed=3)
    ctx2 = periodic_ctx(B, W, 7, amp=3.0, offset=10.0, noise=0.05, seed=4)
    mask = torch.ones_like(ctx, dtype=torch.bool)
    with torch.no_grad():
        q_off = off.forward_horizon(h, ctx=ctx, ctx_mask=mask)
        q_on = on.forward_horizon(h, ctx=ctx, ctx_mask=mask)
    check("C4 enabled and untrained exactly equals disabled", torch.equal(q_off, q_on),
          f"max|delta|={float((q_off - q_on).abs().max()):.3e}")

    # Disabled output is context-independent; randomized enabled output depends on context.
    with torch.no_grad():
        q_off2 = off.forward_horizon(h, ctx=ctx2, ctx_mask=mask)
        on.future_conv.out_proj.weight.normal_(0, 0.5)
        on.future_conv.out_proj.bias.zero_()
        q_on_a = on.forward_horizon(h, ctx=ctx, ctx_mask=mask)
        q_on_b = on.forward_horizon(h, ctx=ctx2, ctx_mask=mask)
    check("C5 disabled output is context-independent", torch.equal(q_off, q_off2))
    check("C6 enabled output depends on context", not torch.equal(q_on_a, q_on_b),
          f"max|delta|={float((q_on_a - q_on_b).abs().max()):.3e}")

    # Missing context must fail rather than silently degrade to the old path.
    try:
        on.forward_horizon(h)
        raised = False
    except ValueError:
        raised = True
    check("C7 missing context raises instead of degrading silently", raised)


# Part D: complete model
def test_part_d() -> None:
    print("[D] build_from_cfg / default-disabled head")
    torch.manual_seed(0)
    _, head_default = build_from_cfg(make_cfg(head_future_conv=False))
    check("D1 default configuration creates no future_conv",
          head_default.future_conv is None)
    torch.manual_seed(0)
    _, head_on = build_from_cfg(make_cfg(head_future_conv=True))
    check("D2 enabled configuration builds successfully", head_on.future_conv is not None,
          f"head parameters={n_params(head_on):,}; future_conv parameters="
          f"{n_params(head_on.future_conv):,}")


def test_part_e() -> None:
    print("[E] last-period fill (parameter-free exact periods)")
    w, p, h = 32, 7, 10
    t = torch.arange(w).float().view(1, w)
    # Constant offsets differ by period; exact copy preserves the latest phase mean.
    x = (t % p + 1.0 + 100.0 * torch.floor(t / p)).view(1, w)
    fut = torch.arange(w, w + h).view(1, h)
    periods = torch.tensor([[p]], dtype=torch.long)
    fill = last_period_fill(x, fut, periods, phase_bins=16)
    distance = fut - (w - 1)
    src = fut - ((distance + p - 1) // p) * p
    expected = x[:, src.squeeze()]
    check("E1 latest period is copied exactly", torch.equal(fill, expected),
          f"src={src.tolist()} fill={fill.tolist()}")
    phase_mean = folded_seasonal_fill(x, fut, periods, phase_bins=16)
    check("E2 phase-mean fill is a distinct mechanism", not torch.equal(fill, phase_mean),
          f"max|delta|={float((fill - phase_mean).abs().max()):.3f}")

    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, src[0, 0].item()] = False
    fill_masked = last_period_fill(x, fut, periods, phase_bins=16, mask=mask)
    check("E3 an invalid source point falls back to the valid mean", torch.equal(fill_masked[0, 0], x[mask].mean()),
          f"fallback={float(fill_masked[0, 0]):.4f}")

    torch.manual_seed(0)
    _, head_last = build_from_cfg(make_cfg(
        head_future_conv=True, head_seasonal_fill_mode="last_period"))
    check("E4 configuration is wired into the head", head_last.seasonal_fill_mode == "last_period")
    try:
        build_from_cfg(make_cfg(head_seasonal_fill_mode="bad"))
        raised = False
    except ValueError:
        raised = True
    check("E5 invalid mode raises instead of degrading silently", raised)


def main() -> None:
    test_part_e()
    test_part_a()
    test_part_b()
    test_part_c()
    test_part_d()
    if fails:
        print(f"\n[FAIL] {len(fails)} checks failed: {fails}")
        sys.exit(1)
    print("\n[OK] all forecast-head shape-path tests passed")


if __name__ == "__main__":
    main()
