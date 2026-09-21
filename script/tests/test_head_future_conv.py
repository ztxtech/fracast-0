"""解码头「形状通路」单元门（CPU 可跑全）—— A 季节填充 / B 未来状态 / C 集成。

为什么要有这道门（用户 2026-09-17 要求「一块一块完成、一块一块测试」）：
本次要修的是**结构性缺陷**（解码头每个未来步看到的样本相关输入完全相同 → 只学得到趋势），
这类改动一旦写错，训练照样跑、loss 照样降，但论文结论是假的。所以每块都必须先绿再上卡：
  · A `folded_seasonal_fill` 必须与**官方源码**逐位一致（不是与我手抄的副本比 ✓）；
  · B `FutureConvStates` 必须**因果**（未来位置看不到更晚的信息）；
  · C `head_future_conv` 关掉时必须与历史实现**逐位恒等**，打开但未训练时也必须恒等
    （`fc_out` 零初始化）→ 保证「修」不会把任何在跑的实验搞坏 ✓。

用法：env -u PYTHONPATH .venv/bin/python script/tests/test_head_future_conv.py
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
    print(f"  {'✓' if cond else '✗'} {name}{('  ' + detail) if detail else ''}")
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
    """干净周期序列：[B, W]，值 = offset + amp·sin(2πt/period) + 噪声 ✓。"""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(w).float().view(1, w)
    y = offset + amp * torch.sin(2 * torch.pi * t / period)
    if noise > 0:
        y = y + noise * torch.randn(b, w, generator=g)
    return y.expand(b, w).clone()


# ─────────────────────────── Part A：季节折叠填充 ───────────────────────────
def test_part_a() -> None:
    print("[A] folded_seasonal_fill（零参数）")
    try:
        from tinycast.backbone import DilatedConvBackbone
    except ImportError:
        print("  [skip] tinycast 未安装，跳过官方源码逐位对拍")
        DilatedConvBackbone = None

    # ① 与**官方源码**逐位对拍：把官方方法绑到一个只有 phase_bins 的壳上，
    #    这样比的是官方真源码，不是我在测试里重抄一遍的实现 ✓
    if DilatedConvBackbone is not None:
        for nb in (16, 8, 4):
            shell = types.SimpleNamespace(phase_bins=nb)
            ref = types.MethodType(DilatedConvBackbone._seasonal_naive, shell)
            x = periodic_ctx(B, W, 7, amp=3.0, offset=10.0, noise=0.05, seed=1)
            fut = torch.arange(W, W + H).view(1, H).expand(B, H)
            periods = torch.tensor([[7, 24, 0, 0]] * B, dtype=torch.long)
            ours = folded_seasonal_fill(x, fut, periods, phase_bins=nb)
            theirs = ref(x, fut, periods)
            check(f"A1 与官方逐位一致（phase_bins={nb}）", torch.equal(ours, theirs),
                  f"max|Δ|={float((ours - theirs).abs().max()):.3e}")

    # ② 填充本身要真的携带形状：周期 7 的输入 → 填充在 lag-7 上应完全自相关
    x = periodic_ctx(1, W, 7, amp=3.0, offset=10.0)
    fut = torch.arange(W, W + H).view(1, H)
    periods = torch.tensor([[7, 0, 0, 0]], dtype=torch.long)
    fill = folded_seasonal_fill(x, fut, periods, phase_bins=16)
    amp_in = float(x[:, -10 * 7:].std())
    amp_fill = float(fill.std())
    # 注意：不能用 torch.roll（会绕回，边界错位）→ 用 [:-7] vs [7:] 的错位比较 ✓
    a, b = fill[0, :-7], fill[0, 7:]
    corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    check("A2 填充保留周期 7 的振幅", amp_fill > 0.25 * amp_in,
          f"输入 std {amp_in:.3f} → 填充 std {amp_fill:.3f}")
    check("A3 填充在 lag-7 上完全自相关", corr > 0.999, f"corr={corr:.4f}")

    # ③ mask：无效点不参与相位均值（官方无此能力，属登记过的必要偏离）
    xm = x.clone()
    xm[:, :W // 2] = 999.0                     # 前半段是垃圾
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :W // 2] = False
    fill_m = folded_seasonal_fill(x, fut, periods, phase_bins=16, mask=mask)
    check("A4 有 mask 时只取有效点（与无 mask 一致）", torch.allclose(fill, fill_m))
    fill_bad = folded_seasonal_fill(xm, fut, periods, phase_bins=16)
    check("A5 不 mask 会被垃圾点污染", not torch.allclose(fill, fill_bad))

    # ④ 周期检测：干净周期 7 → 主周期应检出 7
    p = detect_periods(x[:, -512:], top_k=4, min_period=2, alpha=0.05)
    check("A6 周期图检出主周期 7", int(p[0, 0]) == 7, f"detected={p[0].tolist()}")


# ─────────────────────────── Part B：未来状态 ───────────────────────────
def test_part_b() -> None:
    print("[B] FutureConvStates（因果膨胀卷积）")
    torch.manual_seed(0)
    d, seed = 64, 128
    m = FutureConvStates(d, n_layers=4, seed=seed, kernel=3, ffn_mult=1.5).eval()
    h = torch.randn(B, W, d)
    fill = periodic_ctx(B, H, 7, amp=1.0, offset=0.0)
    pe = torch.randn(B, H, 5)
    with torch.no_grad():
        st = m(h, fill, pe)
    check("B1 形状 [B,H,D]", tuple(st.shape) == (B, H, d), str(tuple(st.shape)))
    check("B2 out_proj 零初始化 ⇒ 状态恒为 0（起点等于基线 ✓）",
          float(st.abs().max()) == 0.0)

    # 随机化 out_proj 之后：因果性 —— 改 fill[:, t] 不能影响 states[:, :t]
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
    check("B3 因果：改第 t 步填充不影响更早的未来位置", float(d_pre) == 0.0,
          f"t<{t0} max|Δ|={float(d_pre):.3e}")
    check("B4 而且确实有影响（不是通路没接上）", float(d_post) > 0.0,
          f"t>={t0} max|Δ|={float(d_post):.3e}")
    check("B5 状态沿地平线确实在变（不是常数）",
          float(base.std(dim=1).mean()) > 0.0,
          f"per-step std={float(base.std(dim=1).mean()):.4f}")


# ─────────────────────────── Part C：集成 ───────────────────────────
def _head(fc: bool):
    m = dict(d_model=64, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             horizon=H, future_queries=3, future_patch=16, head_d_ff=96,
             head_layers=1, head_fc_layers=2, head_fc_kernel=3, head_kind="gather")
    m["head_future_conv"] = fc
    torch.manual_seed(0)
    _, head = build_from_cfg(make_cfg(**m))
    return head.eval()


def test_part_c() -> None:
    print("[C] GatherQuantileHead 集成")
    off, on = _head(False), _head(True)
    check("C1 开关关 ⇒ 不建 future_conv 子模块", off.future_conv is None)
    check("C2 开关开 ⇒ 建了 future_conv 子模块", on.future_conv is not None)
    check("C3 参数量增量",
          n_params(on) > n_params(off),
          f"{n_params(off)} → {n_params(on)}（+{n_params(on) - n_params(off)}）")

    # 把共享部分拷过去，验证「打开但未训练」与「关掉」逐位恒等（fc_out 零初始化）
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
    check("C4 打开但未训练 == 关掉（逐位恒等 ✓）", torch.equal(q_off, q_on),
          f"max|Δ|={float((q_off - q_on).abs().max()):.3e}")

    # 关掉时输出必须与 ctx 无关；打开并随机化 fc_out 后必须依赖 ctx，且差异随步变化
    with torch.no_grad():
        q_off2 = off.forward_horizon(h, ctx=ctx2, ctx_mask=mask)
        on.future_conv.out_proj.weight.normal_(0, 0.5)
        on.future_conv.out_proj.bias.zero_()
        q_on_a = on.forward_horizon(h, ctx=ctx, ctx_mask=mask)
        q_on_b = on.forward_horizon(h, ctx=ctx2, ctx_mask=mask)
    check("C5 关掉时输出与 ctx 无关", torch.equal(q_off, q_off2))
    check("C6 打开后输出依赖 ctx", not torch.equal(q_on_a, q_on_b),
          f"max|Δ|={float((q_on_a - q_on_b).abs().max()):.3e}")

    # 缺 ctx 必须报错（防止训练侧漏接线还照常跑）
    try:
        on.forward_horizon(h)
        raised = False
    except ValueError:
        raised = True
    check("C7 开关开但没传 ctx ⇒ 报错（不静默降级 ✓）", raised)


# ─────────────────────────── Part D：整模型 ───────────────────────────
def test_part_d() -> None:
    print("[D] build_from_cfg / 默认关闭")
    torch.manual_seed(0)
    _, head_default = build_from_cfg(make_cfg(head_future_conv=False))
    check("D1 默认配置不建 future_conv（→ 历史实验逐位不受影响 ✓）",
          head_default.future_conv is None)
    torch.manual_seed(0)
    _, head_on = build_from_cfg(make_cfg(head_future_conv=True))
    check("D2 配置打开后建得起来", head_on.future_conv is not None,
          f"整头参数 {n_params(head_on):,}（含 future_conv "
          f"{n_params(head_on.future_conv):,}）")


def test_part_e() -> None:
    print("[E] last_period_fill（零参数精确周期）")
    w, p, h = 32, 7, 10
    t = torch.arange(w).float().view(1, w)
    # 每个周期叠加不同常数：相位均值会抹平周期间跳变，精确复制必须保留尾周期。
    x = (t % p + 1.0 + 100.0 * torch.floor(t / p)).view(1, w)
    fut = torch.arange(w, w + h).view(1, h)
    periods = torch.tensor([[p]], dtype=torch.long)
    fill = last_period_fill(x, fut, periods, phase_bins=16)
    distance = fut - (w - 1)
    src = fut - ((distance + p - 1) // p) * p
    expected = x[:, src.squeeze()]
    check("E1 尾周期逐位精确复制", torch.equal(fill, expected),
          f"src={src.tolist()} fill={fill.tolist()}")
    phase_mean = folded_seasonal_fill(x, fut, periods, phase_bins=16)
    check("E2 与相位均值不同（不是同一机制重复 ✓）",
          not torch.equal(fill, phase_mean),
          f"max|Δ|={float((fill - phase_mean).abs().max()):.3f}")

    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, src[0, 0].item()] = False
    fill_masked = last_period_fill(x, fut, periods, phase_bins=16, mask=mask)
    check("E3 无效源点回退到有效均值", torch.equal(fill_masked[0, 0], x[mask].mean()),
          f"fallback={float(fill_masked[0, 0]):.4f}")

    torch.manual_seed(0)
    _, head_last = build_from_cfg(make_cfg(
        head_future_conv=True, head_seasonal_fill_mode="last_period"))
    check("E4 配置接线到解码头", head_last.seasonal_fill_mode == "last_period")
    try:
        build_from_cfg(make_cfg(head_seasonal_fill_mode="bad"))
        raised = False
    except ValueError:
        raised = True
    check("E5 非法模式报错（不静默降级 ✓）", raised)


def main() -> None:
    test_part_e()
    test_part_a()
    test_part_b()
    test_part_c()
    test_part_d()
    if fails:
        print(f"\n[FAIL] {len(fails)} 项未过：{fails}")
        sys.exit(1)
    print("\n[OK] 解码头形状通路：A/B/C/D/E 全部通过 ✓")


if __name__ == "__main__":
    main()
