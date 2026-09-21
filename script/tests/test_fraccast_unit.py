"""FracCast 单元门（CPU 可跑全）：参数账 / 形状 / 因果性 / Δ_i identity / 结构 / 梯度 / 外推。

为什么要有这个门：
FracCast 的**主张**是一个结构性事实 —— 「一份权重服务所有尺度」；
它一旦写错（比如每级又偷偷建了一份参数、或者因果填充写反），
训练照样跑、损失照样降，但论文结论就是假的 ✗。所以这些断言必须**先绿再上卡** ✓。

用法：env -u PYTHONPATH .venv/bin/python script/tests/test_fraccast_unit.py
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
    mark = "✓" if cond else "✗"
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
    mask[0, 0, : w // 8] = False                    # 制造一段未观测（cov=0）
    cov = mask.to(torch.float32)
    return values, mask, cov


def main() -> int:
    print("① 结构：一份权重 vs 每级一份")
    _, core_t, _ = build(share_stages=True, scale_cond="film")
    _, core_u, _ = build(share_stages=False)
    keys_t = {k.split(".")[0] for k in core_t.state_dict()}
    keys_u = {k.split(".")[0] for k in core_u.state_dict()}
    check("tied 只有一个 block（没有 blocks.*）", "block" in keys_t and "blocks" not in keys_t)
    check("untied 有 blocks.0..N-1", "blocks" in keys_u and "block" not in keys_u)
    check("tied 参数严格少于 untied",
          n_params(core_t) < n_params(core_u),
          f"{n_params(core_t):,} < {n_params(core_u):,}")
    n_block = n_params(core_t.block)
    check("untied ≈ tied + (N-1)·block 参数",
          abs(n_params(core_u) - (n_params(core_t) + 9 * n_block)) <= 2 * 64,
          f"diff={n_params(core_u) - n_params(core_t) - 9 * n_block}")

    print("② 前向形状与数值健康度")
    values, mask, cov = rand_input()
    _, _, head_t = build(share_stages=True, scale_cond="film",
                         head_future_conv=False)
    with torch.no_grad():
        h = core_t(values, mask, cov)
        qh = head_t.forward_horizon(h)
    check("core 输出 [B, W, D]", tuple(h.shape) == (B, W, 64), str(tuple(h.shape)))
    check("head 输出 [B, H, Q]", tuple(qh.shape) == (B, H, Q), str(tuple(qh.shape)))
    check("输出无 NaN/Inf", bool(torch.isfinite(h).all() and torch.isfinite(qh).all()))

    print("③ 因果性：改最后一个输入点，不许影响更早位置的输出")
    v2 = values.clone()
    v2[:, 0, -1] += 10.0
    with torch.no_grad():
        h2 = core_t(v2, mask, cov)
    d_early = (h2[:, :-1] - h[:, :-1]).abs().max().item()
    d_last = (h2[:, -1] - h[:, -1]).abs().max().item()
    check("更早位置逐位不变", d_early == 0.0, f"max|Δ|={d_early:.3e}")
    check("最后位置确实变了（探针有效）", d_last > 1e-6, f"max|Δ|={d_last:.3e}")

    print("④ Δ_i 在初始化处严格 identity")
    cond = core_t.scale
    x = torch.randn(2, 7, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        for i in (0, 3, 9):
            y = cond(x, core_t.dilations()[i])
            check(f"stage {i} identity", torch.equal(y, x),
                  f"max|Δ|={(y - x).abs().max().item():.3e}")
        gb = cond.gamma_beta(core_t.dilations())
        check("γ、β 初值全 0", bool((gb == 0).all()))

    print("⑤ 梯度回路：共享块必须从每一级都收到梯度")
    cfg = make_cfg(share_stages=True, scale_cond="film")
    torch.manual_seed(0)
    core, head = build_from_cfg(cfg)
    values, mask, cov = rand_input()
    loss = core(values, mask, cov).pow(2).mean()
    loss.backward()
    check("共享块 conv 权重有梯度", core.block.dw_weight.grad is not None
          and float(core.block.dw_weight.grad.abs().sum()) > 0)
    check("Δ_i 的 to_gb 有梯度", core.scale.to_gb.weight.grad is not None
          and float(core.scale.to_gb.weight.grad.abs().sum()) > 0)
    check("in_proj 有梯度", core.in_proj.weight.grad is not None
          and float(core.in_proj.weight.grad.abs().sum()) > 0)

    print("⑥ 参数量账（写进论文表的数字必须从这里来）")
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
    check("tied 总参数 ≤ 官方 TinyCast 146,505", tied_tot <= 146_505,
          f"{tied_tot:,} vs 146,505")
    check("untied 总参数明显更大（说明共享确实省参）", untied_tot > tied_tot * 2,
          f"{untied_tot:,} vs {tied_tot:,}")

    print("⑦ 上下文外推：加级不加参数（论文 C3 的结构前提）")
    cfg_x = make_cfg(share_stages=True, scale_cond="film", n_stages=14)
    torch.manual_seed(0)
    core_x, _ = build_from_cfg(cfg_x)
    check("N=14 与 N=10 参数量完全相同", n_params(core_x) == n_params(core_t),
          f"{n_params(core_x):,} vs {n_params(core_t):,}")
    check("N=14 的 dilation 阶梯确实更长",
          len(core_x.dilations()) == 14 and max(core_x.dilations()) == 8192,
          f"RF={1 + (core_x.block.kernel - 1) * sum(core_x.dilations())}")
    check("训练外的极大尺度也有定义且不产生 NaN",
          bool(torch.isfinite(core_x.scale.scale_feat(2 ** 20)).all() and
               torch.equal(core_x.scale(x, 2 ** 20), x)),
          "τ=20 的 scale_feat 有限、初始化为 identity")
    with torch.no_grad():
        hx = core_x(values, mask, cov)
    check("N=14 前向形状不变", tuple(hx.shape) == (B, W, 64))
    check("N=14 未出现 NaN", bool(torch.isfinite(hx).all()))

    print("⑧ 配置守卫：非空 pyramid 必须报错（防止悄悄退化成金字塔模型）")
    cfg_bad = make_cfg()
    cfg_bad["pyramid"]["ratios"] = [4, 4]
    try:
        build_from_cfg(cfg_bad)
        check("ratios 非空 → ValueError", False, "没有报错 ✗")
    except ValueError as exc:
        check("ratios 非空 → ValueError", "单条全分辨率" in str(exc), str(exc)[:40])

    print(f"\n结果：{'全绿 ✓' if not fails else '失败 ' + ', '.join(fails) + ' ✗'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
