"""流程：训练我们的模型 → 定期验证 → 保存 checkpoint。

**本文件没有 CLI** ✗ —— 唯一入口是根目录 `main.py`（配置驱动 ✓）：

    python main.py config/fraccast/pretrain_full.yaml
    python main.py config/fraccast/pretrain_smoke.yaml

config 键：`model.*`（结构 + 超参）、`pyramid.*`、`heads.*`、`repr.*`、`data.*`、
`train.*`（`out_dir` 必填）✓；模型侧不钉死任何可调数值 ✗（否则网格扫不动 ✓）。

断点续跑（用户 2026-09-17 定：程序必须能停，也必须能接着跑 ✓）——`train.resume`：
`auto`（默认，读 `<out_dir>/last.pt`）/ `false`（强制从头）/ `<ckpt 路径>`。
checkpoint 里存齐**权重 + 优化器动量 + 全部随机源 + 全局步 + best_val**；
恢复时数据流按 `step × grad_accum` 个 micro batch 精确跳批（sampler 的 RNG
逐位重放 ✓），既不重读已训数据，也不改后续样本顺序 ✓。
要真·从头重训：设 `train.resume: false`（或把 last.pt 挪走）✓。
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import time
from pathlib import Path

import torch

from util.config import config_summary   # noqa: E402
from module.losses.quantile import (pinball_loss,            # noqa: E402
                                     pinball_loss_mask)


def _build_model(cfg: dict):
    """Build the single pretraining model family shipped in this repository."""
    family = str((cfg.get("model") or {}).get("family", "fraccast"))
    if family != "fraccast":
        raise ValueError(f"unsupported model.family: {family!r}")
    from model.fraccast.model import build_from_cfg
    return build_from_cfg(cfg)


def _load_resume(out: Path, t_cfg: dict):
    """按 `train.resume` 找 checkpoint：返回 `(ckpt, path)`；不续跑就是 `(None, None)` ✓。"""
    spec = t_cfg.get("resume", "auto")
    if spec is None or spec is False or str(spec).lower() in (
            "false", "off", "none", "0", ""):
        return None, None
    path = (out / "last.pt") if str(spec) == "auto" else Path(str(spec))
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        print(f"[resume] {path} 不存在 → 从头训练 ✓", flush=True)
        return None, None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "core" not in ckpt:
        raise ValueError(f"[resume] {path} 不像训练 checkpoint（没有 core）✗")
    print(f"[resume] {path} step={int(ckpt.get('step') or 0):,} "
          f"chunk={ckpt.get('chunk')}"
          f" | 优化器动量={'有 ✓' if ckpt.get('optim') else '无 ✗（旧格式只能热启动）'}"
          f" | 随机源={'有 ✓' if ckpt.get('rng') else '无 ✗'}", flush=True)
    return ckpt, path


def _restore_rng(ckpt, device: str, rndmod, npmod, aug_rng) -> None:
    """还原 checkpoint 里的全部随机源（漏一个，续跑就不是原来那条轨迹 ✓）。"""
    rng = (ckpt or {}).get("rng") or {}
    if not rng:
        return
    if "python" in rng:
        rndmod.setstate(rng["python"])
    if "numpy" in rng:
        npmod.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "cuda" in rng and device == "cuda":
        torch.cuda.set_rng_state_all(rng["cuda"])
    if "aug" in rng:
        aug_rng.setstate(rng["aug"])
    print("[resume] 随机源已还原（python / numpy / torch / cuda / augment）✓",
          flush=True)


def _add_committing(loss, qh, tgt_d, copy_h, tgt_mask, q_t, T_avail,
                    cfg, device, head):
    """把官方 TinyCast 的门控 committing 项加到 pinball 损失上（train.commit_w>0 时）。

    官方来源：https://github.com/raws-labs/tinycast `tinycast/losses.py`。
    只作用于**中位数**，以 `seasonal_copy_baseline` 为参照，窗口级门控（copy 更好才罚）。
    默认 commit_w=0 → 直接返回原 loss（数值逐位不变）。
    """
    cw = float((cfg.get("train") or {}).get("commit_w", 0.0) or 0.0)
    if cw <= 0 or copy_h is None or loss is None:
        return loss
    if hasattr(head, "combined_loss"):      # StudentT 头不支持（旧分支）
        return loss
    from module.losses.tinycast import committing_loss
    q_mid = q_t.shape[0] // 2
    med2 = qh[:, :T_avail, q_mid]                       # [B,H]
    tgt2 = tgt_d[:, :T_avail]                           # [B,H]
    cp2 = copy_h[:, :T_avail].to(device)                # [B,H]
    msk2 = tgt_mask[:, :T_avail].to(device) if tgt_mask is not None else None
    return loss + committing_loss(med2, tgt2, cp2, mask=msk2, weight=cw)


def _augment_window(win, winm, sf, aug, rng):
    """官方 A.2 的四个数据增强（各 p=0.5，逐 batch 抽 ✓）。

    官方论文 A.2 给出数据增强口径，但没有发布对应实现：时间翻转 / 符号翻转 /
    整数降采样 {2,3,4} / mixup。
    我们的落地口径（官方没给细节，属于**声明过的偏离** ✓）：
      · 时间翻转 = 窗口按时间倒序（值与掩码一起翻）；
      · 符号翻转 = 整体取负；
      · 整数降采样 = 每第 k 点取一个，左侧补首值把目标段对齐回右端；
      · mixup = 与批内随机重排的自己线性混合（λ~U(0,1)），掩码取交集。
    """
    if float(aug.get("time_flip", 0.0)) > 0 and rng.random() < float(aug["time_flip"]):
        win, winm = win.flip(1), winm.flip(1)
    if float(aug.get("sign_flip", 0.0)) > 0 and rng.random() < float(aug["sign_flip"]):
        win = -win
    ks = [int(k) for k in (aug.get("downsample") or [])]
    if ks and rng.random() < float(aug.get("downsample_p", 0.0)):
        k = int(rng.choice(ks))
        if k > 1:
            kept_v, kept_m = win[:, ::k], winm[:, ::k]
            pad = int(win.shape[1] - kept_v.shape[1])
            if pad > 0:
                kept_v = torch.cat([win[:, :1].expand(-1, pad), kept_v], dim=1)
                kept_m = torch.cat(
                    [winm.new_zeros((winm.shape[0], pad)), kept_m], dim=1)
            win, winm = kept_v, kept_m
    if float(aug.get("mixup", 0.0)) > 0 and rng.random() < float(aug["mixup"]):
        lam = float(rng.random())
        perm = torch.randperm(win.shape[0], device=win.device)
        win = lam * win + (1.0 - lam) * win[perm]
        winm = winm & winm[perm]
    return win, winm, sf


def _rollout_loss(core, head, win, winm, sf, q_t, cfg, amp_scope, eps):
    """官方 TinyCast 的 AR rollout（K 块 × p 步 + scheduled sampling）✓。

    官方出处：https://github.com/raws-labs/tinycast `tinycast/train.py`
      · `_rollout_loss`：末块之前把模型自己的中位数写回上下文（概率 ε），未观测位强制写回，
        写回值 detach（是输入、不是梯度通路）✓；
      · `_chunk_loss`：单块损失 = 掩码 pinball + 门控 committing；预测 clamp ±5、目标 clamp ±10、
        上下文量程 ≤1e-4 的样本不计分；每块用**自己那段上下文**的统计量重新归一化（detach）✓。
    我们只把「归一化 + 建金字塔」换成 torch 等价实现（`module/pyramid/levels.py`）✓。
    """
    from module.pyramid.levels import build_levels, window_minmax
    from module.losses.tinycast import committing_loss, seasonal_copy_baseline

    t_cfg, m_cfg = cfg["train"], cfg["model"]
    K = max(1, int(t_cfg.get("ar_chunks", 1) or 1))
    p = int(head.horizon)                      # 单块步数 = 头一次输出的步数 ✓
    L = int(win.shape[1]) - K * p
    ratios = list(cfg["pyramid"]["ratios"])
    # 逐级 token 宽度：标量（全级同宽）或 `model.level_widths` 列表（粗级在前）✓
    width = m_cfg.get("level_widths", m_cfg["W"])
    n_levels = len(ratios) + 1
    commit_w = float(t_cfg.get("commit_w", 0.0) or 0.0)
    pred_clamp = float(t_cfg.get("pred_clamp", 5.0))
    tgt_clamp = float(t_cfg.get("target_clamp", 10.0))
    min_range = float(t_cfg.get("min_range", 1e-4))
    q_mid = int(q_t.shape[0]) // 2
    qv = q_t.view(1, 1, -1)
    n_q = float(q_t.shape[0])

    total = win.new_zeros(())
    for k in range(K):
        s = k * p
        ctx, ctxm = win[:, s:s + L], winm[:, s:s + L]
        vmin, vrange = window_minmax(ctx)
        ctxn = torch.where(ctxm, (ctx - vmin) / vrange, torch.zeros_like(ctx))
        with amp_scope():
            lv, lm, lc = build_levels(ctxn, ctxm, ratios, width, n_levels)
            h = core(lv, lm, lc)
            # ctx/ctx_mask = **归一化后**的上下文（只有头开了 future_conv 才会用到；
            # 关掉时多传两个 kwarg 不参与任何计算 ⇒ 逐位等于历史实现 ✓）
            qh = head.forward_horizon(h, last_obs=lv[:, -1, -1].unsqueeze(-1),
                                      ctx=ctxn, ctx_mask=ctxm)
        qh = qh.clamp(-pred_clamp, pred_clamp)
        tgt, tgtm = win[:, L + s:L + s + p], winm[:, L + s:L + s + p]
        tgtn = ((tgt - vmin) / vrange).clamp(-tgt_clamp, tgt_clamp)
        err = tgtn.unsqueeze(-1) - qh
        per = torch.maximum(qv * err, (qv - 1.0) * err)
        obs = tgtm.to(per.dtype).unsqueeze(-1)
        n_obs = tgtm.sum(dim=1).clamp(min=1.0)
        per_s = (per * obs).sum(dim=(1, 2)) / (n_obs * n_q)
        med = qh[..., q_mid]
        if commit_w > 0.0:
            copy_raw = seasonal_copy_baseline(ctx, p, sf)
            per_s = per_s + committing_loss(
                med, tgtn, (copy_raw - vmin) / vrange, tgtm,
                weight=commit_w, gated=True, reduction="none")
        valid = ((vrange.squeeze(-1) > min_range)
                 & torch.isfinite(qh).all(dim=(1, 2))
                 & (tgtm.sum(dim=1) > 0)).to(per_s.dtype)
        total = total + (torch.nan_to_num(per_s, 0.0, 0.0, 0.0) * valid).sum() \
            / valid.sum().clamp(min=1.0)
        if k < K - 1:
            med_raw = med * vrange + vmin
            use_pred = ((torch.rand(win.shape[0], 1, device=win.device) < eps)
                        | (~tgtm))
            fed = torch.where(use_pred, med_raw, tgt)
            win = torch.cat([win[:, :L + s], fed.detach(), win[:, L + s + p:]],
                            dim=1)
    return total / float(K)


def _average_checkpoints(out: Path, n_avg: int, cfg: dict):
    """最后 n 个周期 ckpt 均匀平均（官方收尾口径 ✓）；返回写出路径或 None。

    官方出处：https://github.com/raws-labs/tinycast `tinycast/train.py` 与
    `tinycast/export.py`；released 权重 = 最后 8 个周期 ckpt 的均匀平均。
    """
    cks = sorted(out.glob("chunk*.pt"))
    if len(cks) < 2:
        return None
    cks = cks[-max(1, int(n_avg)):]
    core_sum, head_sum, n = {}, {}, 0
    for pth in cks:
        # 本地训练产物包含 NumPy 标量；PyTorch 2.6 默认 safe-load 会拒绝。
        st = torch.load(pth, map_location="cpu", weights_only=False)
        for k, v in st["core"].items():
            core_sum[k] = core_sum.get(k, torch.zeros_like(v)) + v.float()
        for k, v in st["head"].items():
            head_sum[k] = head_sum.get(k, torch.zeros_like(v)) + v.float()
        n += 1
    ck0 = torch.load(cks[0], map_location="cpu", weights_only=False)
    avg_core = {k: (v / n).to(ck0["core"][k].dtype) for k, v in core_sum.items()}
    avg_head = {k: (v / n).to(ck0["head"][k].dtype) for k, v in head_sum.items()}
    dst = out / "averaged.pt"
    torch.save(dict(step=int(ck0.get("step", 0)), averaged=n,
                    sources=[p.name for p in cks],
                    core=avg_core, head=avg_head, config=cfg), dst)
    return dst


def evaluate(core, head, loader, device, q_t, max_batches: int = 20,
             shard_mode: bool = False, use_anchor: bool = False) -> float:
    """验证集抽样评估（max_batches 个 batch），全量留给正式评测。"""
    # 必须与训练用同一条头：horizon head 走 forward_horizon，否则评的是没训练的分支。
    core.eval()
    tot, n = 0.0, 0
    is_st = hasattr(head, "combined_loss")   # StudentT 分布头
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            if shard_mode:
                # 12 字段 = 附带季节复制参照；11 字段 = 旧格式（向后兼容）
                if len(batch) >= 12:
                    (xn, xm, xc, lids, xt, xta, tgt, loc, scale,
                     anchor, tgt_mask, copy_h) = batch[:12]
                else:
                    (xn, xm, xc, lids, xt, xta, tgt, loc, scale,
                     anchor, tgt_mask) = batch
                    copy_h = None
            else:
                (xn, xm, xc, lids, xt, xta, tgt, loc, scale, anchor) = batch
                tgt_mask = None
            h = core(xn.to(device), xm.to(device), xc.to(device),
                     level_ids=lids.to(device), ts_norm=xt.to(device),
                     t_abs=xta.to(device))
            # multi_horizon: 目标是多步段 [B, H]，用末列（最远期）做单点评估
            if tgt.dim() == 2 and tgt.shape[1] > 1:
                tgt_eval = tgt[:, -1]
                mask_eval = tgt_mask[:, -1] if tgt_mask is not None else None
            else:
                tgt_eval = tgt
                mask_eval = tgt_mask
            if is_st:
                lo_d = xn.reshape(xn.shape[0], -1)[:, -1].unsqueeze(-1).to(device)
                df_, mu_, sc_ = head.forward_horizon(h, last_obs=lo_d)
                nll = head.nll_loss(tgt_eval.to(device), df_[:, -1],
                                    mu_[:, -1], sc_[:, -1])
                loss = nll
            elif getattr(head, "horizon", 0) > 0:
                # 与训练一致：horizon head 一次性输出整段 H 步分位数
                lo_d = xn.reshape(xn.shape[0], -1)[:, -1].unsqueeze(-1).to(device)
                # 头的 ctx 口径 = 与编码器输入同一归一化空间的上下文（reshape 不改元素序 ✓）
                _ctx = xn.reshape(xn.shape[0], -1).to(device)
                _cm = xm.reshape(xm.shape[0], -1).to(device)
                if use_anchor:
                    qh = head.forward_horizon(h, last_obs=None,
                                              anchor=anchor.to(device),
                                              ctx=_ctx, ctx_mask=_cm)
                else:
                    qh = head.forward_horizon(h, last_obs=lo_d,
                                              ctx=_ctx, ctx_mask=_cm)
                H = min(qh.shape[1], tgt.shape[-1])
                # 整段 horizon 的 masked pinball（比只看末点稳定得多）
                qf = qh[:, :H].reshape(-1, q_t.shape[0])
                tf = tgt[:, :H].reshape(-1).to(device)
                mf = (tgt_mask[:, :H].reshape(-1).to(device)
                      if tgt_mask is not None else None)
                loss = pinball_loss_mask(qf, tf, q_t, mf)
            else:
                pred = head(h[:, -1])
                loss = pinball_loss_mask(pred, tgt_eval.to(device), q_t,
                                         mask_eval.to(device)
                                         if mask_eval is not None else None)
            tot += loss.mean().item() * len(tgt)
            n += len(tgt)
    core.train()
    return tot / max(n, 1)


def run(config: dict) -> dict:
    """按 config 训练一次，返回数值摘要（会进该 run 的 metrics.json ✓）。

    调用链：main.py → `Pipeline(config).run()` → 本函数 ✓ —— 参数全在 config 里，
    本文件没有 main / 没有 argparse ✗（用户 2026-09-13 定）。
    """
    cfg = dict(config)

    # ★ 显式种子（2026-09-12「噪声底危机」修复）
    # 背景：此前只有 data.seed 控制**数据顺序**（dataset 内部用 default_rng），
    #   而 torch 的 RNG（权重初始化、dropout）**从未设种子** → 同配方重复极差实测 0.0761(6.7%)，
    #   且实验不可复现。这里统一固定 random/numpy/torch，并把实际 seed 记进 config_used。
    import random as _rndmod
    import numpy as _npmod
    _seed = int(cfg.get("train", {}).get("seed", cfg.get("data", {}).get("seed", 42)))
    _rndmod.seed(_seed)
    _npmod.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    cfg.setdefault("train", {})["seed_used"] = _seed
    print(f"  [seed] random/numpy/torch 固定为 {_seed}（train.seed 优先，回退 data.seed）", flush=True)

    t_cfg = cfg["train"]
    out = Path(t_cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # ── 断点续跑（用户 2026-09-17 定：必须能停、也必须能接着跑 ✓）──
    # 恢复四件套：权重 / 优化器动量 / 全局步 / 随机源；数据流另按 step 精确跳批 ✓
    resume_ckpt, resume_path = _load_resume(out, t_cfg)
    resume_step = int((resume_ckpt or {}).get("step") or 0)
    if resume_path is not None:      # 写进 config_used，事后一眼能看出这条是续跑的 ✓
        t_cfg["resume_from"] = str(resume_path)
    (out / "config_used.yaml").write_text(json.dumps(cfg, indent=1))

    try:
        from torch.utils.tensorboard import SummaryWriter
        # 同 tag 重跑 = 全新实验：清旧事件，避免曲线叠加（TB 追加语义）
        if (out / "tb").exists():
            import shutil
            shutil.rmtree(out / "tb")
        writer = SummaryWriter(log_dir=str(out / "tb"))
        try:
            m = cfg["model"]
            # 实验说明卡（TB Text 页签呈现）：ID / 配方 / 验证问题
            note = (t_cfg.get("note", "")
                    or f"实验 {out.name}：{cfg['train'].get('out_dir','')}")
            writer.add_text(
                "exp/note",
                (f"**{out.name}**  |  {t_cfg.get('note', '')}\n\n"
                 f"配方: family={m.get('family', 'fractal')} "
                 f"d={m.get('d_model')} ff={m.get('d_ff')} h={m.get('n_heads')} "
                 f"L={m.get('n_layers')} loop={m.get('loop_iters', 1)} "
                 f"| anchor_snaive={m.get('anchor_snaive', False)} "
                 f"| delta_reg={m.get('delta_reg', 0)} "
                 f"| lr_schedule={t_cfg.get('lr_schedule', 'cosine')} "
                 f"| beta2={t_cfg.get('beta2', 0.999)} "
                 f"| wd_group={t_cfg.get('wd_group', False)} "
                 f"| lr={t_cfg.get('lr', 3e-4)} "
                 f"| max_per_domain={cfg['data'].get('max_per_domain')} "
                 f"| steps={t_cfg['total_steps']} "
                 f"| head={cfg['heads']['forecast'].get('type', 'quantile')}"),
                0)
            # 旧版兼容：config 也写入
            writer.add_text(
                "exp/config",
                (f"family={m.get('family', 'fractal')} "
                 f"d={m.get('d_model')} ff={m.get('d_ff')} h={m.get('n_heads')} "
                 f"L={m.get('n_layers')} loop={m.get('loop_iters', 1)} "
                 f"| anchor_snaive={m.get('anchor_snaive', False)} "
                 f"| delta_reg={m.get('delta_reg', 0)} "
                 f"| max_per_domain={cfg['data'].get('max_per_domain')} "
                 f"| steps={t_cfg['total_steps']} "
                 f"| head={cfg['heads']['forecast'].get('type', 'quantile')}"),
                0)
        except Exception:
            pass
    except Exception:
        writer = None   # TB 可选：无依赖时训练不中断
    # 设备优先级: CUDA(远端) > MPS(Mac) > CPU；CUDA_VISIBLE_DEVICES 已隔离卡
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[env] device={device} | {config_summary(cfg)}")
    amp_enabled = bool(t_cfg.get("amp", False)) and device == "cuda"

    def amp_scope():
        if amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    print(f"[amp] bf16_autocast={amp_enabled}")

    core, head = _build_model(cfg)
    core, head = core.to(device), head.to(device)

    # 续跑：权重必须在 torch.compile **之前**装 —— ckpt 里存的是剥掉 `_orig_mod.`
    # 的裸键名，装进编译后的壳子会一个都对不上 ✗（2026-09-14 踩过同类坑）。
    if resume_ckpt is not None:
        core.load_state_dict(resume_ckpt["core"], strict=True)
        head.load_state_dict(resume_ckpt["head"], strict=True)
        print(f"[resume] 权重已装载（step={resume_step:,}）✓", flush=True)

    # 存 ckpt 统一走 util.ckpt.clean_state_dict：剥掉 torch.compile 的
    # `_orig_mod.` 包装前缀，否则评测端按裸模型键名读会一个都对不上
    # （2026-09-14 事故：224 个网格 ckpt 全部退化成随机权重）。
    from util.ckpt import clean_state_dict
    # ── 速度开关（用户 2026-09-14：单进程性能优先 ✓）──
    # TF32 只影响 fp32 矩阵乘（amp 关时才有意义）：默认关，配 train.tf32: true 才开 ✓
    if device == "cuda" and bool(t_cfg.get("tf32", False)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[speed] TF32 已开（fp32 矩阵乘）", flush=True)
    if bool(t_cfg.get("compile", False)):
        # compile_mode：default / reduce-overhead（CUDA Graphs，小模型 launch-bound 时更优）/ max-autotune
        _cmode = t_cfg.get("compile_mode") or None
        core = torch.compile(core, mode=_cmode) if _cmode else torch.compile(core)
        head = torch.compile(head, mode=_cmode) if _cmode else torch.compile(head)
        print(f"[speed] torch.compile 已启用（mode={_cmode or 'default'}；首步含编译开销 ✓）",
              flush=True)
    # 固定形状训练 → 让 cudnn 自己比一遍挑最快卷积核（官方复现线同口径 ✓）；
    # 默认关：形状频繁变化的调试场景下它会反复 benchmark 反而变慢 ✗。
    if device == "cuda" and bool(t_cfg.get("cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True
        print("[speed] cudnn.benchmark 已开（固定形状选最快 kernel）", flush=True)

    # 可选 profiler（train.profile_steps=N）：只测前 N 步，结果写 <out>/profiler.txt ✓
    _prof_n = int(t_cfg.get("profile_steps") or 0)
    prof = None
    if _prof_n > 0 and device == "cuda":
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
        prof.__enter__()
    n_par = (sum(p.numel() for p in core.parameters())
             + sum(p.numel() for p in head.parameters())) / 1e6
    print(f"[model] full={n_par:.2f}M params mode={cfg['pyramid']['mode']}")

    # 数据：统一从 DataPort 取（corpus_dirs / shard_dir / 旧流式三选一在 dataport 里选 ✓，
    # 流程不碰数据格式细节 ✗；返回 shard_mode 表示 batch 是不是分片窗口格式）。
    from dataport.dataport import build_train_loaders

    accum = max(1, int(t_cfg.get("grad_accum_steps", 1)))
    # 续跑：数据流从 `step × accum` 个 micro batch 处精确接上（sampler 的 RNG 逐位
    # 重放 ✓，不重读已经训过的数据；step = 已完成的**优化步**数 ✓）。
    skip_micro = resume_step * accum

    sampling = t_cfg.get("sampling") or {}
    _policy = str(sampling.get("policy", "mixture"))
    if _policy == "rotation":
        from pipeline.policies.rotation import make_rotation_sampler
        sampler_factory = lambda ds, bs, seed, ga: make_rotation_sampler(
            ds, bs, seed, ga, cfg, skip_micro=skip_micro)
    elif _policy == "band_mix":
        # 官方 A.2 口径：按频段份额抽 batch（小语料 / 合成片不被大语料淹没 ✓）
        from pipeline.policies.band_mix import make_band_mix_sampler
        sampler_factory = lambda ds, bs, seed, ga: make_band_mix_sampler(
            ds, bs, seed, ga, cfg, skip_micro=skip_micro)
    else:
        sampler_factory = None
    tr_loader, va_loader, shard_mode = build_train_loaders(
        cfg, sampler_factory=sampler_factory, skip_micro=skip_micro)
    if skip_micro:
        print(f"[resume] 数据流跳过 {skip_micro:,} 个 micro batch "
              f"(= {resume_step:,} 步 × {accum} 累积) → 无缝接上 ✓", flush=True)

    # 一个 epoch = 跑完一轮训练集（用 loader 的真实长度 ✓）：配置不写 steps_per_epoch 就用它；
    # 只写 epochs 时由它反推 total_steps —— 手算 total_steps 很容易跟数据错位 ✗
    spe = int(t_cfg.get("steps_per_epoch") or 0) or max(
        1, math.ceil(len(tr_loader) / accum))
    if not t_cfg.get("total_steps"):
        t_cfg["total_steps"] = spe * max(1, int(t_cfg.get("epochs") or 1))
    t_cfg["steps_per_epoch"] = spe
    print(f"[plan] 1 epoch={spe:,} 步 → total_steps={t_cfg['total_steps']:,}",
          f" epochs={t_cfg['total_steps'] / spe:.2f}", flush=True)

    # decay/no-decay 参数分组（Moirai 式：LN/bias 不 weight decay）
    def _no_decay(name: str) -> bool:
        return name.endswith(".bias") or "norm" in name or "ln" in name

    if t_cfg.get("wd_group", False):
        decay_p, no_decay_p = [], []
        for nm, p in list(core.named_parameters()) + list(head.named_parameters()):
            (no_decay_p if _no_decay(nm) else decay_p).append(p)
        params = [
            {"params": decay_p, "weight_decay": t_cfg["weight_decay"]},
            {"params": no_decay_p, "weight_decay": 0.0},
        ]
    else:
        params = list(core.parameters()) + list(head.parameters())

    # 梯度裁剪必须用真实参数列表：wd_group=true 时 params 是 param-group 字典，
    # 直接传给 clip_grad_norm_ 会报 'dict' object has no attribute 'grad'。
    clip_params = list(core.parameters()) + list(head.parameters())

    # fused AdamW：CUDA 上把多张量更新并成少数 kernel（单步更快 ✓）；不支持就回退 ✗
    _fused = device == "cuda" and bool(t_cfg.get("fused_optim", True))
    try:
        opt = torch.optim.AdamW(params, lr=t_cfg["lr"],
                                betas=(0.9, t_cfg.get("beta2", 0.999)),
                                weight_decay=t_cfg["weight_decay"], fused=_fused)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(params, lr=t_cfg["lr"],
                                betas=(0.9, t_cfg.get("beta2", 0.999)),
                                weight_decay=t_cfg["weight_decay"])
    print(f"[speed] fused_optim={_fused}", flush=True)

    # 续跑：优化器动量（Adam 的一阶/二阶矩）必须一并恢复 —— 少了它，退火段会走出
    # 另一条轨迹（2026-09-17 的教训：3src_full 只存了权重，而剩下的正好是决定
    # 最终分的退火段，热启动等于换掉最要紧的那一段 ✗）。
    if resume_ckpt is not None and resume_ckpt.get("optim"):
        opt.load_state_dict(resume_ckpt["optim"])
        print("[resume] 优化器动量已恢复 ✓", flush=True)
    elif resume_ckpt is not None:
        print("[resume] 旧格式 ckpt 没有优化器动量 → 只能热启动（Adam 要几百步重新"
              "估计矩；退火段会偏离原轨迹 ✗）", flush=True)

    # ── 官方 rollout 模式（2026-09-15 用户定：除模型外全部对齐 TinyCast）──
    # 数据侧只给原始窗口 [ctx + K×p]；归一化 / 建金字塔 / 回喂都在 _rollout_loss 里做 ✓
    # 判据只看**数据格式**（raw 窗口 = 3 字段 win/winm/sf），不看块数 ✓ ——
    # 2026-09-15 修：旧写法多带一条 `ar_chunks > 1`，于是 `ar_chunks: 1`（= 单块、
    # 不做 AR rollout 的对照格）会掉进 11 字段的非 rollout 分支 → ValueError（实测 ✗）。
    # 块数由 `_rollout_loss` 的 K = max(1, ar_chunks) 决定，K=1 就是单块损失 ✓。
    rollout_on = (shard_mode
                  and str((cfg.get("data") or {}).get("window_mode", "levels")) == "raw")
    eps_max = float(t_cfg.get("scheduled_sampling_max", 0.5))
    _aug = dict(t_cfg.get("augment") or {})
    _aug_on = bool(_aug.pop("enabled", False)) and rollout_on
    _aug_rng = random.Random(int(cfg["data"].get("seed", 42)) + 7)
    # 续跑：把增强 / mixup / dropout 的随机源一并还原 ✓
    _restore_rng(resume_ckpt, device, _rndmod, _npmod, _aug_rng)
    if rollout_on:
        print(f"[rollout] AR {int(t_cfg['ar_chunks'])} 块 × {int(head.horizon)} 步 / "
              f"ε_max={eps_max} / 增强={_aug_on}（官方口径 ✓）", flush=True)

    sched = t_cfg.get("lr_schedule", "cosine")   # cosine | wsd | cosine_restarts

    def lr_at(step):
        if step < t_cfg["warmup_steps"]:
            return t_cfg["lr"] * step / t_cfg["warmup_steps"]
        if sched == "wsd":
            # Toto 式 WSD：stable plateau 到 70% 总步，然后 1-sqrt decay 到 min_lr
            stable_end = int(t_cfg["total_steps"] * t_cfg.get("wsd_stable_frac", 0.7))
            if step < stable_end:
                return t_cfg["lr"]
            p = (step - stable_end) / max(
                1, t_cfg["total_steps"] - stable_end)
            return t_cfg.get("min_lr", 1e-5) + (t_cfg["lr"] - t_cfg.get(
                "min_lr", 1e-5)) * (1 - math.sqrt(min(p, 1.0)))
        if sched == "cosine_restarts":
            # Moirai 式 cosine with warm restarts（周期 = restart_frac × 总步）
            period = max(1, int(t_cfg["total_steps"] * t_cfg.get(
                "restart_frac", 0.25)))
            p = ((step - t_cfg["warmup_steps"]) % period) / period
            return t_cfg["lr"] * 0.5 * (1 + math.cos(math.pi * p))
        p = (step - t_cfg["warmup_steps"]) / max(
            1, t_cfg["total_steps"] - t_cfg["warmup_steps"])
        return t_cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    # 分位网格：QuantileHead 用 head.q；StudentTHead 用其 q_grid
    q_t = getattr(head, "q", None)
    if q_t is None:
        q_t = head.q_grid
    q_t = q_t.to(device)
    ckpt_best = None
    best_val = (float(resume_ckpt["best_val"])
                if resume_ckpt is not None
                and resume_ckpt.get("best_val") is not None
                else float("inf"))
    step = resume_step
    t_start = time.time()
    core.train()
    if resume_step:
        print(f"[resume] 从 step {resume_step:,} 继续 → 还剩 "
              f"{int(t_cfg['total_steps']) - resume_step:,} 步（共 "
              f"{int(t_cfg['total_steps']):,}）", flush=True)

    # 单步拆解（用户 2026-09-14：先看清时间花在哪再谈优化 ✓）：
    #   data=等 batch、fwd_bwd=前向+反向、opt=优化器/裁剪/存盘；第 1 步不计入（含预热/编译 ✗）
    acc = {"data": 0.0, "fwd_bwd": 0.0, "opt": 0.0}
    t_prev = time.perf_counter()
    group_micro = 0
    group_data = 0.0
    group_fwd = 0.0
    loss_sum = torch.zeros((), device=device)

    spe = int(t_cfg.get("steps_per_epoch", 0) or 0)
    # ckpt 节奏（用户 2026-09-14：一次训完、中途按固定间隔存，
    #   不要为了看"2 轮 / 4 轮"的曲线反复从头训 ✗）：
    #   save_every_epochs: 2  → 每 2 个 epoch 存一个（2,4,6,…）
    #   save_epochs: 2,5,9    → 明确指定（两者可叠加）
    _se = t_cfg.get("save_epochs")
    save_epoch_set = ({int(x) for x in _se} if isinstance(_se, (list, tuple))
                      else {int(x) for x in str(_se or "").split(",") if x.strip()})
    _every = int(t_cfg.get("save_every_epochs") or 0)
    if _every > 0:
        _n_ep = int(math.ceil(t_cfg["total_steps"] / max(1, spe)))
        save_epoch_set |= set(range(_every, _n_ep + 1, _every))
    if save_epoch_set:
        print(f"[ckpt] 将保存 epoch {sorted(save_epoch_set)}（1 epoch={spe:,} 步）", flush=True)

    def _rng_snapshot() -> dict:
        """打包会让训练轨迹分叉的全部随机源 ✓。"""
        snap = {"torch": torch.get_rng_state(),
                "python": _rndmod.getstate(),
                "numpy": _npmod.random.get_state(),
                "aug": _aug_rng.getstate()}
        if device == "cuda":
            snap["cuda"] = torch.cuda.get_rng_state_all()
        return snap

    def _ckpt_payload(_step: int, **extra) -> dict:
        """统一 checkpoint 载荷：能停能续 = 权重 / 优化器 / 随机源 / 进度 四件套齐全 ✓。"""
        payload = dict(step=int(_step),
                       core=clean_state_dict(core),
                       head=clean_state_dict(head),
                       config=cfg,
                       optim=opt.state_dict(),
                       rng=_rng_snapshot(),
                       best_val=(float(best_val) if math.isfinite(best_val)
                                 else None),
                       grad_accum_steps=accum,
                       steps_per_epoch=spe)
        payload.update(extra)
        return payload

    while step < t_cfg["total_steps"]:
        for batch in tr_loader:
            if step >= t_cfg["total_steps"]:
                break
            if shard_mode:
                if rollout_on:
                    # 官方 rollout 口径：batch = 原始窗口 [ctx + K×p] + 掩码 + 季节 scale ✓
                    win, winm, sf = (x.to(device, non_blocking=True)
                                     for x in batch[:3])
                elif len(batch) >= 12:
                    # 12 字段 = 附带季节复制参照；11 字段 = 旧格式（向后兼容）
                    (xn, xm, xc, lids, xt, xta, tgt, loc, scale,
                     anchor, tgt_mask, copy_h) = batch[:12]
                else:
                    (xn, xm, xc, lids, xt, xta, tgt, loc, scale,
                     anchor, tgt_mask) = batch
                    copy_h = None
            else:
                (xn, xm, xc, lids, xt, xta, tgt, loc, scale,
                 anchor) = batch
                tgt_mask = None
            t_now = time.perf_counter()          # 拿到 batch 的时刻（与上一步相隔 = 等数据 ✓）
            if group_micro == 0:
                cur_lr = lr_at(step)
                for g in opt.param_groups:
                    g["lr"] = cur_lr
                opt.zero_grad(set_to_none=True)
            group_data += t_now - t_prev
            if not rollout_on:
                xn, xm, xc, lids, xt = (x.to(device, non_blocking=True)
                                        for x in (xn, xm, xc, lids, xt))
                xta = xta.to(device)
                # pin_memory + non_blocking：H2D 拷贝与 GPU 计算重叠
                xta = xta.to(device, non_blocking=True)
                tgt_d = tgt.to(device, non_blocking=True)
            if rollout_on:
                # 官方口径：ε 在前半程线性爬到 eps_max（官方 train.py 同式 ✓）
                # `scheduled_sampling_ramp_steps` 可选：把爬升区间钉到**参照线的真实区间**
                # （官方 36,621 步 → 18,310）。为什么需要：短探针（1000 步）若按自己的 total_steps
                # 算半程，第 150 步 ε=0.15，而参照线同一步只有 0.004 —— rollout 难度不同，
                # 同步数损失不可比 ✗（2026-09-15 发现的口径混淆）。缺省 0 = 老行为 ✓。
                _ramp = float(t_cfg.get("scheduled_sampling_ramp_steps") or 0.0)
                if _ramp <= 0.0:
                    _ramp = 0.5 * t_cfg["total_steps"]
                eps = eps_max * min(1.0, step / max(1.0, _ramp))
                if _aug_on:
                    win, winm, sf = _augment_window(win, winm, sf, _aug, _aug_rng)
                loss = _rollout_loss(core, head, win, winm, sf, q_t, cfg,
                                     amp_scope, eps)
            elif cfg["model"].get("multi_horizon", False) or head.horizon > 0:
                # Chronos-2 式联合多步：context → 一次性输出整段 H 步分位数
                # （不用 future-token——v7 证伪；监督密度 = H 步/样本，非 1 点）
                with amp_scope():
                    h_ctx = core(xn, xm, xc, level_ids=lids,
                                 ts_norm=xt, t_abs=xta)
                # 掩码加权池化用：把 token 级有效掩码展平 [B, N]（仅 head_pool=masked 时用）
                _tok_w = xm.reshape(xm.shape[0], -1).to(device) \
                    if getattr(head, "head_pool", "mean") == "masked" else None
                H = head.horizon if head.horizon > 0 else tgt_d.shape[-1]
                # last_obs = 归一化后的最后观测（残差结构锚点）
                # xn [B, L, W] 粗级在前展平 → 最后 token = 最细级末观测
                lo = xn.reshape(xn.shape[0], -1)[:, -1].unsqueeze(-1)  # [B, 1]
                T_avail = tgt_d.shape[-1]
                use_anchor = bool(cfg["model"].get("anchor_snaive", False))
                if use_anchor:
                    # v28a：SNaive 季节锚（per-step [B,H]）
                    with amp_scope():
                        qh = head.forward_horizon(
                            h_ctx, last_obs=None,
                            anchor=anchor[:, :H].to(device), token_weight=_tok_w)
                elif hasattr(head, "combined_loss"):
                    # StudentT 分布头（Toto 式）：NLL + Barron robust
                    with amp_scope():
                        df, mu, sc = head.forward_horizon(h_ctx, last_obs=lo)
                    qh = head.quantiles(df, mu, sc)        # [B,H,9]（delta_reg 用）
                    loss = head.combined_loss(
                        tgt_d, df[:, :T_avail], mu[:, :T_avail],
                        sc[:, :T_avail]).mean()
                else:
                    with amp_scope():
                        qh = head.forward_horizon(h_ctx, last_obs=lo,
                                                  token_weight=_tok_w)
                if not use_anchor and not hasattr(head, "combined_loss"):
                    # 目标段 [B, H]（dataset multi_horizon 分支已给连续段）
                    qf = qh[:, :T_avail].reshape(-1, q_t.shape[0])  # [B*H, Q]
                    tf = tgt_d.reshape(-1)                          # [B*H]
                    mask_f = (tgt_mask[:, :T_avail].reshape(-1).to(device)
                              if tgt_mask is not None else None)
                    loss = pinball_loss_mask(qf, tf, q_t, mask_f)
                    loss = _add_committing(loss, qh, tgt_d, copy_h, tgt_mask,
                                           q_t, T_avail, cfg, device, head)
                elif use_anchor:
                    # anchor 模式：目标仍是绝对段（预测=anchor+delta 与 tgt 比）
                    qf = qh[:, :T_avail].reshape(-1, q_t.shape[0])
                    tf = tgt_d.reshape(-1)
                    mask_f = (tgt_mask[:, :T_avail].reshape(-1).to(device)
                              if tgt_mask is not None else None)
                    loss = pinball_loss_mask(qf, tf, q_t, mask_f)
                    loss = _add_committing(loss, qh, tgt_d, copy_h, tgt_mask,
                                           q_t, T_avail, cfg, device, head)
                # delta 收缩正则：让模型默认贴近锚、仅强证据时偏离。
                # 锚 = SNaive 季节锚（use_anchor）或平线 last_obs（TiRex 式）
                dr = cfg["model"].get("delta_reg", 0.0)
                if dr > 0 and loss is not None:
                    # 中位（qh 中间列）相对锚的偏差
                    med = qh[:, :T_avail, q_t.shape[0] // 2]
                    if use_anchor:
                        ref = anchor[:, :T_avail].to(device)
                    else:
                        ref = lo.expand_as(med)
                    dev_all = (med - ref).pow(2)
                    if tgt_mask is None:
                        dev = dev_all.mean()
                    else:
                        m_dev = tgt_mask[:, :T_avail].to(device).to(dev_all.dtype)
                        denom = m_dev.sum()
                        # 判定不落 host（同 pinball_loss_mask）→ 少一个每步 GPU 同步 ✓
                        dev = torch.where(
                            denom > 0,
                            (dev_all * m_dev).sum() / denom.clamp_min(
                                torch.finfo(dev_all.dtype).tiny),
                            dev_all.mean())
                    loss = loss + dr * dev
            else:
                with amp_scope():
                    h = core(xn, xm, xc, level_ids=lids,
                             ts_norm=xt, t_abs=xta)
                pred = head(h[:, -1])
                loss = pinball_loss(pred, tgt_d, q_t)
                # DINO 式跨尺度对齐损失（可选）：相邻级表征 InfoNCE 对齐
                align_lambda = cfg["model"].get("align_lambda", 0.0)
                z_levels = getattr(core, "z_levels", None)
                if align_lambda > 0 and z_levels is not None:
                    z = torch.nn.functional.normalize(z_levels, dim=-1)
                    B, L, _ = z.shape
                    align_loss = torch.tensor(0.0, device=device)
                    n_pairs = 0
                    for l in range(L - 1):
                        # 正对：同样本相邻级；负对：批内其他样本同相邻级
                        pos = (z[:, l] * z[:, l + 1]).sum(-1)          # [B]
                        # 负对：每个样本 vs 批内所有其他样本的 l+1 级
                        sim = z[:, l] @ z[:, l + 1].T                  # [B, B]
                        sim.fill_diagonal_(float("-inf"))
                        logits = torch.cat([pos.unsqueeze(-1), sim], dim=-1)
                        labels = torch.zeros(B, dtype=torch.long, device=device)
                        align_loss = align_loss + torch.nn.functional.cross_entropy(
                            logits / 0.5, labels)   # 温度 0.5（0.1 太冷导致 align loss 爆炸）
                        n_pairs += 1
                    loss = loss + align_lambda * (align_loss / max(n_pairs, 1))
            loss_sum += loss.detach()
            (loss / accum).backward()
            t1 = time.perf_counter()
            group_fwd += t1 - t_now
            group_micro += 1
            if group_micro < accum:
                t_prev = t1
                continue
            torch.nn.utils.clip_grad_norm_(clip_params, t_cfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            t2 = time.perf_counter()
            if step > 1:        # 第 1 步含首次填充 / kernel 选择 / compile 开销 → 不计入 ✓
                acc["data"] += group_data
                acc["fwd_bwd"] += group_fwd
                acc["opt"] += t2 - t1
            t_prev = t2
            loss_avg = loss_sum / accum
            loss_sum.zero_()
            group_micro = 0
            group_data = 0.0
            group_fwd = 0.0
            if prof is not None and step >= _prof_n:
                prof.__exit__(None, None, None)
                _tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                (out / "profiler.txt").write_text(_tbl)
                print(f"[speed] profiler 前 {_prof_n} 步（已写 {out}/profiler.txt）", flush=True)
                prof = None

            # epoch 检查点必须当场存：训练结束后再循环存会把最终权重写成所有 epoch
            # （2026-09-10 发现：epoch1..4.pt 权重相同，mini97 逐 epoch 结果完全一致）
            if spe > 0 and step % spe == 0 and (step // spe) in save_epoch_set:
                _ep = step // spe
                torch.save(_ckpt_payload(step, epoch=_ep),
                           out / f"epoch{_ep}.pt")
                print(f"[ckpt] epoch{_ep} @ step {step} 已存", flush=True)

            _chunk_steps = int(
                t_cfg.get("ckpt_every_steps")
                or (sampling.get("rotation") or {}).get("chunk_steps") or 0)
            if _chunk_steps > 0 and (step % _chunk_steps == 0
                                     or step == t_cfg["total_steps"]):
                _chunk = int(math.ceil(step / _chunk_steps))
                # chunk 与 last 存同一份载荷：续跑只读 last.pt；chunk 序列留给曲线与
                # 收尾的权重平均 ✓（两份都是完整可续的，不是剪辑版）
                _pay = _ckpt_payload(step, chunk=_chunk)
                torch.save(_pay, out / f"chunk{_chunk:02d}.pt")
                torch.save(_pay, out / "last.pt")
                print(f"[ckpt] chunk{_chunk:02d} @ step {step} 已存",
                      flush=True)

            if step % t_cfg["log_every"] == 0:
                el = time.time() - t_start
                tot = max(acc["data"] + acc["fwd_bwd"] + acc["opt"], 1e-9)
                n_meas = max(step - 1, 1)
                print(f"[step {step:>6}] loss={loss_avg.item():.4f} lr={cur_lr:.2e} "
                      f"elapsed={el/60:.1f}min | data {100*acc['data']/tot:.0f}% "
                      f"fwd+bwd {100*acc['fwd_bwd']/tot:.0f}% opt {100*acc['opt']/tot:.0f}% "
                      f"| {1000*tot/n_meas:.0f} ms/step "
                      f"{t_cfg['batch_size']*accum*n_meas/tot:.1f} 样本/s")
                if writer is not None:
                    writer.add_scalar("train/loss_pinball", loss_avg.item(), step)
                    writer.add_scalar("hyper/lr", cur_lr, step)
            _ee = int(t_cfg.get("eval_every", 0) or 0)
            if (_ee > 0 and step % _ee == 0) or step == t_cfg["total_steps"]:
                vl = evaluate(core, head, va_loader, device, q_t,
                              shard_mode=shard_mode,
                              use_anchor=bool(cfg["model"].get(
                                  "anchor_snaive", False)))
                print(f"[eval ] step={step} val_pinball={vl:.4f}")
                if writer is not None:
                    writer.add_scalar("val/pinball", vl, step)
                if vl < best_val:
                    best_val = vl
                    ckpt_best = dict(step=step, val=vl,
                                     core=clean_state_dict(core),
                                     head=clean_state_dict(head))
                    torch.save(ckpt_best, out / "best.pt")

    # 官方收尾口径：最后 N 个周期 ckpt 均匀平均（`train.ckpt_avg_last`，0=关 ✓）
    _avg_n = int(t_cfg.get("ckpt_avg_last", 0) or 0)
    avg_path = _average_checkpoints(out, _avg_n, cfg) if _avg_n > 0 else None
    if avg_path is not None:
        print(f"[ckpt] 最后 {_avg_n} 个周期 ckpt 均匀平均 → {avg_path.name}", flush=True)
    wall = time.time() - t_start
    torch.save(_ckpt_payload(step), out / "last.pt")
    # 每 epoch 检查点（可配 save_epochs="1,2,4"；基于 steps_per_epoch 换算）
    # 注意：epoch 检查点已在训练循环内按步保存（见上），此处不再补存。
    summary = dict(steps=step, wall_minutes=wall / 60,
                   best_val=(best_val if math.isfinite(best_val) else None),
                   params_M=n_par, device=device)
    summary.update(checkpoint_averaged=(avg_path.name if avg_path else None))
    # 吞吐与耗时拆解也回传（网格里就能直接按"哪个配方跑得动"筛 ✓）
    _tot = max(acc["data"] + acc["fwd_bwd"] + acc["opt"], 1e-9)
    _n = max(step - 1, 1)
    summary.update(ms_per_step=round(1000 * _tot / _n, 2),
                   samples_per_s=round(t_cfg["batch_size"] * accum * _n / _tot, 1),
                   time_data_pct=round(100 * acc["data"] / _tot, 1),
                   time_fwd_bwd_pct=round(100 * acc["fwd_bwd"] / _tot, 1),
                   time_opt_pct=round(100 * acc["opt"] / _tot, 1))
    summary.update(micro_batch_size=int(t_cfg["batch_size"]),
                   grad_accum_steps=accum,
                   effective_batch_size=int(t_cfg["batch_size"]) * accum,
                   samples_seen=step * int(t_cfg["batch_size"]) * accum)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[done] {json.dumps(summary)}")
    if writer is not None:
        writer.close()
    return summary
