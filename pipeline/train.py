"""Pretrain FracCast, run periodic validation, and write checkpoints.

This module has no command-line interface. Invoke it through the root
``main.py`` entry point with a configuration file, for example:

    python main.py config/fraccast/pretrain_full.yaml
    python main.py config/fraccast/pretrain_smoke.yaml

All tunable values come from ``model``, ``pyramid``, ``heads``, ``repr``,
``data``, and ``train`` sections. Checkpoints store model state, optimizer
state, RNG state, global step, and best validation loss so interrupted runs
can resume exactly. The sampler replays ``step * grad_accum`` micro batches,
which preserves batch order without rereading completed samples.
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
    """Resolve ``train.resume`` and return ``(checkpoint, path)`` or ``(None, None)``."""
    spec = t_cfg.get("resume", "auto")
    if spec is None or spec is False or str(spec).lower() in (
            "false", "off", "none", "0", ""):
        return None, None
    path = (out / "last.pt") if str(spec) == "auto" else Path(str(spec))
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        print(f"[resume] {path} does not exist; starting a new run", flush=True)
        return None, None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "core" not in ckpt:
        raise ValueError(f"Invalid training checkpoint without model state: {path}")
    print(f"[resume] {path} step={int(ckpt.get('step') or 0):,} "
          f"chunk={ckpt.get('chunk')}"
          f" | optimizer={'yes' if ckpt.get('optim') else 'no (warm start only)'}"
          f" | RNG={'yes' if ckpt.get('rng') else 'no'}", flush=True)
    return ckpt, path


def _restore_rng(ckpt, device: str, rndmod, npmod, aug_rng) -> None:
    """Restore every RNG state stored in a checkpoint."""
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
    print("[resume] Restored Python, NumPy, PyTorch, CUDA, and augmentation RNG",
          flush=True)


def _add_committing(loss, qh, tgt_d, copy_h, tgt_mask, q_t, T_avail,
                    cfg, device, head):
    """Add TinyCast's gated committing term to the pinball loss when enabled."""
    cw = float((cfg.get("train") or {}).get("commit_w", 0.0) or 0.0)
    if cw <= 0 or copy_h is None or loss is None:
        return loss
    if hasattr(head, "combined_loss"):      # Student-t heads use another objective.
        return loss
    from module.losses.tinycast import committing_loss
    q_mid = q_t.shape[0] // 2
    med2 = qh[:, :T_avail, q_mid]                       # [B,H]
    tgt2 = tgt_d[:, :T_avail]                           # [B,H]
    cp2 = copy_h[:, :T_avail].to(device)                # [B,H]
    msk2 = tgt_mask[:, :T_avail].to(device) if tgt_mask is not None else None
    return loss + committing_loss(med2, tgt2, cp2, mask=msk2, weight=cw)


def _augment_window(win, winm, sf, aug, rng):
    """Apply the stochastic augmentations described by TinyCast appendix A.2."""
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
    """Run TinyCast-style autoregressive rollout with scheduled sampling."""
    from module.pyramid.levels import build_levels, window_minmax
    from module.losses.tinycast import committing_loss, seasonal_copy_baseline

    t_cfg, m_cfg = cfg["train"], cfg["model"]
    K = max(1, int(t_cfg.get("ar_chunks", 1) or 1))
    p = int(head.horizon)
    L = int(win.shape[1]) - K * p
    ratios = list(cfg["pyramid"]["ratios"])
    # Level widths are scalar (all levels equal) or coarse-to-fine.
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
            # The normalized context is consumed only when future_conv is enabled.
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
    """Uniformly average the final ``n_avg`` chunk checkpoints."""
    cks = sorted(out.glob("chunk*.pt"))
    if len(cks) < 2:
        return None
    cks = cks[-max(1, int(n_avg)):]
    core_sum, head_sum, n = {}, {}, 0
    for pth in cks:
        # Checkpoints may contain NumPy scalar types rejected by safe loading.
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
    """Evaluate sampled validation batches."""
    # Use the same horizon path that training optimizes.
    core.eval()
    tot, n = 0.0, 0
    is_st = hasattr(head, "combined_loss")   # Student-t distribution head.
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            if shard_mode:
                # Current batches include a seasonal copy target; 11-field
                # batches are retained for backward compatibility.
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
            # For multi-horizon targets, evaluate the final forecast point.
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
                # Emit the full horizon exactly as training does.
                lo_d = xn.reshape(xn.shape[0], -1)[:, -1].unsqueeze(-1).to(device)
                # The head consumes context in the normalized encoder space.
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
                # Masked pinball over the full horizon is more stable.
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
    """Train once from a resolved configuration and return the run summary."""
    cfg = dict(config)

    # Seed every source that can alter sampling or model initialization.
    import random as _rndmod
    import numpy as _npmod
    _seed = int(cfg.get("train", {}).get("seed", cfg.get("data", {}).get("seed", 42)))
    _rndmod.seed(_seed)
    _npmod.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    cfg.setdefault("train", {})["seed_used"] = _seed
    print(f"  [seed] random/numpy/torch set to {_seed}", flush=True)

    t_cfg = cfg["train"]
    out = Path(t_cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # Resume restores weights, optimizer moments, global step, and random sources.
    # The sampler skips every micro batch consumed by completed steps.
    resume_ckpt, resume_path = _load_resume(out, t_cfg)
    resume_step = int((resume_ckpt or {}).get("step") or 0)
    if resume_path is not None:
        t_cfg["resume_from"] = str(resume_path)
    (out / "config_used.yaml").write_text(json.dumps(cfg, indent=1))

    try:
        from torch.utils.tensorboard import SummaryWriter
        # TensorBoard appends events, so a rerun with the same tag starts clean.
        if (out / "tb").exists():
            import shutil
            shutil.rmtree(out / "tb")
        writer = SummaryWriter(log_dir=str(out / "tb"))
        try:
            m = cfg["model"]
            # A compact experiment card appears on the TensorBoard text tab.
            note = (t_cfg.get("note", "")
                    or f"Experiment {out.name}: {cfg['train'].get('out_dir', '')}")
            writer.add_text(
                "exp/note",
                (f"**{out.name}**  |  {t_cfg.get('note', '')}\n\n"
                 f"Recipe: family={m.get('family', 'fractal')} "
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
            # Keep the plain config view alongside the experiment card.
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
        writer = None   # TensorBoard is optional; training continues without it.
    # Device preference: CUDA > MPS > CPU. CUDA_VISIBLE_DEVICES selects the GPU.
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

    # Load resumed weights before torch.compile; checkpoints use canonical names.
    if resume_ckpt is not None:
        core.load_state_dict(resume_ckpt["core"], strict=True)
        head.load_state_dict(resume_ckpt["head"], strict=True)
        print(f"[resume] loaded weights at step {resume_step:,}", flush=True)

    # Save canonical state dictionaries so evaluation consumes the same names.
    from util.ckpt import clean_state_dict
    # Performance switches. TF32 is opt-in because BF16 autocast supersedes it.
    if device == "cuda" and bool(t_cfg.get("tf32", False)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[speed] TF32 enabled", flush=True)
    if bool(t_cfg.get("compile", False)):
        # reduce-overhead helps launch-bound models; max-autotune trades startup time.
        _cmode = t_cfg.get("compile_mode") or None
        core = torch.compile(core, mode=_cmode) if _cmode else torch.compile(core)
        head = torch.compile(head, mode=_cmode) if _cmode else torch.compile(head)
        print(f"[speed] torch.compile enabled with mode={_cmode or 'default'}", flush=True)
    # cudnn benchmarking helps fixed shapes but slows workloads with changing shapes.
    if device == "cuda" and bool(t_cfg.get("cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True
        print("[speed] cudnn.benchmark enabled", flush=True)

    # Optionally profile the first N steps and write a summary table.
    _prof_n = int(t_cfg.get("profile_steps") or 0)
    prof = None
    if _prof_n > 0 and device == "cuda":
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
        prof.__enter__()
    n_par = (sum(p.numel() for p in core.parameters())
             + sum(p.numel() for p in head.parameters())) / 1e6
    print(f"[model] full={n_par:.2f}M params mode={cfg['pyramid']['mode']}")

    # DataPort owns corpus, shard, and legacy streaming input selection.
    from dataport.dataport import build_train_loaders

    accum = max(1, int(t_cfg.get("grad_accum_steps", 1)))
    # On resume, replay sampler randomness and skip consumed micro batches.
    skip_micro = resume_step * accum

    sampling = t_cfg.get("sampling") or {}
    _policy = str(sampling.get("policy", "mixture"))
    if _policy == "rotation":
        from pipeline.policies.rotation import make_rotation_sampler
        sampler_factory = lambda ds, bs, seed, ga: make_rotation_sampler(
            ds, bs, seed, ga, cfg, skip_micro=skip_micro)
    elif _policy == "band_mix":
        # Sample by frequency-band share to prevent large corpora dominating small ones.
        from pipeline.policies.band_mix import make_band_mix_sampler
        sampler_factory = lambda ds, bs, seed, ga: make_band_mix_sampler(
            ds, bs, seed, ga, cfg, skip_micro=skip_micro)
    else:
        sampler_factory = None
    tr_loader, va_loader, shard_mode = build_train_loaders(
        cfg, sampler_factory=sampler_factory, skip_micro=skip_micro)
    if skip_micro:
        print(f"[resume] skipped {skip_micro:,} micro batches "
              f"({resume_step:,} steps x {accum} accumulation)", flush=True)

    # Derive total steps from the actual loader unless the config sets them explicitly.
    spe = int(t_cfg.get("steps_per_epoch") or 0) or max(
        1, math.ceil(len(tr_loader) / accum))
    if not t_cfg.get("total_steps"):
        t_cfg["total_steps"] = spe * max(1, int(t_cfg.get("epochs") or 1))
    t_cfg["steps_per_epoch"] = spe
    print(f"[plan] one epoch={spe:,} steps; total_steps={t_cfg['total_steps']:,}",
          f" epochs={t_cfg['total_steps'] / spe:.2f}", flush=True)

    # Keep norm and bias parameters outside weight decay when grouping is enabled.
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

    # Gradient clipping needs the flat parameter list even when optimizer groups exist.
    clip_params = list(core.parameters()) + list(head.parameters())

    # Use fused AdamW when supported and fall back transparently otherwise.
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

    # Resume Adam moments with the schedule state; omitting them changes the decay path.
    if resume_ckpt is not None and resume_ckpt.get("optim"):
        opt.load_state_dict(resume_ckpt["optim"])
        print("[resume] optimizer moments restored", flush=True)
    elif resume_ckpt is not None:
        print("[resume] checkpoint has no optimizer moments; warm-starting Adam", flush=True)

    # TinyCast-compatible rollout accepts raw [context + K x horizon] windows.
    # Normalization, pyramid construction, and feedback remain in _rollout_loss.
    # Dispatch by data layout; K=1 is a single-block rollout.
    rollout_on = (shard_mode
                  and str((cfg.get("data") or {}).get("window_mode", "levels")) == "raw")
    eps_max = float(t_cfg.get("scheduled_sampling_max", 0.5))
    _aug = dict(t_cfg.get("augment") or {})
    _aug_on = bool(_aug.pop("enabled", False)) and rollout_on
    _aug_rng = random.Random(int(cfg["data"].get("seed", 42)) + 7)
    # Restore augmentation randomness with the other resume state.
    _restore_rng(resume_ckpt, device, _rndmod, _npmod, _aug_rng)
    if rollout_on:
        print(f"[rollout] AR chunks={int(t_cfg['ar_chunks'])}; "
              f"horizon={int(head.horizon)}; epsilon_max={eps_max}; augmentation={_aug_on}", flush=True)

    sched = t_cfg.get("lr_schedule", "cosine")   # cosine | wsd | cosine_restarts

    def lr_at(step):
        if step < t_cfg["warmup_steps"]:
            return t_cfg["lr"] * step / t_cfg["warmup_steps"]
        if sched == "wsd":
            # Warmup-stable-decay plateaus, then decays as one minus the square root.
            stable_end = int(t_cfg["total_steps"] * t_cfg.get("wsd_stable_frac", 0.7))
            if step < stable_end:
                return t_cfg["lr"]
            p = (step - stable_end) / max(
                1, t_cfg["total_steps"] - stable_end)
            return t_cfg.get("min_lr", 1e-5) + (t_cfg["lr"] - t_cfg.get(
                "min_lr", 1e-5)) * (1 - math.sqrt(min(p, 1.0)))
        if sched == "cosine_restarts":
            # Cosine schedule with periodic warm restarts.
            period = max(1, int(t_cfg["total_steps"] * t_cfg.get(
                "restart_frac", 0.25)))
            p = ((step - t_cfg["warmup_steps"]) % period) / period
            return t_cfg["lr"] * 0.5 * (1 + math.cos(math.pi * p))
        p = (step - t_cfg["warmup_steps"]) / max(
            1, t_cfg["total_steps"] - t_cfg["warmup_steps"])
        return t_cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    # QuantileHead exposes q; StudentTHead exposes its own quantile grid.
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
        print(f"[resume] continuing from step {resume_step:,}; "
              f"{int(t_cfg['total_steps']) - resume_step:,} of "
              f"{int(t_cfg['total_steps']):,} steps remain", flush=True)

    # Timing buckets separate data waiting, forward/backward, and optimizer work.
    # The first step is excluded because it includes warmup and compile overhead.
    acc = {"data": 0.0, "fwd_bwd": 0.0, "opt": 0.0}
    t_prev = time.perf_counter()
    group_micro = 0
    group_data = 0.0
    group_fwd = 0.0
    loss_sum = torch.zeros((), device=device)

    spe = int(t_cfg.get("steps_per_epoch", 0) or 0)
    # Persist periodic checkpoints during a single uninterrupted run.
    _se = t_cfg.get("save_epochs")
    save_epoch_set = ({int(x) for x in _se} if isinstance(_se, (list, tuple))
                      else {int(x) for x in str(_se or "").split(",") if x.strip()})
    _every = int(t_cfg.get("save_every_epochs") or 0)
    if _every > 0:
        _n_ep = int(math.ceil(t_cfg["total_steps"] / max(1, spe)))
        save_epoch_set |= set(range(_every, _n_ep + 1, _every))
    if save_epoch_set:
        print(f"[ckpt] saving epochs {sorted(save_epoch_set)}; one epoch={spe:,} steps", flush=True)

    def _rng_snapshot() -> dict:
        """Capture every RNG state that can change the training trajectory."""
        snap = {"torch": torch.get_rng_state(),
                "python": _rndmod.getstate(),
                "numpy": _npmod.random.get_state(),
                "aug": _aug_rng.getstate()}
        if device == "cuda":
            snap["cuda"] = torch.cuda.get_rng_state_all()
        return snap

    def _ckpt_payload(_step: int, **extra) -> dict:
        """Build a complete checkpoint with model, optimizer, RNG, and progress."""
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
                    # Raw rollout batches contain context, mask, and seasonal scale.
                    win, winm, sf = (x.to(device, non_blocking=True)
                                     for x in batch[:3])
                elif len(batch) >= 12:
                    # Current shard batches add a seasonal copy target; older batches have 11 fields.
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
            t_now = time.perf_counter()  # Wait until the batch arrives.
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
                # Overlap host-to-device copies with GPU computation.
                xta = xta.to(device, non_blocking=True)
                tgt_d = tgt.to(device, non_blocking=True)
            if rollout_on:
                # Scheduled sampling rises linearly to epsilon_max during the first half.
                # An optional explicit ramp keeps probes comparable to the reference run.
                _ramp = float(t_cfg.get("scheduled_sampling_ramp_steps") or 0.0)
                if _ramp <= 0.0:
                    _ramp = 0.5 * t_cfg["total_steps"]
                eps = eps_max * min(1.0, step / max(1.0, _ramp))
                if _aug_on:
                    win, winm, sf = _augment_window(win, winm, sf, _aug, _aug_rng)
                loss = _rollout_loss(core, head, win, winm, sf, q_t, cfg,
                                     amp_scope, eps)
            elif cfg["model"].get("multi_horizon", False) or head.horizon > 0:
                # Predict the full horizon jointly, as Chronos-2 does.
                # Every horizon point contributes supervision.
                with amp_scope():
                    h_ctx = core(xn, xm, xc, level_ids=lids,
                                 ts_norm=xt, t_abs=xta)
                # Flatten the token mask only for masked-pool heads.
                _tok_w = xm.reshape(xm.shape[0], -1).to(device) \
                    if getattr(head, "head_pool", "mean") == "masked" else None
                H = head.horizon if head.horizon > 0 else tgt_d.shape[-1]
                # The final normalized observation anchors residual forecasts.
                # Levels are coarse-to-fine, so the final token is the finest-level observation.
                lo = xn.reshape(xn.shape[0], -1)[:, -1].unsqueeze(-1)  # [B, 1]
                T_avail = tgt_d.shape[-1]
                use_anchor = bool(cfg["model"].get("anchor_snaive", False))
                if use_anchor:
                    # Seasonal-naive anchor, one value per forecast step.
                    with amp_scope():
                        qh = head.forward_horizon(
                            h_ctx, last_obs=None,
                            anchor=anchor[:, :H].to(device), token_weight=_tok_w)
                elif hasattr(head, "combined_loss"):
                    # Student-t distribution head with a robust loss term.
                    with amp_scope():
                        df, mu, sc = head.forward_horizon(h_ctx, last_obs=lo)
                    qh = head.quantiles(df, mu, sc)  # Used by the delta regularizer.
                    loss = head.combined_loss(
                        tgt_d, df[:, :T_avail], mu[:, :T_avail],
                        sc[:, :T_avail]).mean()
                else:
                    with amp_scope():
                        qh = head.forward_horizon(h_ctx, last_obs=lo,
                                                  token_weight=_tok_w)
                if not use_anchor and not hasattr(head, "combined_loss"):
                    # The dataset supplies a contiguous multi-horizon target segment.
                    qf = qh[:, :T_avail].reshape(-1, q_t.shape[0])  # [B*H, Q]
                    tf = tgt_d.reshape(-1)                          # [B*H]
                    mask_f = (tgt_mask[:, :T_avail].reshape(-1).to(device)
                              if tgt_mask is not None else None)
                    loss = pinball_loss_mask(qf, tf, q_t, mask_f)
                    loss = _add_committing(loss, qh, tgt_d, copy_h, tgt_mask,
                                           q_t, T_avail, cfg, device, head)
                elif use_anchor:
                    # Compare the anchored prediction directly with the absolute target.
                    qf = qh[:, :T_avail].reshape(-1, q_t.shape[0])
                    tf = tgt_d.reshape(-1)
                    mask_f = (tgt_mask[:, :T_avail].reshape(-1).to(device)
                              if tgt_mask is not None else None)
                    loss = pinball_loss_mask(qf, tf, q_t, mask_f)
                    loss = _add_committing(loss, qh, tgt_d, copy_h, tgt_mask,
                                           q_t, T_avail, cfg, device, head)
                # Regularize deviations from the anchor unless the data provide strong evidence.
                # The anchor is either seasonal naive or the last normalized observation.
                dr = cfg["model"].get("delta_reg", 0.0)
                if dr > 0 and loss is not None:
                    # Measure median deviation from the selected anchor.
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
                        # Keep the predicate on GPU to avoid a per-step host synchronization.
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
                # Optional cross-scale InfoNCE alignment between adjacent levels.
                align_lambda = cfg["model"].get("align_lambda", 0.0)
                z_levels = getattr(core, "z_levels", None)
                if align_lambda > 0 and z_levels is not None:
                    z = torch.nn.functional.normalize(z_levels, dim=-1)
                    B, L, _ = z.shape
                    align_loss = torch.tensor(0.0, device=device)
                    n_pairs = 0
                    for l in range(L - 1):
                        # Positive pairs are adjacent levels of the same sample.
                        pos = (z[:, l] * z[:, l + 1]).sum(-1)          # [B]
                        # Negative pairs use other samples at the adjacent level.
                        sim = z[:, l] @ z[:, l + 1].T                  # [B, B]
                        sim.fill_diagonal_(float("-inf"))
                        logits = torch.cat([pos.unsqueeze(-1), sim], dim=-1)
                        labels = torch.zeros(B, dtype=torch.long, device=device)
                        align_loss = align_loss + torch.nn.functional.cross_entropy(
                            logits / 0.5, labels)   # Temperature 0.5 is stable for this loss.
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
            if step > 1:  # Exclude first-step fill, kernel selection, and compilation.
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
                print(f"[speed] profiled first {_prof_n} steps; wrote {out}/profiler.txt", flush=True)
                prof = None

            # Save epoch checkpoints immediately; recreating them later would overwrite
            # each earlier epoch with the final weights.
            if spe > 0 and step % spe == 0 and (step // spe) in save_epoch_set:
                _ep = step // spe
                torch.save(_ckpt_payload(step, epoch=_ep),
                           out / f"epoch{_ep}.pt")
                print(f"[ckpt] saved epoch {_ep} at step {step}", flush=True)

            _chunk_steps = int(
                t_cfg.get("ckpt_every_steps")
                or (sampling.get("rotation") or {}).get("chunk_steps") or 0)
            if _chunk_steps > 0 and (step % _chunk_steps == 0
                                     or step == t_cfg["total_steps"]):
                _chunk = int(math.ceil(step / _chunk_steps))
                # Chunk and last checkpoints share the same resumable payload. last.pt is
                # used for resume; chunk files support curves and checkpoint averaging.
                _pay = _ckpt_payload(step, chunk=_chunk)
                torch.save(_pay, out / f"chunk{_chunk:02d}.pt")
                torch.save(_pay, out / "last.pt")
                print(f"[ckpt] saved chunk {_chunk:02d} at step {step}", flush=True)

            if step % t_cfg["log_every"] == 0:
                el = time.time() - t_start
                tot = max(acc["data"] + acc["fwd_bwd"] + acc["opt"], 1e-9)
                n_meas = max(step - 1, 1)
                print(f"[step {step:>6}] loss={loss_avg.item():.4f} lr={cur_lr:.2e} "
                      f"elapsed={el/60:.1f}min | data {100*acc['data']/tot:.0f}% "
                      f"fwd+bwd {100*acc['fwd_bwd']/tot:.0f}% opt {100*acc['opt']/tot:.0f}% "
                      f"| {1000*tot/n_meas:.0f} ms/step "
                      f"{t_cfg['batch_size'] * accum * n_meas / tot:.1f} samples/s")
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

    # Uniformly average the final N chunk checkpoints when enabled.
    _avg_n = int(t_cfg.get("ckpt_avg_last", 0) or 0)
    avg_path = _average_checkpoints(out, _avg_n, cfg) if _avg_n > 0 else None
    if avg_path is not None:
        print(f"[ckpt] averaged the final {_avg_n} chunk checkpoints into {avg_path.name}", flush=True)
    wall = time.time() - t_start
    torch.save(_ckpt_payload(step), out / "last.pt")
    # Epoch checkpoints are already written at their exact steps inside the loop.
    summary = dict(steps=step, wall_minutes=wall / 60,
                   best_val=(best_val if math.isfinite(best_val) else None),
                   params_M=n_par, device=device)
    summary.update(checkpoint_averaged=(avg_path.name if avg_path else None))
    # Return throughput and timing buckets so batch runs remain easy to triage.
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
