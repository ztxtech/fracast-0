"""Checkpoint state-dict helpers.

The training code may wrap models with ``torch.compile``.  Saving a compiled
model directly leaves ``_orig_mod.`` in every state-dict key, which makes a
checkpoint impossible to load into the plain model later.  The helpers here
keep the on-disk checkpoint format stable across CPU, CUDA, MPS, and compiled
training runs.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


_WRAPPER_PREFIXES = ("_orig_mod.", "module.")


def unwrap(module: Any) -> Any:
    """Return the underlying module from common PyTorch wrappers."""
    return getattr(module, "_orig_mod", module)


def clean_state_dict(module: Any) -> dict:
    """Return a state dict with wrapper prefixes removed."""
    return unwrap(module).state_dict()


def strip_prefixes(state_dict: Mapping[str, Any]) -> dict:
    """Remove nested ``_orig_mod.`` and ``module.`` prefixes."""
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        name = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
                    break
        out[name] = value
    return out


def load_weights(
    module: Any,
    state_dict: Mapping[str, Any],
    name: str = "state_dict",
    allow_missing: Iterable[str] = (),
    allow_unexpected: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Load a state dict and reject unexpected key mismatches.

    ``allow_missing`` and ``allow_unexpected`` are intended only for explicit,
    documented compatibility cases.  A typo in a checkpoint path should fail
    loudly instead of silently producing a randomly initialized model.
    """
    raw = strip_prefixes(state_dict)
    missing, unexpected = module.load_state_dict(raw, strict=False)
    missing, unexpected = list(missing), list(unexpected)
    allowed_missing = set(allow_missing)
    allowed_unexpected = set(allow_unexpected)
    bad_missing = [key for key in missing if key not in allowed_missing]
    bad_unexpected = [key for key in unexpected if key not in allowed_unexpected]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"[ckpt] {name} does not match the model: "
            f"missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}"
        )
    return missing, unexpected
