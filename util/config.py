"""Load YAML configurations with inheritance and dotted-path overrides.

Priority is low to high:

1. The file named by ``inherit``.
2. Keys in the current configuration.
3. ``key.path=value`` entries supplied by the entry point.

Keys beginning with underscore are run metadata. ``load_config`` removes them,
while entry points can read them separately to dispatch pipelines or expand
grids. The resolved configuration is persisted beside the run output so later
runs can reproduce the exact model and hyperparameters.
"""







from __future__ import annotations



from pathlib import Path
from typing import Any
import yaml

# Repository root: util/config.py -> parents[1]
_PROJ = Path(__file__).resolve().parents[1]

# Top-level keys with this prefix are metadata rather than pipeline settings.
META_PREFIX = "_"


def deep_merge(base: dict, over: dict) -> dict:
    """Merge nested dictionaries recursively; non-mapping values replace their target."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_path(path_like: str, base_dir: Path) -> Path:
    """Resolve ``inherit`` relative to the current file, then the repository root."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    cand = base_dir / p
    if cand.exists():
        return cand
    return _PROJ / p


def strip_meta(cfg: dict) -> dict:
    """Remove top-level metadata keys before passing configuration to a pipeline.

    Entry points use ``_run`` for dispatch; pipeline implementations do not need it.
    """
    return {k: v for k, v in cfg.items() if not str(k).startswith(META_PREFIX)}


def read_meta(path: str | Path) -> dict:
    """Read top-level metadata keys from one configuration file.

    Metadata is deliberately not inherited.
    """

    raw = yaml.safe_load(Path(path).read_text()) or {}
    return {k: v for k, v in raw.items() if str(k).startswith(META_PREFIX)}


def load_config(path: str | Path, overrides: list[str] | None = None,
                _seen: set | None = None) -> dict:
    """Load a configuration through its inherit chain and apply final overrides.

    Args:
        path: Configuration file path.
        overrides: ``key.path=value`` entries applied with the highest priority.
        _seen: Internal set used to detect inheritance cycles.
    """





    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    _seen = _seen or set()
    real = path.resolve()
    if real in _seen:
        raise ValueError(f"Circular inherit path: {path}")
    _seen.add(real)

    # Metadata is removed before merging.
    cfg = strip_meta(yaml.safe_load(path.read_text()) or {})
    # Inherited values have lower priority than values in the current file.
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(_resolve_path(str(parent), path.parent), None, _seen)
        cfg = deep_merge(base, cfg)  # Current keys take precedence over inherited values.
    # Overrides have the highest priority.
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg: dict, items: list[str] | None) -> dict:
    """Apply ``key.path=value`` overrides in place and return the configuration.

    Entry points and configuration loading share this implementation.
    """
    for ov in items or []:
        if "=" not in ov:
            raise ValueError(f"Override must use key.path=value syntax: {ov}")
        key, raw = ov.split("=", 1)
        keys = key.split(".")
        node: Any = cfg
        for part in keys[:-1]:
            nxt = node.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise TypeError(f"Override path {key!r} conflicts at {part!r}")
            node = nxt
        # Keep prose fields as strings even when their text contains YAML punctuation.
        node[keys[-1]] = raw if keys[-1] in ("note", "notes", "desc") else _coerce(raw)
    return cfg


def _coerce(v: str):
    """Convert a string literal to bool, integer, float, None, or list."""
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def config_summary(cfg: dict) -> str:
    m, p = cfg["model"], cfg["pyramid"]
    return (
        f"model=fracast d={m['d_model']} W={m['W']} "
        f"stages={m.get('n_stages')} shared={m.get('share_stages')} "
        f"head={m.get('head_kind')} future_conv={m.get('head_future_conv', False)} "
        f"| steps={cfg['train']['total_steps']}"
    )


def estimate_params(m: dict) -> float:
    # Kept only for compatibility with older notebooks. Fracast's exact count is
    # reported by the model builder at runtime.
    d = int(m.get("d_model", 0))
    stages = int(m.get("n_stages", 0))
    return (stages * 4 * d * d) / 1e6


if __name__ == "__main__":
    import sys
    c = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/base.yaml",
                    sys.argv[2:])
    print(c)


def save_config(path, config):
    """Write a configuration dictionary as JSON and create parent directories."""
    from collections.abc import Mapping  # noqa: F401
    from util.io import save_json
    return save_json(path, dict(config))
