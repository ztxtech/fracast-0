"""Expand the configuration grid declared in ``_run.grid``.

Simple dimensions use dotted configuration paths and candidate lists. A mapping
candidate declares a coordinated set of values for dimensions that must change
together, such as model width and head count. The ``_name`` field labels that
candidate only.

Expansion is deterministic: dictionary order defines dimensions, candidate order
defines combinations, and every run tag and output path receives the same
suffix. A ``{grid}`` placeholder in an output path is replaced directly. This
module only expands configurations; execution belongs to the entry point.

Example::

    _run:
      grid:
        model.d_model: [64, 128]
        train.lr: [1.0e-3, 3.0e-4]
"""

from __future__ import annotations

import copy
import itertools
import re
from typing import Any, Mapping, Sequence

# Output keys whose values are paths receive a unique grid suffix.
OUT_KEYS = ("out_dir", "out")

# Keep suffixes safe for filenames and labels.
_SLUG_RE = re.compile(r"[^0-9A-Za-z_.-]+")


def has_grid(cfg: Mapping[str, Any]) -> bool:
    """Return whether the run metadata declares a non-empty grid."""
    return bool(grid_of(cfg))


def grid_of(cfg: Mapping[str, Any]) -> dict[str, Sequence[Any]]:
    """Return ``_run.grid`` or an empty mapping."""
    raw = (cfg.get("_run") or {}).get("grid") or {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"_run.grid must be a mapping, got {type(raw).__name__}"
        )
    return dict(raw)


def grid_size(grid: Mapping[str, Sequence[Any]]) -> int:
    """Return the number of grid combinations."""
    n = 1
    for values in grid.values():
        n *= len(_values(values))
    return n


def slug(value: Any) -> str:
    """Convert a candidate value to a short filename- and label-safe string."""
    s = str(value).strip().lower()
    s = s.replace("-", "m")  # Preserve a compact form for negative exponents.
    s = _SLUG_RE.sub("_", s).strip("_")
    return s or "v"


def combo_suffix(keys: Sequence[str], combo: Sequence[Any]) -> str:
    """Build a deterministic suffix from the final segment of each key."""
    parts = [f"{str(key).split('.')[-1]}{slug(_label_of(value))}"
             for key, value in zip(keys, combo)]
    return "-".join(parts)


def is_group(value: Any) -> bool:
    """Return whether a dimension contains mapping-based coordinated candidates."""
    return any(isinstance(v, Mapping) for v in _values(value))


def select_group(value: Any, name: Any) -> Mapping[str, Any]:
    """Select one coordinated candidate by ``_name`` or ordinal position.

    This is useful for rerunning a single grid cell from a queue.
    """
    vals = _values(value)
    want = str(name).strip().lower()
    for i, item in enumerate(vals):
        if not isinstance(item, Mapping):
            continue
        nm = item.get("_name")
        if (nm is not None and str(nm).strip().lower() == want) or str(i) == want:
            return dict(item)
    raise ValueError(
        f"Unknown coordinated candidate {name!r}; choose from "
        f"{[v.get('_name', i) for i, v in enumerate(vals) if isinstance(v, Mapping)]}"
    )


def expand_grid(cfg: Mapping[str, Any],
                grid: Mapping[str, Sequence[Any]] | None = None) -> list[dict]:
    """Expand a grid and return one resolved configuration per combination.

    Without a grid, this returns a single deep copy of ``cfg``.
    """
    grid = dict(grid if grid is not None else grid_of(cfg))
    if not grid:
        return [copy.deepcopy(dict(cfg))]

    keys = list(grid)
    value_lists = [_values(grid[k]) for k in keys]
    variants: list[dict] = []
    for combo in itertools.product(*value_lists):
        variant = copy.deepcopy(dict(cfg))
        for key, value in zip(keys, combo):
            if isinstance(value, Mapping):  # Keep coordinated values together.
                apply_group(variant, key, value)
            else:
                _set_path(variant, key, value)
        suffix = combo_suffix(keys, combo)
        run_meta = variant.setdefault("_run", {})
        base_tag = str(run_meta.get("tag") or "grid")
        run_meta["tag"] = f"{base_tag}__{suffix}"
        _apply_output_suffix(variant, suffix)
        variants.append(variant)
    return variants


# Internal helpers


def _values(value: Any) -> list[Any]:
    """Normalize candidate values to a list and reject an empty dimension.

    A mapping is also valid when it declares one coordinated candidate.
    """
    vals = list(value) if isinstance(value, (list, tuple, set)) else [value]
    if not vals:
        raise ValueError("A grid dimension has no candidate values")
    return vals


def apply_group(cfg: dict, label: str, group: Mapping[str, Any]) -> None:
    """Apply every non-metadata entry in a coordinated candidate.

    The dimension label is metadata and is deliberately not written to ``cfg``.
    """
    items = {k: v for k, v in group.items() if not str(k).startswith("_")}
    if not items:
        raise ValueError(f"Coordinated candidate {label!r} has no configuration keys")
    for path, value in items.items():
        _set_path(cfg, path, value)


def _label_of(value: Any) -> Any:
    """Return the short label used in a run suffix."""
    if isinstance(value, Mapping):
        name = value.get("_name")
        if name is not None:
            return name
        vals = [v for k, v in value.items() if not str(k).startswith("_")]
        return "-".join(str(v) for v in vals)
    return value


def _set_path(cfg: dict, path: str, value: Any) -> None:
    """Set a nested value, creating intermediate mappings as needed."""
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        raise ValueError(f"Empty grid parameter path: {path!r}")
    node: Any = cfg
    for part in parts[:-1]:
        nxt = node.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise TypeError(f"Grid path {path!r} conflicts at {part!r}")
        node = nxt
    node[parts[-1]] = value


def _suffix_path(path: str, suffix: str) -> str:
    """Append a suffix to an output path or expand its ``{grid}`` placeholder."""
    if "{grid}" in path:
        return path.replace("{grid}", suffix)
    return f"{path}__{suffix}"


def _apply_output_suffix(node: Any, suffix: str) -> None:
    """Recursively append suffixes to configured output paths."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and key in OUT_KEYS and "/" in value:
                node[key] = _suffix_path(value, suffix)
            else:
                _apply_output_suffix(value, suffix)
    elif isinstance(node, list):
        for item in node:
            _apply_output_suffix(item, suffix)
