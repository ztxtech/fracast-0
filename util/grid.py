"""参数网格：把配置里的 `_run.grid` 展开成一组可独立运行的配置。

参数实验使用配置文件里的网格，避免把搜索逻辑写死在训练代码里。

写法（`_run` 是元数据块；点号路径指向配置主体里的嵌套键 ✓）：

    _run:
      kind: train_eval
      tag: g_dmodel
      grid:
        model.d_model: [64, 128]
        train.lr: [1.0e-3, 3.0e-4]

**成组候选（几个键必须一起动的时候 ✓）**：候选值写成映射，键名只是这一维的标签；
映射里的每个 `点号路径: 值` 都会写进配置，`_name` 只用于后缀（不写进配置 ✓）。

    _run:
      grid:
        width:                                  # 标签（不是配置路径）
          - {_name: w96,  model.d_model: 96,  model.n_heads: 3, model.d_ff: 384}
          - {_name: w128, model.d_model: 128, model.n_heads: 4, model.d_ff: 1024}
        model.n_layers: [1, 2, 3]                # → 2×3 = 6 个组合 ✓

为什么要成组：宽度（`d_model`）一改，head 数与 FFN 宽度**必须跟着改**（整除、宽深配比），
纯笛卡尔积会展开出 `d_model=96 + n_heads=4 + d_ff=1024` 这种不想要的组合 ✗。

展开规则：
  · 笛卡尔积；顺序稳定（键按声明顺序、值按声明顺序 ✓）—— 同样输入必得同样顺序 ✓
  · 每个组合的 `_run.tag` 追加组合后缀：`g_dmodel__d_model64-lr1em3` /
    `g_depth_width__width-w128-layers3` ✓
  · 输出路径（键名 `out_dir` / `out`、值是含 `/` 的字符串）同步追加同一后缀 ——
    不换目录就会互相覆盖 ✗；想自己安排路径就在值里写 `{grid}` 占位符 ✓
  · 组合身份由配置内容决定，便于复现与去重。

本模块只负责展开；执行由 `main.py` 负责。
"""
from __future__ import annotations

import copy
import itertools
import re
from typing import Any, Mapping, Sequence

# 值是路径的输出键：网格展开时同步加后缀（否则组合之间互相覆盖 ✗）
OUT_KEYS = ("out_dir", "out")

# 后缀里只允许这些字符，其余换成下划线（避免空格/斜杠进文件名 ✓）
_SLUG_RE = re.compile(r"[^0-9A-Za-z_.-]+")


def has_grid(cfg: Mapping[str, Any]) -> bool:
    """配置里有没有非空 `_run.grid` ✓。"""
    return bool(grid_of(cfg))


def grid_of(cfg: Mapping[str, Any]) -> dict[str, Sequence[Any]]:
    """取 `_run.grid`（没有就是空 dict ✓）。"""
    raw = (cfg.get("_run") or {}).get("grid") or {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"_run.grid 必须是映射（键=参数路径，值=候选列表），"
            f"收到 {type(raw).__name__}"
        )
    return dict(raw)


def grid_size(grid: Mapping[str, Sequence[Any]]) -> int:
    """网格组合数（空网格 = 1，即不展开 ✓）。"""
    n = 1
    for values in grid.values():
        n *= len(_values(values))
    return n


def slug(value: Any) -> str:
    """候选值 → 文件名/标签安全的短串（`1e-3` → `1em3` ✓）。"""
    s = str(value).strip().lower()
    s = s.replace("-", "m")          # 负号与科学计数法的减号统一成 m
    s = _SLUG_RE.sub("_", s).strip("_")
    return s or "v"


def combo_suffix(keys: Sequence[str], combo: Sequence[Any]) -> str:
    """组合后缀：`d_model64-lr1em3`（键只取点号路径的最后一段 ✓）。"""
    parts = [f"{str(key).split('.')[-1]}{slug(_label_of(value))}"
             for key, value in zip(keys, combo)]
    return "-".join(parts)


def is_group(value: Any) -> bool:
    """这一维是不是成组候选（候选里含映射 ✓）。"""
    return any(isinstance(v, Mapping) for v in _values(value))


def select_group(value: Any, name: Any) -> Mapping[str, Any]:
    """按 `_name`（或下标）从成组候选里选一组 ✓；找不到就早失败 ✗。

    用途：把一个网格的单个格子单独跑（`-o width=w128`，队列一任务一格 ✓）。
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
        f"成组候选里没有 {name!r}；可选："
        f"{[v.get('_name', i) for i, v in enumerate(vals) if isinstance(v, Mapping)]}")


def expand_grid(cfg: Mapping[str, Any],
                grid: Mapping[str, Sequence[Any]] | None = None) -> list[dict]:
    """展开网格 → 一组配置（无网格时返回"原配置的单元素列表" ✓）。

    Parameters
    ----------
    cfg
        已加载的配置（含 `_run` 元数据）。
    grid
        显式网格；不给就取 `cfg["_run"]["grid"]` ✓。
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
            if isinstance(value, Mapping):     # 成组候选：几个键必须一起动 ✓
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


# --------------------------------------------------------------------- 内部
def _values(value: Any) -> list[Any]:
    """候选值统一成列表；空列表直接报错（免得静默少跑 ✓）。

    单个映射也是合法候选（成组候选只写一组时 ✓）。
    """
    vals = list(value) if isinstance(value, (list, tuple, set)) else [value]
    if not vals:
        raise ValueError("_run.grid 里有空候选列表 —— 会静默少跑 ✗，请写明候选值或删掉该行")
    return vals


def apply_group(cfg: dict, label: str, group: Mapping[str, Any]) -> None:
    """成组候选：把映射里的每个 `路径: 值` 写进配置（`_name` 只当标签 ✓）。

    `label`（这一维的名字，如 `width`）**不写进配置** ✗ —— 写进去会在配置里留一个
    模型读不到、以后也会让人困惑的无用键 ✗。
    """
    items = {k: v for k, v in group.items() if not str(k).startswith("_")}
    if not items:
        raise ValueError(f"_run.grid[{label!r}] 的成组候选里没有任何键（只有 _name ✗）")
    for path, value in items.items():
        _set_path(cfg, path, value)


def _label_of(value: Any) -> Any:
    """成组候选 → 后缀里用的短标签（`_name` 优先，否则按值顺序拼 ✓）。"""
    if isinstance(value, Mapping):
        name = value.get("_name")
        if name is not None:
            return name
        vals = [v for k, v in value.items() if not str(k).startswith("_")]
        return "-".join(str(v) for v in vals)
    return value


def _set_path(cfg: dict, path: str, value: Any) -> None:
    """按点号路径写嵌套键（中间层不存在就建 ✓）。"""
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        raise ValueError(f"网格参数路径为空: {path!r}")
    node: Any = cfg
    for part in parts[:-1]:
        nxt = node.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise TypeError(f"网格参数路径 {path!r} 冲突：{part!r} 已经是 {type(nxt).__name__}")
        node = nxt
    node[parts[-1]] = value


def _suffix_path(path: str, suffix: str) -> str:
    """给输出路径加后缀：`output/g_x` → `output/g_x__<后缀>` ✓。

    值里写了 `{grid}` 占位符就只做替换（路径由用户自己安排 ✓）。
    """
    if "{grid}" in path:
        return path.replace("{grid}", suffix)
    return f"{path}__{suffix}"


def _apply_output_suffix(node: Any, suffix: str) -> None:
    """递归给输出路径键加后缀（原地改 ✓）。"""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and key in OUT_KEYS and "/" in value:
                node[key] = _suffix_path(value, suffix)
            else:
                _apply_output_suffix(value, suffix)
    elif isinstance(node, list):
        for item in node:
            _apply_output_suffix(item, suffix)
