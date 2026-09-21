"""配置加载：YAML → 嵌套 dict，支持 inherit 继承与 CLI 覆盖（key.path=value）。

三层优先级（低 → 高）：
  1. `inherit:` 指向的父配置（通常是 config/base.yaml 的默认值）
  2. 本文件的键（只写增量即可）
  3. 入口的 `-o key.path=value` 覆盖（主入口 main.py ✓）

（历史注：以前的“版本 spec 默认值”层已随 V 体系退场 ✗ —— 模型只有一个 `class Model` ✓）

约定（与 TEFN 的 config/ 思路一致，见 config/readme.md）：
  - **一个实验一个配置文件**，放在 config/<族>/<名字>.yaml，可 diff、可 grep、可归档；
  - 以 `_` 开头的顶层键是**元数据**（`_run: {...}` / `_comment: ...`），
    加载时被剥离，不会污染训练/评测读到的配置，也不会写进 config_used.yaml；
    入口（main.py）单独读它们来知道「这个配置跑什么、在哪个卡、网格是什么」；
    流程侧拿到的是 `strip_meta()` 剥完元数据的纯配置 ✓。
  - `inherit` 是**路径**：相对本文件所在目录解析，找不到再相对仓库根解析。

解析后的完整配置由 trainer 落盘为 `<out_dir>/config_used.yaml` —— 那是实验档案，
即使父配置后来改了，也能精确复现当时的结构与超参。
"""
from __future__ import annotations



from pathlib import Path
from typing import Any
import yaml

# 仓库根：util/config.py → parents[1]
_PROJ = Path(__file__).resolve().parents[1]

# 元数据前缀：这些顶层键不进入配置主体
META_PREFIX = "_"


def deep_merge(base: dict, over: dict) -> dict:
    """递归合并：嵌套 dict 逐键合并，其余类型直接覆盖。"""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_path(path_like: str, base_dir: Path) -> Path:
    """解析 inherit 路径：绝对路径直用；否则先试相对本文件，再试相对仓库根。"""
    p = Path(path_like)
    if p.is_absolute():
        return p
    cand = base_dir / p
    if cand.exists():
        return cand
    return _PROJ / p


def strip_meta(cfg: dict) -> dict:
    """剥掉顶层元数据键（`_run` / `_comment` …）→ 流程只看到纯配置 ✓。

    入口 main.py 靠 `_run` 做分派，流程实现不该看到它（也不会写进 config_used.yaml）。
    """
    return {k: v for k, v in cfg.items() if not str(k).startswith(META_PREFIX)}


def read_meta(path: str | Path) -> dict:
    """读取配置文件里的元数据块（`_run` / `_comment` 等），供入口/脚本使用。

    只读本文件的顶层 `_*` 键；不解析 inherit（元数据不继承，
    每个实验文件自报「跑什么、在哪张卡」）。
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return {k: v for k, v in raw.items() if str(k).startswith(META_PREFIX)}


def load_config(path: str | Path, overrides: list[str] | None = None,
                _seen: set | None = None) -> dict:
    """加载配置：inherit 链 → 本文件键 → `-o` 覆盖（优先级依次升高 ✓）。

    Parameters
    ----------
    path
        配置文件路径（如 `config/base.yaml` 或
        `config/fraccast/pretrain_full.yaml`）。
    overrides
        `key.path=value` 形式的覆盖项（优先级最高 ✓）。
    _seen
        内部用：inherit 链去重，检测循环继承。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    _seen = _seen or set()
    real = path.resolve()
    if real in _seen:
        raise ValueError(f"inherit 循环: {path}")
    _seen.add(real)

    # 1) 剥离元数据（`_run` / `_comment` 等），不进入配置主体
    cfg = strip_meta(yaml.safe_load(path.read_text()) or {})
    # 2) inherit：父配置（低） → 本文件键（高）
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(_resolve_path(str(parent), path.parent), None, _seen)
        cfg = deep_merge(base, cfg)          # 本文件的键压过父链
    # 3) `-o` 覆盖（最高优先级）
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg: dict, items: list[str] | None) -> dict:
    """按 `key.path=value` 写配置（原地改并返回 ✓，优先级最高）。

    入口（main.py）与 load_config 共用这一份实现 —— 两处各写一遍就会漂移 ✗。
    """
    for ov in items or []:
        if "=" not in ov:
            raise ValueError(f"override 格式应为 key.path=value，收到: {ov}")
        key, raw = ov.split("=", 1)
        keys = key.split(".")
        node: Any = cfg
        for part in keys[:-1]:
            nxt = node.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise TypeError(f"覆盖路径 {key!r} 冲突：{part!r} 已经是 {type(nxt).__name__}")
            node = nxt
        # note/描述类字段按原字符串存（yaml 会把 "xx: yy" 解析成 dict）
        node[keys[-1]] = raw if keys[-1] in ("note", "notes", "desc") else _coerce(raw)
    return cfg


def _coerce(v: str):
    """把字符串字面量转成 bool/int/float/None/list。"""
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
    # Kept only for compatibility with older notebooks. FracCast's exact count is
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
    """将配置字典导出为 JSON 文件，自动创建父目录（模板 util 口径 ✓）。"""
    from collections.abc import Mapping  # noqa: F401
    from util.io import save_json
    return save_json(path, dict(config))
