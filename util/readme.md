# util/

Small repository-local utilities with no model or data-format policy:

| File | Responsibility |
| --- | --- |
| `config.py` | YAML loading, inheritance, metadata stripping, CLI overrides |
| `grid.py` | deterministic Cartesian expansion of `_run.grid` |
| `seed.py` | Python/NumPy/PyTorch seed setup |
| `freq.py` | frequency parsing and seasonal-period lookup |
| `io.py` | JSON and directory helpers |
| `timing.py` | timers and timestamp formatting |
| `ckpt.py` | stable state-dict handling across compiled/uncompiled models |
| `paths.py` | repository-local `ROOT`, `DATA`, `OUTPUT`, `TMP`, `HF_CACHE` |

Utilities must remain side-effect free on import.  A capability should have
one implementation and be imported by its callers rather than copied.

CLI tools do not belong here; put them under `script/`.
