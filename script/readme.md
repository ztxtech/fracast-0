# script/

Repository tools are grouped by purpose:

| Directory | Purpose |
| --- | --- |
| `data/` | download public data, create demo data, build TinyCast synthetic shards |
| `corpus/` | convert, merge, and audit fast corpus parts |
| `tests/` | CPU-only structural and resume checks |

All tools use repository-relative paths.  Python entry points are run from the
repository root.

The corpus conversion flow itself is not a script-level entry point; it is
dispatched by `main.py` using `config/corpus/*.yaml`.
