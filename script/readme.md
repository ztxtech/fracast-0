# script/

Repository tools are grouped by purpose:

| Directory | Purpose |
| --- | --- |
| `data/` | download public data, create demo data, build TinyCast synthetic shards |
| `corpus/` | convert, merge, and audit fast corpus parts |
| `gift_eval/` | run the official 97-configuration leaderboard submission |
| `fev_bench/` | archive the official 100-task FEV-Bench submission and rankings |
| `benchmark_inference.py` | measure release load time, latency, RSS, and throughput |
| `reproduce.sh` | one-command data preparation, smoke, full training, and tests |
| `tests/` | CPU-only structural and resume checks |

All tools use repository-relative paths.  Python entry points are run from the
repository root.

The corpus conversion flow itself is not a script-level entry point; it is
dispatched by `main.py` using `config/corpus/*.yaml`.
