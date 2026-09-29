# BOOM evaluation

`run_fracast.py` follows the official Datadog BOOM protocol: Gift-Eval's
rolling `Dataset`, GluonTS evaluation, 2,048-point context, and nine quantiles.
The pinned dataset revision is
`69325b544c45ff0d6c43c7a99c49a6601a01725b`; the pinned protocol revision is
`1527c41589189ad1bc3883ed4d3d97b3e5a3b47c`.

BOOM is the complete benchmark. BOOMLET is its official 32-query subset.
The runner writes the official `all_results.csv` schema and can resume after an
interruption by skipping dataset/term rows already present in that file.

Create the Python 3.11 evaluation environment once:

```bash
uv venv --python 3.11 tmp/boom-eval-venv
uv pip install --python tmp/boom-eval-venv/bin/python \
  -r requirements-boom.txt torch numpy safetensors
```

Download the pinned BOOMLET subset and run all 96 configurations:

```bash
env -u PYTHONPATH PYTHONPATH=. tmp/boom-eval-venv/bin/python \
  script/boom_bench/run_fracast.py \
  --benchmark boomlet --download --device mps --batch-size 64
```

For the complete benchmark, replace `boomlet` with `boom`. The full run
downloads the remaining partitions and evaluates 623,602 forecasts. `--model`
accepts a local release directory or a Hugging Face model id.

The archived full-BOOM result is in `results/Fracast-0/`. The public model
identifier in that archive is `Fracast-0`. The combined official-comparator
leaderboard is in `leaderboards/BOOM_leaderboard_with_fracast.csv`.
