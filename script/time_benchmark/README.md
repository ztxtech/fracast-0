# Fracast-0 TIME benchmark archive

This directory archives the reproducible analysis used for the Fracast-0
[TIME benchmark](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard)
submission. Evaluation ran locally on Apple MPS with batch size 512. Publication
of this directory does not start a remote evaluation or test.

The official result submission is the Hugging Face dataset pull request
[`Real-TSF/TIME-Output#42`](https://huggingface.co/datasets/Real-TSF/TIME-Output/discussions/42),
not a GitHub pull request. Its head is
`c04be85a3108f901d5a656f3ee7ec5f50baf1f77` and it remains open. The raw
98-task output is kept there because each task contains `config.json`,
`metrics.npz`, and `predictions.npz`; this archive stores the compact public
analysis instead of duplicating those files.

## Results

The normalized scores aggregate seasonal-naive baselines over
`(dataset_id, horizon)` and take a geometric mean over MASE and CRPS.

| Scope | MASE norm | CRPS norm | MASE rank | CRPS rank |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.767965 | 0.649192 | 24 / 29 | 24 / 29 |
| Short | 0.701284 | 0.586018 | 25 / 29 | 24 / 29 |
| Medium | 0.855065 | 0.736344 | 25 / 29 | 24 / 29 |
| Long | 0.833422 | 0.708427 | 24 / 29 | 20 / 29 |

The evaluation completed 98 of 98 tasks. Of those, 47 use the released
48-step forecast head with median-quantile feedback for horizons longer than
48. Those rollouts are disclosed in every task configuration and are not a
strict official-protocol claim. The score may be recomputed after benchmark
review.

## Parameter Pareto analysis

For each axis, lower parameter count and lower normalized score are better.
A model `x` dominates `y` when `x.parameter_count <= y.parameter_count` and
`x.metric < y.metric`. Fracast-0 has 85,001 parameters and is non-dominated on
both parameter-MASE and parameter-CRPS. Its dominator lists are empty. The
smallest better-performing comparator is Toto-2.0-4m with 4,144,456
parameters, MASE 0.691002, and CRPS 0.582490.

Parameter counts are exact tensor or `state_dict` counts where public weights
exist. `TIME_pareto.csv` records the count kind and evidence for every model.
Two entries are estimates: OmniScient uses the Chronos-2 base as a conservative
lower bound, and PatchTST-FM-Extended uses the public R1 count from the same
20-layer, d=1024 architecture.

## Reproduce the analysis

Pin the official TIME code first:

```bash
git clone https://github.com/zqiao11/TIME.git
git -C TIME checkout 3ca5c41d71c76f3c70632c0121142d325350ee32
```

Run the official implementation and this repository from local paths only:

```bash
python -m pip install -e .
python script/time_benchmark/run_fracast.py \
  --dataset all_datasets \
  --time-source /path/to/TIME \
  --package-source . \
  --output-dir output/results \
  --data-root data/time \
  --weights-dir weights \
  --device mps \
  --batch-size 512 \
  --skip-existing
```

The runner does not download data unless `--download` is explicitly set.
After an evaluation, validate the raw task layout and regenerate the public
tables with:

```bash
python script/time_benchmark/validate_submission.py output/results/Fracast-0 \
  --expected dataset_id/freq/term ...
python script/time_benchmark/analyze_pareto.py
```

`validate_submission.py` expects each official task directory to contain the
three TIME saver files and the provenance keys recorded in the submission
README.

## Archive contents

- `run_fracast.py`: official TIME Dataset/saver integration for Fracast-0.
- `validate_submission.py`: raw task-layout and finite-metric checks.
- `analyze_pareto.py`: parameter evidence and deterministic Pareto computation.
- `analysis/`: overall, horizon, per-task, and Pareto result tables.
- `submission/Fracast-0/README.md`: public disclosure attached to the raw
  submission.
- `run-manifest.json`: pinned versions, hashes, scope, and submission state.

The analysis and disclosure artifacts are covered by
`script/tests/test_time_benchmark_artifacts.py`.
