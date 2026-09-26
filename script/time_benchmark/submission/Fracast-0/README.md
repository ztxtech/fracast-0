# Fracast-0 TIME submission

This directory contains the 98 TIME benchmark task results for `Fracast-0`.

## Evaluation settings

- Model weights: `ztxtech/fracast-0`, FP32.
- Parameter count: 85,001.
- Evaluation device: Apple MPS; batch size: 512.
- Quantile levels: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`.
- Released forecast head: 48 steps.
- Horizons longer than 48 steps: the local runner rolls the released head forward with median-quantile feedback.
- 47 of 98 tasks use this rollout path.

Each task directory contains the official TIME saver outputs:
`config.json`, `metrics.npz`, and `predictions.npz`. The overall numbers were
aggregated by seasonal-naive normalization over `(dataset_id, horizon)` and a
geometric mean over MASE and CRPS.

The long-horizon results are a local rollout evaluation and are not an
official leaderboard submission protocol claim.
