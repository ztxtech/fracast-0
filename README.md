# Fracast-0

[![Project page](https://img.shields.io/badge/Project-GitHub_Pages-7C3AED?logo=github)](https://ztxtech.github.io/fracast-0/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.32209-b31b1b?logo=arxiv)](https://arxiv.org/abs/2609.32209)
[![Repository](https://img.shields.io/badge/GitHub-fracast--0-181717?logo=github)](https://github.com/ztxtech/fracast-0)
[![PyPI](https://img.shields.io/pypi/v/fracast?label=PyPI&color=3776AB)](https://pypi.org/project/fracast/)
[![Model](https://img.shields.io/badge/Hugging%20Face-fracast--0-FFD21E?logo=huggingface)](https://huggingface.co/ztxtech/fracast-0)
[![Demo](https://img.shields.io/badge/Space-fracast--0--demo-FFD21E?logo=huggingface)](https://huggingface.co/spaces/ztxtech/fracast-0-demo)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)

---

Fracast-0 is an **85K-parameter pretrained Time Series Foundation Model**, small
enough to load directly in a web page.

<div align="center">
  <em>&ldquo;We built this solely to explore whether model compression can be pushed to an even more extreme state. We spent 15 days on this exploration. Although the work is not perfect, it is at least usable, so we are releasing it. Fracast-0 is the published version, and future versions will only get better.&rdquo;</em>
  <br /><br />
  <strong>Tianxiang Zhan</strong>
</div>

---

Fracast-0 is a compact time-series foundation model for forecasting without
per-dataset fine-tuning.
The released model has **85,001 parameters** and keeps a single
full-resolution context stream.  Its core idea is simple: a causal
dilated-convolution block is reused across a geometric ladder of time scales,
with a small scale-conditioning signal at each level.  The model is
attention-free, fully convolutional, and designed to make the relationship
between parameter count and temporal context explicit.

This repository is the self-contained pretraining release.  It contains:

- the Fracast-0 model and its reusable modules;
- the public-data download and fast-corpus preparation flow;
- the TinyCast synthetic-corpus build/conversion wrapper;
- a CPU/MPS smoke recipe and the full CUDA pretraining recipe;
- a single configuration-driven entry point, `main.py`.

The repository includes the small release checkpoints under `weights/`.
Downloaded datasets, training runs, evaluation products, and runtime logs are
created under `data/`, `output/`, and `tmp/`, all of which are gitignored.

## Installation

Install the Python package and its bundled Fracast-0 checkpoints with one
command:

```bash
python -m pip install fracast
```

The package is self-contained: `FracastModel.from_pretrained()` reads the
FP32 or W8 weights from the installed wheel without downloading a model
repository. The GitHub workflow builds every push to `main` and publishes a
new `pyproject.toml` version to PyPI when that version is not already online.

## Model

The default model is configured in `config/base.yaml`.

| Component | Default |
| --- | --- |
| Context stream | one full-resolution stream |
| Core block | shared causal depthwise-separable dilated convolution + SwiGLU |
| Scale ladder | dilation `1, 2, 4, ...` |
| Scale conditioning | `ScaleCondition` FiLM, identity-initialized |
| Periodic structure | zero-parameter normalized-periodogram phase features |
| Decoder | gather/query head with an optional future-state convolution |
| Quantiles | 9 levels from 0.1 through 0.9 |
| Horizon | 48 steps |
| Parameters | **85,001** in the release configuration |

The implementation is split by responsibility:

- `model/fracast/model.py`: block ordering and model assembly;
- `module/fracast/`: reusable Fracast-0 blocks;
- `module/periodic/`: period detection, phase encoding, seasonal fill;
- `dataport/`: corpus readers and the training data port;
- `pipeline/`: the two supported flows, `build_corpus` and `train`;
- `script/data/`: public-data download and TinyCast synthetic-shard tooling;
- `script/corpus/`: corpus conversion, merge, and audit tools.

## Requirements

Python **3.10 or newer** is recommended.  TinyCast's upstream package requires
Python 3.10+, so the TinyCast synthetic-corpus path needs it.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core requirements are installable on Python 3.9.  Rebuilding the TinyCast
synthetic shards additionally requires Python 3.10+ and the pinned upstream
package:

```bash
python -m pip install -r requirements-tinycast.txt
```

The training code chooses `CUDA > MPS > CPU` automatically.  CUDA is the
intended pretraining device.  MPS and CPU are supported for the smoke recipe
and for local development.  Mixed-precision autocast is enabled only on CUDA.

## One-command reproduction

After creating and activating the virtual environment, the complete public
data path and the full recipe are:

```bash
./script/reproduce.sh prepare-data --workers 8
./script/reproduce.sh train
```

The first command downloads the five pinned source datasets, converts and
merges all five fast corpora, builds the four TinyCast synthetic shards, and
runs the corpus audits.  It is resumable: completed corpus roots are skipped,
while an incomplete generated root is rebuilt from its raw source.

For a quick local check that downloads nothing:

```bash
./script/reproduce.sh run --profile smoke
```

Both commands use `PYTHON=python` by default.  Set `PYTHON=.venv/bin/python`
if the interpreter is not already active.

## Quick smoke test

The underlying smoke commands are also directly runnable:

```bash
python script/data/make_demo_corpus.py
python main.py config/corpus/demo.yaml
python main.py config/fracast/pretrain_smoke.yaml
```

The smoke recipe writes to `tmp/fracast-smoke/`.  It exercises the same
Arrow-to-fast-corpus path, model builder, data loader, rollout loss, checkpoint
writer, and resume logic used by the full recipe.

To test resume without changing the recipe:

```bash
python main.py config/fracast/pretrain_smoke.yaml \
  -o train.total_steps=2 -o train.resume=false
python main.py config/fracast/pretrain_smoke.yaml \
  -o train.total_steps=4 -o train.resume=auto
```

The second command must report a restored optimizer state and random-number
state, then continue from step 2.

## Public pretraining data

The five source datasets are pinned to immutable Hugging Face revisions in
`script/data/download_public_data.py`.  Their approximate download sizes are:

| Name | Hugging Face dataset | Approx. size |
| --- | --- | ---: |
| `pretrain` | `Salesforce/GiftEvalPretrain` | 908.1 GiB |
| `lotsa` | `Salesforce/lotsa_data` | 861.2 GiB |
| `chronos` | `autogluon/chronos_datasets` | 833.3 GiB |
| `boom` | `Datadog/BOOM` | 2.6 GiB |
| `fev` | `autogluon/fev_datasets` | 0.6 GiB |

The complete download is about **2.6 TiB** before conversion.  Check available
storage and network capacity before starting:

```bash
python script/data/download_public_data.py --list
python script/data/download_public_data.py pretrain lotsa chronos boom fev
# or: python script/data/download_public_data.py --all
```

Downloads go to:

```text
data/pretrain_full/
data/lotsa_full/
data/chronos_full/
data/boom/
data/fev/
```

BOOM is both a pretraining corpus and an independent evaluation benchmark. To
run the official BOOMLET subset after pretraining, see
[`script/boom_bench/readme.md`](script/boom_bench/readme.md). The runner uses
the official rolling protocol, supports resumable full-BOOM evaluation, and
writes the same `all_results.csv` schema used by Datadog's leaderboard code.

### Convert raw data to fast corpus

Each source is converted independently.  The conversion produces contiguous
`float32` arrays, `int64` offsets, compact indices, and a manifest.  The
conversion is CPU/IO work and does not need a GPU:

```bash
python main.py config/corpus/pretrain_full.yaml --workers 32
python main.py config/corpus/lotsa_full.yaml --workers 32
python main.py config/corpus/chronos_full.yaml --workers 32
python main.py config/corpus/boom_full.yaml --workers 8
python main.py config/corpus/fev_full.yaml --workers 4
```

After all parts finish, merge each source into a global index:

```bash
python script/corpus/merge_corpus_parts.py --corpus data/corpus_fast/pret
python script/corpus/merge_corpus_parts.py --corpus data/corpus_fast/lotsa
python script/corpus/merge_corpus_parts.py --corpus data/corpus_fast/chronos
python script/corpus/merge_corpus_parts.py --corpus data/corpus_fast/boom
python script/corpus/merge_corpus_parts.py --corpus data/corpus_fast/fev
```

The converter only accepts complete `p*/index.npz` parts.  A partially written
part is skipped by the training reader and can be rebuilt without touching the
other parts.

### TinyCast synthetic shards

TinyCast publishes four synthetic shards.  The official builder is CUDA-only
and requires the optional TinyCast package; the wrapper below keeps that
requirement explicit and then converts the shards to the Fracast-0 fast-corpus
layout:

```bash
python script/data/prepare_tinycast_synth.py
```

If the shards were already built, convert them without rebuilding:

```bash
python script/data/prepare_tinycast_synth.py --skip-build
```

The output is `data/corpus_fast/tinycast_synth/`.  The wrapper preserves the
official per-series scale sidecar used by the committing objective.

## Full pretraining

After all six corpus roots exist, start the full recipe through the same
reproduction entry point:

```bash
./script/reproduce.sh train
```

The full recipe uses:

- six corpus roots: `pret`, `lotsa`, `chronos`, `boom`, `fev`, `tinycast_synth`;
- micro-batch 512 with 8 gradient-accumulation steps;
- effective batch 4,096;
- 36,621 optimizer steps;
- CUDA bf16 autocast, `torch.compile`, fused AdamW, TF32, and cuDNN benchmark;
- checkpoints every 1,145 steps plus `best.pt`, `last.pt`, and the final
  averaged checkpoint.

On a Mac, use the local recipe instead of the full CUDA recipe:

```bash
./script/reproduce.sh train --profile local
```

`config/fracast/pretrain_local.yaml` is the full-data CPU/MPS recipe for a
machine that has already prepared all six corpus roots; it uses a smaller
micro-batch and disables CUDA-only optimizations.  The smoke profile remains
the fastest way to verify installation without downloading the corpora.

### Resume

Training checkpoints contain the model, head, optimizer state, all random
streams, global step, and best validation value.  The default is:

```yaml
train:
  resume: auto
```

Resume from the current output directory:

```bash
python main.py config/fracast/pretrain_full.yaml
```

Start a fresh run explicitly:

```bash
python main.py config/fracast/pretrain_full.yaml -o train.resume=false
```

Resume from a named checkpoint:

```bash
python main.py config/fracast/pretrain_full.yaml \
  -o train.resume=output/fracast-0-full/last.pt
```

## Configuration and overrides

Every flow is selected by `_run.kind` in YAML.  `main.py` only handles
discovery, dry runs, overrides, and optional process-level parallelism:

```bash
python main.py config/corpus --list
python main.py config/corpus/demo.yaml --dry-run
python main.py config/fracast/pretrain_smoke.yaml \
  -o train.total_steps=8 -o train.log_every=1
```

The supported kinds are exactly:

| Kind | Meaning |
| --- | --- |
| `build_corpus` | raw Arrow/Parquet -> fast corpus |
| `train` | pretrain Fracast-0 and write checkpoints |

No pipeline module defines its own CLI or `main()`.  Temporary configurations
should be written under `tmp/` and passed to `main.py`.

## Python inference and benchmark

The installed package loads its bundled W8 checkpoint by default:

```python
import numpy as np
from fracast import FracastModel

model = FracastModel.from_pretrained(device="cpu")
context = np.sin(np.arange(240, dtype=np.float32) / 7.0)
forecast = model.forecast(context)
assert forecast.shape == (48, 9)

# The native head predicts 48 steps. Longer horizons append median blocks and
# re-normalize each 2,048-point context before the next forward pass.
long_forecast = model.forecast(context, horizon=144)
assert long_forecast.shape == (144, 9)

# Independent channels can share one batch invocation.
batch = model.forecast_batch(
    np.stack([context, context * 0.5 + 2.0]), horizon=144
)
assert batch.shape == (2, 144, 9)
```

Pass `weights="fp32"` to use the bundled full-precision checkpoint. The first
argument may also be a local checkpoint directory or a Hugging Face repository
id such as `ztxtech/fracast-0`. Inputs are `[T]`, `[1,T]`, or `[V,T]`; outputs
are `[H,Q]`, `[1,H,Q]`, or `[V,H,Q]`, where `H` defaults to the native 48
steps and `Q=9`. The model uses the latest 2,048 observations, masks missing
values, and forecasts channels independently.

Run the release benchmark with:

```bash
./script/reproduce.sh benchmark
# or, for a custom device and batch:
python script/benchmark_inference.py --weights w8 --device cuda --batch 32
```

The JSON report is written under `output/benchmark/` and includes load time,
parameter count, resident RSS, p50/p95 latency, and series throughput.

## GIFT-Eval Benchmark

The official 97-configuration evaluator is available under
`script/gift_eval/`. The archived result and official aggregate are also
tracked there. The FP32 release checkpoint produces:

| Protocol split | Normalized MASE | Normalized MWQL |
| --- | ---: | ---: |
| Short (55) | 0.769573 | 0.568038 |
| Medium (21) | 0.842959 | 0.557024 |
| Long (21) | 0.875558 | 0.555953 |
| **Overall (97)** | **0.807133** | **0.563008** |

The official aggregate first divides each configuration by the matching
Seasonal_Naive result and then takes the geometric mean across all 97
configurations. The result is labeled `pretrained` with
`testdata_leakage: "Yes"` because the complete pretraining recipe contains
dataset families from the GIFT-Eval test corpus. The official results are
listed in the
[GIFT-Eval `results/Fracast-0` directory](https://github.com/SalesforceAIResearch/gift-eval/tree/main/results/Fracast-0)
and the [GIFT-Eval leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval).
The same two official files are archived under
`script/gift_eval/results/Fracast-0/`, and the split/overall summary is in
`script/gift_eval/analysis/protocol_summary.csv`.

Run the evaluator with Python 3.10 or newer:

```bash
python -m pip install -r requirements-gift-eval.txt
python script/gift_eval/evaluate.py --data-root /path/to/gift_eval_raw
```

For a from-scratch reproduction, direct its gitignored output to a temporary
directory:

```bash
python script/gift_eval/evaluate.py \
  --data-root /path/to/gift_eval_raw \
  --output-dir tmp/gift_eval_repro
```

## TIME Benchmark

The complete 98-task TIME evaluation is archived under
`script/time_benchmark/`. It includes the official runner, validation, pinned
analysis script, parameter-Pareto tables, and a run manifest. The archived run
used Apple MPS, batch size 512, and the FP32 release checkpoint; all 98 tasks
completed. The ranking table is the official 29-model
[TIME leaderboard](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard),
and the raw 98-task outputs are in
[`Real-TSF/TIME-Output`](https://huggingface.co/datasets/Real-TSF/TIME-Output).

| Scope | Normalized MASE | Normalized CRPS | MASE rank | CRPS rank |
| --- | ---: | ---: | ---: | ---: |
| Short | 0.701284 | 0.586018 | 25 / 29 | 24 / 29 |
| Medium | 0.855065 | 0.736344 | 25 / 29 | 24 / 29 |
| Long | 0.833422 | 0.708427 | 24 / 29 | 20 / 29 |
| **Overall** | **0.767965** | **0.649192** | **24 / 29** | **24 / 29** |

Fracast-0 is non-dominated on both parameter-MASE and parameter-CRPS fronts
with 85,001 parameters. Of the 98 tasks, 47 use the released 48-step head with
median-quantile feedback beyond 48 steps; this is disclosed in every raw task
configuration and is not presented as a strict official-protocol result. The
compact scores in this archive come from the archived raw outputs and the
official 29-model table; no separate recomputation is recorded for them.

## FEV-Bench

The complete local 100-task FEV-Bench submission is archived under
`script/fev_bench/`. It includes the official adapter, raw result CSV, nine
ranking tables, a pinned analysis script, and a reproduction entry. The run
used Apple MPS, batch size 512, FP32, and the `w8` checkpoint; all 100 tasks
completed without a task failure.

| Metric | Raw rank | Controlled rank |
| --- | ---: | ---: |
| SQL | 17 / 30 | 16 / 30 |
| MASE | 20 / 30 | 16 / 30 |
| WQL | 17 / 30 | 16 / 30 |
| WAPE | 20 / 30 | 17 / 30 |

The raw SQL win rate is 44.44% with a 35.72% skill score. Fracast-0 was
pretrained on every `autogluon/fev_datasets` configuration used by this
benchmark, so the official leakage-controlled aggregate replaces its errors
with Chronos-Bolt. It must not be presented as an independent zero-shot
result. The protocol and comparison table are in the
[FEV-Bench leaderboard](https://huggingface.co/spaces/autogluon/fev-bench) and
the [FEV-Bench repository](https://github.com/autogluon/fev).

572 of 235,039 sequence windows (0.243%) had fewer than eight finite
observations after official task slicing. The wrapper repeats the most recent
finite value for those windows and records each affected task. Reproduce from
the archive documentation with:

```bash
./script/fev_bench/reproduce.sh tmp/fev output/fev_bench
python script/fev_bench/analyze.py --fev-repo tmp/fev --out-dir output/fev_analysis
```

## BOOM Benchmark

The complete official BOOM protocol contains 7,413 dataset-term configurations
and 623,602 forecasts. Fracast-0 completed every configuration with the FP32
checkpoint and no task failures. In the official 24-model comparison, it
obtains scaled MASE `0.723`, scaled CRPS `0.434`, and mean per-dataset rank
`11.808`; the corresponding ordinal positions are `12 / 24` by MASE and
`11 / 24` by CRPS.

The pinned dataset revision, evaluator revision, hashes, and per-configuration
results are archived under `script/boom_bench/`. The combined comparator table
against the official 24-model table is in
[`script/boom_bench/leaderboards/BOOM_leaderboard_with_fracast.csv`](script/boom_bench/leaderboards/BOOM_leaderboard_with_fracast.csv).
The official protocol and comparison table are in the
[BOOM leaderboard](https://huggingface.co/spaces/Datadog/BOOM).

`Datadog/BOOM` is also part of the complete Fracast-0 pretraining recipe, so
this result is in-corpus and must not be presented as zero-shot.

## Scenario Highlights

These slices use per-configuration or per-task error ratios against Seasonal
Naive. Lower is better, and TIME ranks use the official 29-model table.

- **GIFT-Eval, Sales:** All four short-horizon Sales configurations improve on
  Seasonal Naive in normalized MASE and normalized MWQL. The geometric means
  are 0.693 and 0.422, and Fracast-0 leads TinyCast on both metrics in every
  configuration.
- **GIFT-Eval, Cloud operations:** Across the three hourly `bizitobs_l2c`
  short, medium, and long slices, normalized MASE is 0.430 and normalized MWQL
  is 0.353.
- **TIME, Solar forecasting:** `Australia_Solar/H` ranks 4th to 6th of 29 by
  MASE and 7th to 13th by CRPS over short, medium, and long horizons.
- **TIME, Manufacturing:** On `Smart_Manufacturing/H`, medium and long
  horizons rank 8th of 29 by MASE, while CRPS ranks range from 8th to 10th
  across all three horizons.
These are scenario slices, not separate aggregate rankings. The benchmark
disclosures above still apply.

## Outputs

| Directory | Contents |
| --- | --- |
| `data/` | downloaded data and generated fast corpora |
| `weights/` | committed FP32 and W8 release checkpoints |
| `output/` | full training runs and checkpoints |
| `tmp/` | smoke runs, temporary configs, and diagnostics |

All runtime directories are ignored by Git. The small release checkpoints and
the compact GIFT-Eval, TIME, FEV-Bench, and BOOM benchmark archives are
committed; PyTorch training checkpoints, raw benchmark workspaces, and
generated runtime outputs are not.

## Reproducibility

- Dataset revisions are pinned in the download script.
- `train.seed` controls Python, NumPy, PyTorch, data order, augmentation, and
  rollout sampling.
- `config_used.yaml` is written into every output run.
- The fast-corpus merge step checks manifest counts against actual offsets and
  lengths.
- `script/tests/` contains CPU-only structural, causal, parameter-count, and
  resume-alignment checks, plus public weight-loader coverage.

Run the release checks with:

```bash
python script/tests/test_freq.py
python script/tests/test_fracast_unit.py
python script/tests/test_head_future_conv.py
python script/tests/test_resume_stream.py
python script/tests/test_fracast_release.py
python script/tests/test_gift_eval_artifacts.py
python script/tests/test_fev_bench_artifacts.py
python script/tests/test_time_benchmark_artifacts.py
```

## Citation

```bibtex
@article{zhan2026fracast0fractalweightsharing,
  title={Fracast-0: Fractal Weight Sharing for a Time Series Foundation Model with Only 85K Parameters},
  author={Tianxiang Zhan and Huanyao Zhang and Yuanpeng He},
  year={2026},
  journal={arXiv preprint arXiv:2609.32209},
  eprint={2609.32209},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2609.32209},
  url={https://arxiv.org/abs/2609.32209}
}
```

## Acknowledgements

Fracast-0 depends on the teams behind the benchmark, compact baseline, and
public pretraining corpora:

- [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) provides the
  97-configuration evaluation protocol used for the public result.
- [FEV-Bench](https://github.com/autogluon/fev) provides the 100-dataset
  evaluation protocol and leaderboard submission format.
- [TIME](https://github.com/zqiao11/TIME) provides the 98-task evaluation
  workflow, output format, and leaderboard submission protocol.
- [TinyCast](https://github.com/raws-labs/tinycast) provides the compact
  design baseline and the Apache-2.0 components recorded in
  `THIRD_PARTY_NOTICES.md`.
- [Salesforce/GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain),
  [Salesforce/lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data),
  [autogluon/chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets),
  [Datadog/BOOM](https://huggingface.co/datasets/Datadog/BOOM), and
  [autogluon/fev_datasets](https://huggingface.co/datasets/autogluon/fev_datasets)
  provide the public training data.
- The TinyCast team also publishes the synthetic pretraining shards used by the
  recipe.

## Development support

The implementation, debugging, and release work used paid API access to
**Xiaomi MiMo 2.6 Pro** and **DeepSeek V4.1 Flash**. Both were effective for
this project and are recommended.

## Supporters / sponsors

There are no sponsors yet. Continued maintenance still needs API credits and
server resources. Feedback and support are welcome through:

- [zhantianxianguestc@hotmail.com](mailto:zhantianxianguestc@hotmail.com)

## Third-party code

Fracast-0 includes small, explicitly documented portions adapted from TinyCast
under the Apache-2.0 license.  The upstream commit, source paths, and local
locations are recorded in `THIRD_PARTY_NOTICES.md`; the license text is in
`LICENSES/TinyCast-Apache-2.0.txt`.
