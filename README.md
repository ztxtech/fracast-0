# FracCast

[![Repository](https://img.shields.io/badge/GitHub-fracast--0-181717?logo=github)](https://github.com/ztxtech/fracast-0)
[![Model](https://img.shields.io/badge/Hugging%20Face-fracast--0-FFD21E?logo=huggingface)](https://huggingface.co/ztxtech/fracast-0)
[![Demo](https://img.shields.io/badge/Space-fracast--0--demo-FFD21E?logo=huggingface)](https://huggingface.co/spaces/ztxtech/fracast-0-demo)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)

FracCast is a compact time-series foundation model for zero-shot forecasting.
The released model has **85,001 parameters** and keeps a single
full-resolution context stream.  Its core idea is simple: a causal
dilated-convolution block is reused across a geometric ladder of time scales,
with a small scale-conditioning signal at each level.  The model is
attention-free, fully convolutional, and designed to make the relationship
between parameter count and temporal context explicit.

This repository is the self-contained pretraining release.  It contains:

- the FracCast model and its reusable modules;
- the public-data download and fast-corpus preparation flow;
- the TinyCast synthetic-corpus build/conversion wrapper;
- a CPU/MPS smoke recipe and the full CUDA pretraining recipe;
- a single configuration-driven entry point, `main.py`.

The repository intentionally does **not** contain downloaded datasets,
checkpoints, evaluation products, or local runtime logs.  Those files are
created under `data/`, `output/`, and `tmp/`, all of which are gitignored.

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

- `model/fraccast/model.py`: block ordering and model assembly;
- `module/fraccast/`: reusable FracCast blocks;
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

## Quick smoke test

This path downloads nothing and takes only a few minutes on a laptop:

```bash
python script/data/make_demo_corpus.py
python main.py config/corpus/demo.yaml
python main.py config/fraccast/pretrain_smoke.yaml
```

The smoke recipe writes to `tmp/fracast-smoke/`.  It exercises the same
Arrow-to-fast-corpus path, model builder, data loader, rollout loss, checkpoint
writer, and resume logic used by the full recipe.

To test resume without changing the recipe:

```bash
python main.py config/fraccast/pretrain_smoke.yaml \
  -o train.total_steps=2 -o train.resume=false
python main.py config/fraccast/pretrain_smoke.yaml \
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
requirement explicit and then converts the shards to the FracCast fast-corpus
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

After all six corpus roots exist, start the full recipe:

```bash
python main.py config/fraccast/pretrain_full.yaml
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
python main.py config/fraccast/pretrain_local.yaml
```

It keeps the same data and effective batch size but uses a smaller micro-batch,
no autocast, no compilation, and no fused optimizer.  It is intended for
reproducibility and small local runs, not for a realistic wall-clock estimate.

### Resume

Training checkpoints contain the model, head, optimizer state, all random
streams, global step, and best validation value.  The default is:

```yaml
train:
  resume: auto
```

Resume from the current output directory:

```bash
python main.py config/fraccast/pretrain_full.yaml
```

Start a fresh run explicitly:

```bash
python main.py config/fraccast/pretrain_full.yaml -o train.resume=false
```

Resume from a named checkpoint:

```bash
python main.py config/fraccast/pretrain_full.yaml \
  -o train.resume=output/fracast-0-full/last.pt
```

## Configuration and overrides

Every flow is selected by `_run.kind` in YAML.  `main.py` only handles
discovery, dry runs, overrides, and optional process-level parallelism:

```bash
python main.py config/corpus --list
python main.py config/corpus/demo.yaml --dry-run
python main.py config/fraccast/pretrain_smoke.yaml \
  -o train.total_steps=8 -o train.log_every=1
```

The supported kinds are exactly:

| Kind | Meaning |
| --- | --- |
| `build_corpus` | raw Arrow/Parquet -> fast corpus |
| `train` | pretrain FracCast and write checkpoints |

No pipeline module defines its own CLI or `main()`.  Temporary configurations
should be written under `tmp/` and passed to `main.py`.

## Outputs

| Directory | Contents |
| --- | --- |
| `data/` | downloaded data and generated fast corpora |
| `output/` | full training runs and checkpoints |
| `tmp/` | smoke runs, temporary configs, and diagnostics |

All three directories are ignored by Git.  Checkpoints are PyTorch files and
are not committed.  The repository itself remains code, configuration, and
documentation only.

## Reproducibility

- Dataset revisions are pinned in the download script.
- `train.seed` controls Python, NumPy, PyTorch, data order, augmentation, and
  rollout sampling.
- `config_used.yaml` is written into every output run.
- The fast-corpus merge step checks manifest counts against actual offsets and
  lengths.
- `script/tests/` contains CPU-only structural, causal, parameter-count, and
  resume-alignment checks.

Run the release checks with:

```bash
python script/tests/test_freq.py
python script/tests/test_fraccast_unit.py
python script/tests/test_head_future_conv.py
python script/tests/test_resume_stream.py
```

## Third-party code

FracCast includes small, explicitly documented portions adapted from TinyCast
under the Apache-2.0 license.  The upstream commit, source paths, and local
locations are recorded in `THIRD_PARTY_NOTICES.md`; the license text is in
`LICENSES/TinyCast-Apache-2.0.txt`.
