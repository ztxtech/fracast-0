# FEV-Bench result archive

This directory archives the complete local Fracast-0 FEV-Bench run. It contains the
official evaluation adapter, the pinned evaluation dependency, the raw 100-task CSV,
raw and leakage-controlled comparison tables, and a deterministic reproduction entry.

The source benchmark is `autogluon/fev` at commit
`81cf1255bb0c88dc039ae9bca23f73db6d9dfa61`. The run used Apple MPS, batch size 512,
and the `w8` weights. Its SHA256 identity is recorded in `run-manifest.json`.

Fracast-0 was pretrained on every `autogluon/fev_datasets` configuration in this
benchmark. Therefore the official controlled mode replaces all Fracast-0 errors with
Chronos-Bolt and the resulting aggregate must not be read as an independent zero-shot
score. The protocol and comparison table are in the
[FEV-Bench leaderboard](https://huggingface.co/spaces/autogluon/fev-bench) and the
[FEV-Bench repository](https://github.com/autogluon/fev).

572 of 235,039 sequence windows (0.243%) had fewer than eight finite observations after
the official task slicing. The wrapper deterministically repeats the most recent finite
value for those windows; it does not read labels or alter forecast values. No task
failed, and all 100 rows contain finite `SQL` and `MASE`.

## Reproduce evaluation

Install the model and evaluation dependencies first:

```bash
python -m pip install -r requirements.txt
python -m pip install -r models/fracast-0/requirements.txt
```

Use a local checkout of official FEV at the pinned commit. The script stages the adapter
inside that checkout, writes the result under the ignored `output/` directory, and never
starts a remote evaluation:

```bash
git clone https://github.com/autogluon/fev.git tmp/fev
git -C tmp/fev checkout 81cf1255bb0c88dc039ae9bca23f73db6d9dfa61
./script/fev_bench/reproduce.sh tmp/fev output/fev_bench
```

Set `FEV_DEVICE=cpu` or `FEV_DEVICE=cuda` when reproducing on another local device. The
archived run used the default `FEV_DEVICE=mps`. This command takes about one hour on
Apple MPS and downloads the benchmark datasets through Hugging Face.

## Reproduce rankings

With the same pinned FEV checkout installed in the active Python environment, rebuild
every archived leaderboard table:

```bash
python script/fev_bench/analyze.py --fev-repo tmp/fev --out-dir output/fev_analysis
```

`analyze.py` excludes the redundant `Toto-2.0-4m`, `Toto-2.0-313m`, and
`Toto-2.0-1B` sizes exactly as the published display leaderboard does. Raw mode reports
Fracast-0 as submitted. Controlled mode follows the official leakage rule for models
whose training corpus overlaps the benchmark.
