# GIFT-Eval submission

`evaluate.py` runs the official 97-configuration protocol and produces the two
files required by a GIFT-Eval leaderboard pull request:

```text
results/Fracast-0/all_results.csv
results/Fracast-0/config.json
```

The evaluator uses the official `gift_eval.data.Dataset`, GluonTS
`evaluate_model`, and the standard 15-column result schema. Multivariate
datasets are expanded channel-wise, matching the official univariate
evaluation path. The submission is labeled `pretrained` because the released
checkpoint is applied as-is to every configuration.

The submission is labeled `testdata_leakage: "Yes"` under GIFT-Eval's
dataset-level rule. The LOtsa, Chronos, and FEV roots in the training recipe
contain datasets from the GIFT-Eval test corpus. `GiftEvalPretrain` itself is
published as a non-leaking pretraining set, but the complete recipe cannot be
labeled `No`.

The release model requires at least eight observed values in its last
2,048-point context. When an official window ends in a longer missing-value
run, the evaluator carries the nearest observed value into that tail before
the normal masked inference. No test labels or forecast values are changed.

## Requirements

Use Python 3.10 or newer and install the optional evaluator dependencies from
the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-gift-eval.txt
```

## Data

Pass a local copy of the official GIFT-Eval test datasets:

```bash
python script/gift_eval/evaluate.py \
  --data-root /path/to/gift_eval_raw
```

Alternatively, set `GIFT_EVAL` and omit `--data-root`. The directory must
contain the official dataset folders such as `m4_weekly` and
`electricity/15T`.

## Run

The FP32 release checkpoint and 512-window batches are the defaults:

```bash
python script/gift_eval/evaluate.py --data-root /path/to/gift_eval_raw
```

The CSV is flushed after every configuration, so the command can be restarted
without repeating completed work. `config.json` is written only after all 97
configurations validate.

For a single-configuration smoke run:

```bash
python script/gift_eval/evaluate.py \
  --data-root /path/to/gift_eval_raw \
  --only m4_weekly/W/short
```

Smoke output stays in the selected `--output-dir`; use a directory outside the
submission result when performing diagnostic runs.
