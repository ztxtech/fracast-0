# pipeline/

This package contains the two supported flows.  A flow is orchestration only:
it maps a validated configuration to a model, data port, or corpus builder.
Model definitions live in `model/`, reusable blocks in `module/`, and data
format implementations in `dataport/`.

## Flows

| Kind | Entry | Responsibility |
| --- | --- | --- |
| `build_corpus` | `build_corpus.py` | raw Arrow/Parquet -> fast mmap corpus |
| `train` | `train.py` | Fracast pretraining, validation, checkpoints, resume |

`pipeline/pipeline.py` is the dispatcher.  It reads `_run.kind` and calls
exactly one `run(config)` function.

## Entry-point contract

- The only user-facing entry point is the repository-level `main.py`.
- Pipeline modules do not define `main()` or `argparse`.
- Pipeline parameters come from YAML, not from flow-specific command-line
  flags.
- Temporary configurations belong under `tmp/` and are passed to `main.py`.

The dispatcher exposes only these CLI controls:

```text
--list
--dry-run
-o KEY=VALUE
--workers N
```

## Directory layout

```text
pipeline/
  pipeline.py          dispatch by _run.kind
  build_corpus.py      corpus flow
  train.py             training flow
  policies/            sampling policies used by the training data loader
  template/            minimal scaffold for a new flow
```

## Boundaries

Do not place any of the following in this package:

- model definitions or inference adapters;
- reusable neural-network blocks;
- corpus readers or writers;
- GPU scheduling, experiment tracking, or evaluation services.
