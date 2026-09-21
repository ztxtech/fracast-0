# dataport/

`dataport/` is the data-access layer for pretraining.  It contains the corpus
format implementations and the single training-side loader entry point.

## Files

| File | Responsibility |
| --- | --- |
| `dataport.py` | `build_train_loaders(cfg)` selects and constructs the training loaders |
| `build_corpus.py` | Arrow/Parquet -> contiguous mmap arrays with offsets and manifests |
| `corpus_dataset.py` | fast-corpus reader with O(1) variable-length row access |
| `shard_dataset.py` | compact-shard reader and shared window construction |
| `prefetch.py` | optional background sequential prefetch for large shards |
| `template/` | minimal scaffold for a new data source |

## Fast corpus format

Each complete part contains:

```text
p<k>/
  shard0000.values.f32.npy
  shard0000.offsets.npy
  shard0000.lengths.npy
  shard0000.freq_id.npy
  shard0000.ds_id.npy
  shard0000.ts.npy
  index.npz
  manifest.json
  datasets.txt
  freqs.txt
```

`values` is one contiguous `float32` array.  `offsets` contains point offsets,
so a row can be loaded without scanning earlier rows.  Missing values are
preserved as `NaN`; the reader derives the validity mask from `isfinite`.

## Contracts

- Do not truncate series or drop missing values during conversion.
- Do not write raw data or generated corpora into Git.
- Do not add a CLI to this package; the entry point is `main.py`.
- Keep window construction shared between corpus layouts so training semantics
  do not drift.
- A part without `index.npz` is considered incomplete and is skipped by the
  training reader.

The corpus build flow is `pipeline/build_corpus.py`; the conversion
implementation is `dataport/build_corpus.py`.
