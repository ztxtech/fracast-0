# config/corpus/

Configurations for `kind: build_corpus`.

| Config | Raw source | Fast output |
| --- | --- | --- |
| `pretrain_full.yaml` | `data/pretrain_full` | `data/corpus_fast/pret` |
| `lotsa_full.yaml` | `data/lotsa_full` | `data/corpus_fast/lotsa` |
| `chronos_full.yaml` | `data/chronos_full` | `data/corpus_fast/chronos` |
| `boom_full.yaml` | `data/boom` | `data/corpus_fast/boom` |
| `fev_full.yaml` | `data/fev` | `data/corpus_fast/fev` |
| `demo.yaml` | `data/demo_raw` | `data/corpus_fast/demo` |

Each large source is split into independent parts with `_run.grid`.  The
`{grid}` placeholder in `out` keeps part outputs separate.

Example:

```bash
python main.py config/corpus/pretrain_full.yaml --workers 32
python script/corpus/merge_corpus_parts.py --corpus data/corpus_fast/pret
```

The TinyCast synthetic shards use `script/data/prepare_tinycast_synth.py`
because they require a per-series scale sidecar that is not present in the
generic Arrow/Parquet schema.
