# script/corpus/

Tools for converting, merging, and auditing fast corpora.

| File | Purpose |
| --- | --- |
| `merge_corpus_parts.py` | merge independent part directories into one global index |
| `audit_corpus_parts.py` | check part completeness, offsets, lengths, and dataset coverage |
| `audit_corpus.py` | compare local files with a Hugging Face source manifest |
| `audit_corpus_bytes.py` | independent byte/length/offset reconstruction |
| `build_synth_corpus.py` | convert TinyCast shards directly, preserving scale sidecars |

Typical workflow:

```bash
./script/reproduce.sh prepare-data
# or run the conversion stages explicitly:
python main.py config/corpus/pretrain_full.yaml --workers 32
python script/corpus/merge_corpus_parts.py --corpus data/corpus_fast/pret
python script/corpus/audit_corpus_parts.py --corpus pret=data/pretrain_full:32
```

The merge step checks that each part's manifest agrees with its actual
`offsets` and `lengths`.  The audit scripts are read-only and write reports
under `tmp/corpus_report/` unless an explicit output path is supplied.
