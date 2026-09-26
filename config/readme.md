# config/

Configuration is split by flow family:

| Directory | Purpose |
| --- | --- |
| `corpus/` | raw-data to fast-corpus conversion |
| `fracast/` | model pretraining recipes |

`base.yaml` contains shared defaults.  Child files inherit it with a relative
`inherit:` path and only list the keys they change.

Priority, from low to high:

1. inherited parent configuration;
2. keys in the selected YAML file;
3. `-o key.path=value` overrides passed to `main.py`.

Top-level keys beginning with `_` are metadata.  `_run.kind` selects the
pipeline, and `_run.grid` expands a configuration into independent runs.
Metadata is stripped before the flow sees the configuration.

Use `--dry-run` to validate the inheritance chain and inspect the expanded
runs without creating outputs.
