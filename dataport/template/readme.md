# Data adapter template

This directory contains the minimal skeleton for a new data source adapter.
Copy `dataport.py`, rename it, and implement the source-specific reader.

## Contents

- `dataport.py`: functional `read_dataset(root, cfg)` and optional
  `build_loaders(cfg)` hooks.

Raw data belongs in `data/`; completed adapters belong directly under
`dataport/`.  Add the new adapter to the dispatch in `dataport/dataport.py` and
to `dataport/readme.md`.
