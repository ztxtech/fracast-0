# Periodic Encoding

Period detection and phase-encoding components.

## Contents

- `official_periodogram.py`: normalized-periodogram period detector.
- `official_encoding.py`: phase and bounded-recency encoding primitives.
- `encoder.py`: multi-resolution integration used by Fracast.
- `seasonal_fill.py`: seasonal fill utilities for forecast decoding.

## Organization

- Files prefixed with `official_` are kept as reference copies and should only
  be changed when the upstream revision is deliberately updated.
- Keep experiment outputs in `output/` and datasets in `data/`.
