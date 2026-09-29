# Third-Party Notices

Fracast contains code and a brand asset from the following sources.

## TinyCast

- Repository: https://github.com/raws-labs/tinycast
- Revision: `628c0bec8425240d5f12155add1d1d5bc2d92471`
- License: Apache-2.0
- Copyright: RAWS Labs

The local files below contain adapted or copied material.  The upstream
revision and source path are recorded in each file header where applicable.

| Local file | Upstream source | Nature of use |
| --- | --- | --- |
| `module/periodic/official_encoding.py` | `tinycast/encoding.py` | copied implementation |
| `module/periodic/official_periodogram.py` | `tinycast/periodogram.py` | copied implementation |
| `module/losses/tinycast.py` | `tinycast/losses.py`, `tinycast/scale.py` | adapted objectives |
| `module/fracast/future_conv.py` | `tinycast/backbone.py` | adapted decoder state path |
| `module/periodic/seasonal_fill.py` | `tinycast/backbone.py` | adapted seasonal fill |

The Fracast model, training pipeline, data port, corpus tools, and
configuration files are original work in this repository unless a file header
says otherwise.

## arXiv wordmark

- Source: https://arxiv.org/static/base/1.0.1/images/arxiv-logo-primary-light.svg
- Local file: `static/images/arxiv-logo-primary-light.svg`
- Nature of use: unmodified arXiv wordmark in the preprint link on the project page
