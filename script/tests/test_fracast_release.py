"""CPU tests for the public Fracast release loader and shipped weights."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_TEST_HORIZON = 144


def main() -> int:
    from fracast import FracastModel

    weights_dir = ROOT / "weights"
    config = json.loads((weights_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model"]["family"] == "fracast"

    fp32 = FracastModel.from_pretrained(weights_dir, weights="fp32", device="cpu")
    w8 = FracastModel.from_pretrained(weights_dir, weights="w8", device="cpu")
    assert fp32.context_length == 2048
    assert fp32.horizon == 48
    assert fp32.quantiles == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert sum(p.numel() for p in fp32.core.parameters()) + sum(
        p.numel() for p in fp32.head.parameters()
    ) == 85_001
    assert sum(p.numel() for p in w8.core.parameters()) + sum(
        p.numel() for p in w8.head.parameters()
    ) == 85_001
    bundled = FracastModel.from_pretrained(weights="fp32", device="cpu")
    bundled_w8 = FracastModel.from_pretrained(device="cpu")

    context = np.sin(np.arange(240, dtype=np.float32) / 7.0)
    single = w8.forecast(context)
    packaged_single = bundled_w8.forecast(context)
    batch = w8.forecast_batch(context[None, :])
    assert single.shape == (48, 9)
    assert batch.shape == (1, 48, 9)
    assert np.allclose(single, batch[0], rtol=1e-5, atol=1e-5)
    assert np.isfinite(single).all()
    assert np.allclose(packaged_single, single, rtol=1e-5, atol=1e-5)

    channels = np.stack([context, context * 0.5 + 2.0])
    multivariate = w8.forecast(channels)
    assert multivariate.shape == (2, 48, 9)
    assert np.allclose(multivariate[0], single, rtol=1e-5, atol=1e-5)

    long_single = w8.forecast(context, horizon=MAX_TEST_HORIZON)
    long_batch = w8.forecast_batch(channels, horizon=MAX_TEST_HORIZON)
    assert long_single.shape == (MAX_TEST_HORIZON, 9)
    assert long_batch.shape == (2, MAX_TEST_HORIZON, 9)
    assert np.allclose(long_single[:48], single, rtol=1e-5, atol=1e-5)
    assert np.isfinite(long_single).all() and np.isfinite(long_batch).all()
    try:
        w8.forecast(context, horizon=0)
    except ValueError:
        pass
    else:
        raise AssertionError("horizon=0 must raise ValueError")

    bundled_forecast = bundled.forecast(context)
    fp32_forecast = fp32.forecast(context)
    assert bundled_forecast.shape == fp32_forecast.shape == (48, 9)
    assert np.allclose(bundled_forecast, fp32_forecast, rtol=1e-5, atol=1e-5)
    print("[PASS] release loader, shapes, parameter count, and channel independence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
