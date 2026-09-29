"""CPU checks for BOOM runner configuration and GluonTS adaptation."""
from __future__ import annotations

import csv
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "fracast_boom_runner", ROOT / "script" / "boom_bench" / "run_fracast.py"
)
runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)


def test_boomlet_config_expansion():
    properties = runner.load_properties(ROOT / "script" / "boom_bench" / "boomlet_properties.json")
    configs = list(runner.iter_configs(properties, "boomlet"))
    assert len(properties) == 32
    assert len(configs) == 96
    assert tuple(item.term for item in configs) == ("short", "medium", "long") * 32


def test_official_dataset_name_uses_property_frequency():
    properties = runner.load_properties(ROOT / "script" / "boom_bench" / "boomlet_properties.json")
    dataset, fields = next(iter(properties.items()))
    official_name = f"{dataset}/{fields['frequency']}/short"
    assert official_name.split("/")[1] == fields["frequency"]


def test_full_boom_config_expansion():
    properties = runner.load_properties(ROOT / "script" / "boom_bench" / "boom_properties.json")
    configs = list(runner.iter_configs(properties, "boom"))
    short_only = sum(properties[name]["term"] == "short" for name in properties)
    multi_term = len(properties) - short_only
    assert len(configs) == short_only + multi_term * 3


def test_short_term_properties_have_one_config():
    properties = runner.load_properties(ROOT / "script" / "boom_bench" / "boomlet_properties.json")
    assert all(properties[name]["term"] == "long" for name in properties)


def test_predictor_yields_nine_official_quantiles():
    import numpy as np
    import pandas as pd
    from gluonts.model.forecast import QuantileForecast

    class FakeModel:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        def forecast_batch(self, context, horizon):
            assert context.shape == (2, 4)
            assert horizon == 3
            return np.zeros((2, 3, 9), dtype=np.float32)

    predictor = runner.FracastPredictor(FakeModel(), prediction_length=3, batch_size=2)
    period = pd.Period("2026-01-01", freq="D")
    items = [
        {"start": period, "target": np.arange(4, dtype=np.float32), "item_id": f"item-{i}"}
        for i in range(3)
    ]
    forecasts = list(predictor.predict(items))
    assert len(forecasts) == 3
    assert all(isinstance(item, QuantileForecast) for item in forecasts)
    assert forecasts[0].forecast_keys == [str(value) for value in FakeModel.quantiles]
    assert forecasts[0].prediction_length == 3


def test_resume_reads_official_csv_schema():
    path = Path("/tmp") / "fracast-boom-resume-test.csv"
    runner.write_header(path)
    runner.append_result(path, ["ds-0-T/Short/short", "fracast-0", *([0.0] * 11), "[]", 1, 20])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert runner.completed_configs(path) == {"ds-0-T/Short/short"}
    assert rows[0]["eval_metrics/MASE[0.5]"] == "0.0"
    path.unlink()


def test_archived_full_boom_result():
    path = ROOT / "script" / "boom_bench" / "results" / "Fracast-0" / "all_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7413
    assert {row["model"] for row in rows} == {"Fracast-0"}
    assert all(row["dataset"].count("/") == 2 for row in rows)
    assert sum(int(row["dataset_size"]) for row in rows) == 623602
