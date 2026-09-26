#!/usr/bin/env python3
"""Compute parameter-count Pareto fronts for the TIME leaderboard."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
LEADERBOARD = ANALYSIS / "TIME_overall_leaderboard.csv"


# parameter_kind:
# - exact: safetensors elements, state_dict shapes, or public same-architecture weights.
# - lower_bound: public base-model parameters; any adapter/LoRA only adds more.
# - rounded: a paper, model-card, or weight-name approximation.
PARAMETERS = {
    "TimesFM-3": (330_710_976, "exact", "HF safetensors element count"),
    "Toto-2.0-2.5B": (2_454_281_888, "exact", "HF safetensors element count"),
    "Toto-2.0-313m": (312_684_656, "exact", "HF safetensors element count"),
    "Toto-2.0-1B": (1_041_033_096, "exact", "HF safetensors element count"),
    "PatchTST-FM-Extended": (
        257_895_552,
        "rounded",
        "Public R1 same-architecture safetensors: 20 layers and d=1024 as in the paper",
    ),
    "Chronos-2": (119_477_664, "exact", "HF safetensors element count"),
    "t0_beta": (255_614_293, "exact", "HF safetensors element count"),
    "OmniScient": (
        119_477_664,
        "lower_bound",
        "Chronos-2 base parameters; an additional r=8 LoRA can only increase it",
    ),
    "TimesFM-2.5": (231_289_280, "exact", "HF safetensors element count"),
    "Toto-2.0-22m": (21_915_596, "exact", "HF safetensors element count"),
    "PatchTST-FM-R1": (257_895_552, "exact", "HF safetensors element count"),
    "Tabby": (146_695_591, "exact", "paris-noah/Tabby safetensors element count"),
    "TiRex": (35_000_000, "rounded", "Official model card reports 35M"),
    "Toto-2.0-4m": (4_144_456, "exact", "HF safetensors element count"),
    "TS-ICL": (
        34_724_011,
        "exact",
        "Public checkpoint forecaster state_dict shape count",
    ),
    "Toto": (151_306_080, "exact", "HF safetensors element count"),
    "Eidos": (
        113_352_192,
        "exact",
        "Checkpoint cfg points to TimeMoE-50M; public same-architecture safetensors element count",
    ),
    "Timer-S1": (8_303_654_176, "exact", "HF safetensors element count"),
    "Moirai2": (11_387_208, "exact", "HF safetensors element count"),
    "TimesFM-2.0": (498_828_960, "exact", "HF safetensors element count"),
    "Chronos-bolt": (205_292_928, "exact", "HF safetensors element count"),
    "Kairos": (23_000_965, "exact", "HF safetensors element count"),
    "sundial_base": (128_329_680, "exact", "HF safetensors element count"),
    "Fracast-0": (85_001, "exact", "Release safetensors element count"),
    "litespecformer": (36_670_304, "exact", "HF safetensors element count"),
    "TimesFM-1.0": (200_000_000, "rounded", "TIME paper Table 1 reports 200M"),
    "visiontspp_base": (460_000_000, "rounded", "TIME paper Table 1 reports 460M"),
    "Moirai_base": (91_357_728, "exact", "HF safetensors element count"),
    "seasonal_naive": (0, "exact", "Parameter-free statistical baseline"),
}


SOURCES = {
    "fracast_release": "https://huggingface.co/ztxtech/fracast-0",
    "patchtst_fm_paper": "https://arxiv.org/abs/2602.06909",
    "patchtst_fm_r1": "https://huggingface.co/ibm-research/patchtst-fm-r1",
    "tabby": "https://huggingface.co/paris-noah/Tabby",
    "timemoe_50m": "https://huggingface.co/Maple728/TimeMoE-50M",
    "ts_icl": "https://huggingface.co/taharnbl/TS-ICL",
    "timer_s1": "https://huggingface.co/thuml/Timer-S1",
    "parameter_audit": "See parameter_evidence in TIME_pareto.csv",
}


def pareto_flags(rows: list[dict], metric: str, parameter_key: str) -> set[str]:
    flags: set[str] = set()
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other[parameter_key] <= row[parameter_key]
                and other[metric] < row[metric]
            ):
                dominated = True
                break
        if not dominated:
            flags.add(row["model"])
    return flags


def dominators(rows: list[dict], model: str, metric: str) -> list[dict]:
    target = next(row for row in rows if row["model"] == model)
    return [
        row
        for row in rows
        if row["model"] != model
        and row["parameter_count"] <= target["parameter_count"]
        and row[metric] < target[metric]
    ]


def main() -> None:
    with LEADERBOARD.open(newline="", encoding="utf-8") as handle:
        leaderboard = list(csv.DictReader(handle))

    rows: list[dict] = []
    for raw in leaderboard:
        model = raw["model"]
        if model not in PARAMETERS:
            raise KeyError(f"missing parameter evidence for {model}")
        parameters, parameter_kind, parameter_evidence = PARAMETERS[model]
        rows.append(
            {
                "leaderboard_rank": int(raw["leaderboard_rank"]),
                "model": model,
                "MASE_norm": float(raw["MASE_norm"]),
                "CRPS_norm": float(raw["CRPS_norm"]),
                "parameter_count": parameters,
                "parameter_kind": parameter_kind,
                "parameter_evidence": parameter_evidence,
            }
        )

    mase_front = pareto_flags(rows, "MASE_norm", "parameter_count")
    crps_front = pareto_flags(rows, "CRPS_norm", "parameter_count")
    for row in rows:
        row["pareto_parameter_mase"] = row["model"] in mase_front
        row["pareto_parameter_crps"] = row["model"] in crps_front

    target = next(row for row in rows if row["model"] == "Fracast-0")
    mase_dominators = dominators(rows, "Fracast-0", "MASE_norm")
    crps_dominators = dominators(rows, "Fracast-0", "CRPS_norm")
    better_mase = [
        row
        for row in rows
        if row["model"] != "Fracast-0" and row["MASE_norm"] < target["MASE_norm"]
    ]
    better_crps = [
        row
        for row in rows
        if row["model"] != "Fracast-0" and row["CRPS_norm"] < target["CRPS_norm"]
    ]
    smallest_better_mase = min(better_mase, key=lambda row: row["parameter_count"])
    smallest_better_crps = min(better_crps, key=lambda row: row["parameter_count"])

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    output_csv = ANALYSIS / "TIME_pareto.csv"
    fieldnames = [
        "leaderboard_rank",
        "model",
        "MASE_norm",
        "CRPS_norm",
        "parameter_count",
        "parameter_kind",
        "parameter_evidence",
        "pareto_parameter_mase",
        "pareto_parameter_crps",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fieldnames}
            for row in sorted(rows, key=lambda item: item["leaderboard_rank"])
        )

    summary = {
        "scope": {
            "models": len(rows),
            "dominance_definition": (
                "x dominates y if and only if x.parameter_count <= "
                "y.parameter_count and x.metric < y.metric; lower is better "
                "for both axes"
            ),
            "parameter_count_semantics": (
                "exact is a tensor/state_dict element count; lower_bound is "
                "a conservative public lower bound; rounded is a paper or "
                "model-card value"
            ),
        },
        "fracast": {
            "parameter_count": target["parameter_count"],
            "MASE_norm": target["MASE_norm"],
            "CRPS_norm": target["CRPS_norm"],
            "leaderboard_rank": target["leaderboard_rank"],
            "pareto_parameter_mase": target["pareto_parameter_mase"],
            "pareto_parameter_crps": target["pareto_parameter_crps"],
            "mase_dominators": [row["model"] for row in mase_dominators],
            "crps_dominators": [row["model"] for row in crps_dominators],
        },
        "tightest_better_performing_candidates": {
            "MASE": smallest_better_mase,
            "CRPS": smallest_better_crps,
        },
        "fronts": {
            "parameter_mase": sorted(
                mase_front,
                key=lambda model: next(
                    row["parameter_count"] for row in rows if row["model"] == model
                ),
            ),
            "parameter_crps": sorted(
                crps_front,
                key=lambda model: next(
                    row["parameter_count"] for row in rows if row["model"] == model
                ),
            ),
        },
        "sources": SOURCES,
        "limitations": [
            "The official raw submission is merged; no separate official "
            "recomputation is recorded for the local 98/98 compact scores.",
            "47 of 98 tasks use the 48-step release head with "
            "median-quantile rollout, not a strict official protocol claim.",
            "OmniScient uses the conservative Chronos-2 base lower bound and "
            "does not dominate Fracast-0.",
            "PatchTST-FM-Extended uses the public R1 parameter count from "
            "the same 20-layer, d=1024 architecture; no independent "
            "safetensors file was published.",
        ],
    }
    output_json = ANALYSIS / "TIME_pareto_summary.json"
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Models: {len(rows)}")
    print(f"Fracast-0 parameter-MASE front: {target['pareto_parameter_mase']}")
    print(f"Fracast-0 parameter-CRPS front: {target['pareto_parameter_crps']}")
    print(f"MASE dominators: {[row['model'] for row in mase_dominators]}")
    print(f"CRPS dominators: {[row['model'] for row in crps_dominators]}")
    print(f"Smallest better MASE candidate: {smallest_better_mase}")
    print(f"Smallest better CRPS candidate: {smallest_better_crps}")
    print(output_csv)
    print(output_json)


if __name__ == "__main__":
    main()
