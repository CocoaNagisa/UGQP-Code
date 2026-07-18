from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments_20260716.v12f_joint_legacy_graph_queue.aggregate import (
    HISTORICAL_EFFECTS,
    PRACTICAL_THRESHOLDS,
    PRIMARY_METRICS,
    UTILITY_BOUNDS,
    UTILITY_WEIGHTS,
    aggregate_root as aggregate_v12f_schema,
)


MODULE_THRESHOLDS = {
    "full_minus_no_graph": {
        "avg_quality": 0.005,
        "success": 0.030,
        "reward": 0.020,
        "composite_utility": 1.00,
    },
    "full_minus_no_queue": {
                                                                             
                                                                            
                                                                        
        "avg_quality": -0.002,
        "success": 0.030,
        "reward": 0.020,
        "composite_utility": 1.00,
    },
}


def _all_seed_rule(
    subset: pd.DataFrame, comparison: str
) -> dict[str, bool]:
    if comparison == "full_minus_no_graph":
        return {
            metric: bool((subset[metric].astype(float) > 0.0).all())
            for metric in PRIMARY_METRICS
        }
    return {
        "avg_quality": bool(
            (subset["avg_quality"].astype(float) >= -0.002).all()
        ),
        **{
            metric: bool((subset[metric].astype(float) > 0.0).all())
            for metric in PRIMARY_METRICS
            if metric != "avg_quality"
        },
    }


def _direct_queue_rule(subset: pd.DataFrame) -> dict[str, bool]:
    return {
        "mean_queue_wait_lower": bool(
            (subset["mean_chosen_queue_wait"].astype(float) < 0.0).all()
        ),
        "same_tier_regret_lower": bool(
            (subset["mean_same_tier_queue_regret"].astype(float) < 0.0).all()
        ),
        "short_queue_selection_mean_gain_ge_5pp": bool(
            subset["same_tier_low_queue_selection_rate"].astype(float).mean()
            >= 0.05
        ),
        "short_queue_selection_higher_each_seed": bool(
            (
                subset["same_tier_low_queue_selection_rate"].astype(float)
                > 0.0
            ).all()
        ),
        "latency_lower": bool(
            (subset["total_latency"].astype(float) < 0.0).all()
        ),
    }


def aggregate_root(root: Path) -> dict[str, Any]:
    result = aggregate_v12f_schema(root)
    delta_path = root / "analysis" / "table6_delta_aggregate.csv"
    per_seed_path = root / "analysis" / "table6_delta_per_seed.csv"
    deltas = pd.read_csv(delta_path)
    per_seed = pd.read_csv(per_seed_path)
    strict: dict[str, Any] = {}
    for _, row in deltas.iterrows():
        comparison = str(row["comparison"])
        seed_count = int(row["training_seeds"])
        subset = per_seed[per_seed["comparison"] == comparison].copy()
        all_seed_primary = _all_seed_rule(subset, comparison)
        thresholds = MODULE_THRESHOLDS[comparison]
        practical_pass = {
            metric: float(row[f"mean_{metric}"]) >= threshold
            for metric, threshold in thresholds.items()
        }
        history = HISTORICAL_EFFECTS[comparison]
        reference_floor_pass = {
            metric: float(row[f"mean_{metric}"])
            >= max(
                0.0,
                float(history[metric])
                - float(history["tolerance"][metric]),
            )
            for metric in PRIMARY_METRICS
        }
        baseline_latency = max(
            float(row["mean_baseline_total_latency"]), 1e-12
        )
        baseline_cost = max(float(row["mean_baseline_total_cost"]), 1e-12)
        no_catastrophic_secondary_harm = bool(
            float(row["mean_total_latency"]) / baseline_latency <= 0.20
            and float(row["mean_deadline_violation"]) <= 0.10
            and float(row["mean_total_cost"]) / baseline_cost <= 0.25
            and float(row["mean_high_risk_local_ratio"]) <= 0.05
        )
        direct_queue = (
            _direct_queue_rule(subset)
            if comparison == "full_minus_no_queue"
            else {}
        )
        pass_all = bool(
            seed_count >= 3
            and all(all_seed_primary.values())
            and all(practical_pass.values())
            and no_catastrophic_secondary_harm
            and all(direct_queue.values())
        )
        strict[comparison] = {
            "strict_label": (
                "strict_3_of_3_primary_gate_pass"
                if pass_all
                else "strict_3_of_3_primary_gate_fail"
            ),
            "training_seeds": seed_count,
            "all_seed_primary_rule": all_seed_primary,
            "practical_pass": practical_pass,
            "module_thresholds": thresholds,
            "reference_floor_pass": reference_floor_pass,
            "reference_floor_is_diagnostic_not_gate": True,
            "no_catastrophic_secondary_harm": (
                no_catastrophic_secondary_harm
            ),
            "direct_queue_rule": direct_queue,
            "pass": pass_all,
        }
    result["strict_v12g_gate"] = strict
    result["requires_deeper_followup"] = any(
        not decision["pass"] for decision in strict.values()
    )
    aggregate = pd.read_csv(root / "analysis" / "table6_aggregate.csv")
    utility_resolution: dict[str, Any] = {}
    for _, variant_row in aggregate.iterrows():
        saturated: list[str] = []
        for metric, (low, high, _) in UTILITY_BOUNDS.items():
            value = float(variant_row[f"mean_{metric}"])
            if value <= float(low) or value >= float(high):
                saturated.append(metric)
        utility_resolution[str(variant_row["variant"])] = {
            "saturated_components": saturated,
            "saturated_weight": float(
                sum(UTILITY_WEIGHTS[name] for name in saturated)
            ),
            "full_resolution": len(saturated) < 2,
        }
    result["utility_resolution_audit"] = utility_resolution
    result["utility_resolution_note"] = (
        "Eq.31 retains the manuscript's frozen main-comparison bounds.  When "
        "two or more components saturate, composite utility remains reported "
        "but cannot replace the separate primary and secondary metrics."
    )
    output = root / "analysis" / "strict_gate_decision.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


__all__ = [
    "HISTORICAL_EFFECTS",
    "MODULE_THRESHOLDS",
    "PRACTICAL_THRESHOLDS",
    "UTILITY_BOUNDS",
    "UTILITY_WEIGHTS",
    "aggregate_root",
]
