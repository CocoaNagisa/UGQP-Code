from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VARIANTS = ("full", "no_graph", "no_queue")
PRIMARY_METRICS = ("avg_quality", "success", "reward", "composite_utility")
SECONDARY_METRICS = (
    "total_latency",
    "p95_total_latency",
    "deadline_violation",
    "total_cost",
    "mean_chosen_queue_wait",
    "high_risk_local_ratio",
    "offload_ratio",
    "mean_same_tier_queue_regret",
    "same_tier_low_queue_selection_rate",
)
ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS

                                                                                 
                                                                   
UTILITY_BOUNDS = {
    "success": (0.0, 0.5191666666666667, True),
    "avg_quality": (0.4800685594358354, 0.7255083633747441, True),
    "total_latency": (19.27191572602168, 23.92283139631117, False),
    "deadline_violation": (0.0125, 0.1175, False),
    "total_cost": (2.5761054000000003, 4.185566765, False),
    "high_risk_local_ratio": (
        0.0158383838383838,
        0.048525523088023,
        False,
    ),
}
UTILITY_WEIGHTS = {
    "success": 0.30,
    "avg_quality": 0.30,
    "total_latency": 0.15,
    "deadline_violation": 0.10,
    "total_cost": 0.10,
    "high_risk_local_ratio": 0.05,
}
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}

HISTORICAL_EFFECTS = {
    "full_minus_no_graph": {
        "avg_quality": 0.00780,
        "success": 0.04417,
        "reward": 0.02500,
        "composite_utility": 2.04,
        "tolerance": {
            "avg_quality": 0.005,
            "success": 0.020,
            "reward": 0.050,
            "composite_utility": 1.00,
        },
        "source": "V12A random-label graph scheme 1",
    },
    "full_minus_no_queue": {
        "avg_quality": 0.01105,
        "success": 0.06667,
        "reward": 0.10882,
        "composite_utility": 5.29,
        "tolerance": {
            "avg_quality": 0.005,
            "success": 0.020,
            "reward": 0.050,
            "composite_utility": 2.00,
        },
        "source": "V12E-R2 queue scheme 3 R2 (single-seed reference)",
    },
}
PRACTICAL_THRESHOLDS = {
    "avg_quality": 0.005,
    "success": 0.030,
    "reward": 0.020,
    "composite_utility": 1.00,
}


def _normalized(value: float, low: float, high: float, higher: bool) -> float:
    if high <= low:
        raise ValueError("invalid frozen utility bound")
    raw = (value - low) / (high - low)
    if not higher:
        raw = (high - value) / (high - low)
    return float(np.clip(raw, 0.0, 1.0))


def composite_utility(row: dict[str, float] | pd.Series) -> float:
    total = 0.0
    for metric, weight in UTILITY_WEIGHTS.items():
        low, high, higher = UTILITY_BOUNDS[metric]
        total += weight * _normalized(float(row[metric]), low, high, higher)
    return 100.0 * total


def _interval(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    critical = T975.get(n, 1.96)
    half = critical * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _fingerprint(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "episode",
        "scenario",
        "nodes",
        "edges",
        "max_indegree",
        "graph_depth",
        "deadline",
        "task_region",
        "load_pattern",
        "load_intensity",
        "queue_trace_seed",
        "static_profile_signature",
    ]
    return frame[columns].sort_values("episode").reset_index(drop=True)


def _variant_seed_row(
    seed: int,
    variant: str,
    stratum_frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    per_stratum: list[dict[str, float]] = []
    for stratum, frame in sorted(stratum_frames.items()):
        row = {
            "stratum": stratum,
            "avg_quality": float(frame["avg_quality"].mean()),
            "success": float(frame["success"].mean()),
            "reward": float(frame["reward"].mean()),
            "total_latency": float(frame["total_latency"].mean()),
            "p95_total_latency": float(frame["total_latency"].quantile(0.95)),
            "deadline_violation": float(frame["deadline_violation"].mean()),
            "total_cost": float(frame["total_cost"].mean()),
            "mean_chosen_queue_wait": float(
                frame["mean_chosen_queue_wait"].mean()
            ),
            "high_risk_local_ratio": float(
                frame["high_risk_local_ratio"].mean()
            ),
            "offload_ratio": float(frame["offload_ratio"].mean()),
            "mean_same_tier_queue_regret": float(
                frame["mean_same_tier_queue_regret"].mean()
            ),
            "same_tier_low_queue_selection_rate": float(
                frame["same_tier_low_queue_selection_rate"].mean()
            ),
        }
        per_stratum.append(row)
    macro = {
        metric: float(np.mean([row[metric] for row in per_stratum]))
        for metric in (*PRIMARY_METRICS[:3], *SECONDARY_METRICS)
    }
    macro["composite_utility"] = composite_utility(macro)
    return {
        "seed": seed,
        "variant": variant,
        "strata": len(per_stratum),
        "episodes": int(sum(len(frame) for frame in stratum_frames.values())),
        **macro,
    }


def _classify(comparison: str, row: dict[str, Any]) -> dict[str, Any]:
    history = HISTORICAL_EFFECTS[comparison]
    floors = {
        metric: max(
            0.0,
            float(history[metric]) - float(history["tolerance"][metric]),
        )
        for metric in PRIMARY_METRICS
    }
    practical_pass = {
        metric: float(row[f"mean_{metric}"]) >= threshold
        for metric, threshold in PRACTICAL_THRESHOLDS.items()
    }
    reference_floor_pass = {
        metric: float(row[f"mean_{metric}"]) >= floors[metric]
        for metric in PRIMARY_METRICS
    }
    positive_seeds = {
        metric: int(row[f"positive_seeds_{metric}"])
        for metric in PRIMARY_METRICS
    }
    latency_rel = float(row["mean_total_latency"]) / max(
        float(row["mean_baseline_total_latency"]), 1e-12
    )
    cost_rel = float(row["mean_total_cost"]) / max(
        float(row["mean_baseline_total_cost"]), 1e-12
    )
    queue_rel = float(row["mean_mean_chosen_queue_wait"]) / max(
        float(row["mean_baseline_mean_chosen_queue_wait"]), 1e-12
    )
    secondary_tradeoff = bool(
        latency_rel > 0.04
        or float(row["mean_deadline_violation"]) > 0.02
        or cost_rel > 0.10
        or queue_rel > 0.10
        or float(row["mean_high_risk_local_ratio"]) > 0.02
    )
                                                                       
                                                                           
                                                                               
    catastrophic_secondary_harm = bool(
        latency_rel > 0.20
        or float(row["mean_deadline_violation"]) > 0.05
        or cost_rel > 0.25
        or queue_rel > 0.20
        or float(row["mean_high_risk_local_ratio"]) > 0.05
    )
    below_floor = sum(not passed for passed in reference_floor_pass.values())
    negative_upper = sum(
        float(row[f"ci95_high_{metric}"]) < 0.0 for metric in PRIMARY_METRICS
    )
    clearly_worse = bool(
        below_floor >= 2
        or negative_upper >= 1
        or catastrophic_secondary_harm
    )
    similar = bool(
        all(reference_floor_pass.values())
        and all(float(row[f"mean_{metric}"]) >= 0.0 for metric in PRIMARY_METRICS)
        and not catastrophic_secondary_harm
    )
    required_positive_seeds = max(1, math.ceil(int(row["training_seeds"]) / 2))
    positive = bool(
        all(practical_pass.values())
        and all(
            count >= required_positive_seeds for count in positive_seeds.values()
        )
        and not catastrophic_secondary_harm
    )
    if clearly_worse:
        label = "clearly_worse_than_original_reference"
    elif positive and similar and secondary_tradeoff:
        label = "positive_and_comparable_with_secondary_tradeoff"
    elif positive and similar:
        label = "positive_and_comparable_to_original_reference"
    elif positive:
        label = "positive_but_not_reference_equivalent"
    elif similar:
        label = "comparable_trend_but_primary_gate_incomplete"
    else:
        label = "mixed_or_uncertain"
    return {
        "label": label,
        "practical_pass": practical_pass,
        "reference_floor_pass": reference_floor_pass,
        "reference_floors": floors,
        "positive_seed_counts": positive_seeds,
        "secondary_tradeoff": secondary_tradeoff,
        "catastrophic_secondary_harm": catastrophic_secondary_harm,
        "clearly_worse": clearly_worse,
        "similar_to_original_reference": similar,
        "positive_primary_gate": positive,
    }


def aggregate_root(root: Path) -> dict[str, Any]:
    protocol_path = root / "experiment_protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol["seeds"]]
    strata = list(protocol["evaluation_strata"])

    seed_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for seed in seeds:
        frames: dict[str, dict[str, pd.DataFrame]] = {
            variant: {} for variant in VARIANTS
        }
        for stratum in strata:
            for variant in VARIANTS:
                path = (
                    root
                    / f"seed_{seed}"
                    / "eval"
                    / stratum
                    / variant
                    / "episode_records.csv"
                )
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                if frame.empty:
                    raise ValueError(f"empty evaluation file: {path}")
                frames[variant][stratum] = frame
            reference = _fingerprint(frames["full"][stratum])
            for variant in ("no_graph", "no_queue"):
                if not reference.equals(_fingerprint(frames[variant][stratum])):
                    raise ValueError(
                        f"evaluation pairing mismatch seed={seed} stratum={stratum} "
                        f"variant={variant}"
                    )

        by_variant = {
            variant: _variant_seed_row(seed, variant, frames[variant])
            for variant in VARIANTS
        }
        seed_rows.extend(by_variant.values())
        for ablated in ("no_graph", "no_queue"):
            comparison = f"full_minus_{ablated}"
            delta = {
                "seed": seed,
                "comparison": comparison,
                "episodes": by_variant["full"]["episodes"],
            }
            for metric in ALL_METRICS:
                delta[metric] = (
                    float(by_variant["full"][metric])
                    - float(by_variant[ablated][metric])
                )
                delta[f"baseline_{metric}"] = float(by_variant[ablated][metric])
            delta_rows.append(delta)

    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(seed_rows)
    deltas = pd.DataFrame(delta_rows)
    per_seed.to_csv(
        analysis / "table6_per_seed.csv", index=False, encoding="utf-8-sig"
    )
    deltas.to_csv(
        analysis / "table6_delta_per_seed.csv", index=False, encoding="utf-8-sig"
    )

    aggregate_rows: list[dict[str, Any]] = []
    for variant, subset in per_seed.groupby("variant", sort=False):
        row: dict[str, Any] = {
            "variant": variant,
            "training_seeds": len(subset),
        }
        for metric in ALL_METRICS:
            interval = _interval(subset[metric].astype(float).to_numpy())
            row.update({f"{name}_{metric}": value for name, value in interval.items()})
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        analysis / "table6_aggregate.csv", index=False, encoding="utf-8-sig"
    )

    delta_aggregate_rows: list[dict[str, Any]] = []
    decisions: dict[str, Any] = {}
    for comparison, subset in deltas.groupby("comparison", sort=False):
        row = {"comparison": comparison, "training_seeds": len(subset)}
        for metric in ALL_METRICS:
            interval = _interval(subset[metric].astype(float).to_numpy())
            row.update({f"{name}_{metric}": value for name, value in interval.items()})
            row[f"positive_seeds_{metric}"] = int((subset[metric] > 0.0).sum())
            row[f"mean_baseline_{metric}"] = float(
                subset[f"baseline_{metric}"].mean()
            )
        decision = _classify(comparison, row)
        row["decision"] = decision["label"]
        decisions[comparison] = decision
        delta_aggregate_rows.append(row)
    delta_aggregate = pd.DataFrame(delta_aggregate_rows)
    delta_aggregate.to_csv(
        analysis / "table6_delta_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    table_columns = [
        "variant",
        *[
            item
            for metric in (
                "avg_quality",
                "success",
                "reward",
                "composite_utility",
                "total_latency",
                "p95_total_latency",
                "deadline_violation",
                "total_cost",
                "mean_chosen_queue_wait",
            )
            for item in (f"mean_{metric}", f"sd_{metric}")
        ],
    ]
    aggregate[table_columns].to_csv(
        analysis / "manuscript_table6.csv", index=False, encoding="utf-8-sig"
    )

    result = {
        "root": str(root.resolve()),
        "seeds": seeds,
        "evaluation_strata": strata,
        "utility_weights": UTILITY_WEIGHTS,
        "utility_bounds": UTILITY_BOUNDS,
        "utility_clip": True,
        "practical_thresholds": PRACTICAL_THRESHOLDS,
        "historical_effects": HISTORICAL_EFFECTS,
        "decisions": decisions,
        "requires_deeper_followup": any(
            decision["clearly_worse"] for decision in decisions.values()
        ),
    }
    (analysis / "gate_decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
