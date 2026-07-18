from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments_20260716.v12f_joint_legacy_graph_queue.aggregate import (
    T975,
    composite_utility,
)


VARIANTS = ("full", "ppo_mlp", "no_uncertainty_risk_mask")
BASE_METRICS = (
    "avg_quality",
    "success",
    "reward",
    "total_latency",
    "deadline_violation",
    "total_cost",
    "mean_chosen_queue_wait",
    "high_risk_local_ratio",
)
ALL_METRICS = (
    "avg_quality",
    "success",
    "reward",
    "composite_utility",
    "total_latency",
    "p95_total_latency",
    "deadline_violation",
    "total_cost",
    "mean_chosen_queue_wait",
    "high_risk_local_ratio",
)


def _macro(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    strata = []
    for frame in frames.values():
        row = {metric: float(frame[metric].mean()) for metric in BASE_METRICS}
        row["p95_total_latency"] = float(frame["total_latency"].quantile(0.95))
        strata.append(row)
    macro = {
        metric: float(np.mean([row[metric] for row in strata]))
        for metric in (*BASE_METRICS, "p95_total_latency")
    }
    macro["composite_utility"] = composite_utility(macro)
    return macro


def _interval(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    half = T975.get(n, 1.96) * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def aggregate_root(root: Path) -> dict[str, Any]:
    protocol = json.loads((root / "experiment_protocol.json").read_text(encoding="utf-8"))
    reference = Path(protocol["full_reference_root"])
    rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    for seed in protocol["seeds"]:
        by_variant: dict[str, dict[str, pd.DataFrame]] = {v: {} for v in VARIANTS}
        for stratum in protocol["evaluation_strata"]:
            full_path = reference / f"seed_{seed}" / "eval" / stratum / "full" / "episode_records.csv"
            full = pd.read_csv(full_path)
            by_variant["full"][stratum] = full
            for variant in VARIANTS[1:]:
                path = root / f"seed_{seed}" / "eval" / stratum / variant / "episode_records.csv"
                frame = pd.read_csv(path)
                by_variant[variant][stratum] = frame
                keys = ["episode", "scenario", "nodes", "edges", "deadline", "graph_pair_id", "graph_pair_member"]
                exact = all(
                    full[key].astype(str).reset_index(drop=True).equals(
                        frame[key].astype(str).reset_index(drop=True)
                    )
                    for key in keys
                )
                pairing_rows.append(
                    {"seed": seed, "stratum": stratum, "variant": variant, "episodes": len(frame), "task_pairing_exact": int(exact)}
                )
        for variant, frames in by_variant.items():
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "strata": len(frames),
                    "episodes": int(sum(len(frame) for frame in frames.values())),
                    **_macro(frames),
                }
            )
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(analysis / "complete_ablation_per_seed.csv", index=False, encoding="utf-8-sig")
    pairing = pd.DataFrame(pairing_rows)
    pairing.to_csv(analysis / "task_pairing_audit.csv", index=False, encoding="utf-8-sig")

    aggregate_rows = []
    for variant in VARIANTS:
        subset = per_seed[per_seed["variant"] == variant]
        row: dict[str, Any] = {"variant": variant, "training_seeds": len(subset)}
        for metric in ALL_METRICS:
            for key, value in _interval(subset[metric].to_numpy(float)).items():
                row[f"{key}_{metric}"] = value
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(analysis / "complete_ablation_aggregate.csv", index=False, encoding="utf-8-sig")

    delta_rows = []
    full = per_seed[per_seed["variant"] == "full"].set_index("seed")
    for variant in VARIANTS[1:]:
        other = per_seed[per_seed["variant"] == variant].set_index("seed")
        for seed in protocol["seeds"]:
            row = {"seed": seed, "comparison": f"full_minus_{variant}"}
            for metric in ALL_METRICS:
                row[metric] = float(full.loc[seed, metric] - other.loc[seed, metric])
            delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(analysis / "complete_ablation_delta_per_seed.csv", index=False, encoding="utf-8-sig")
    delta_agg = []
    for comparison, subset in deltas.groupby("comparison", sort=True):
        row = {"comparison": comparison, "training_seeds": len(subset)}
        for metric in ALL_METRICS:
            for key, value in _interval(subset[metric].to_numpy(float)).items():
                row[f"{key}_{metric}"] = value
            row[f"positive_seed_count_{metric}"] = int((subset[metric] > 0).sum())
        delta_agg.append(row)
    pd.DataFrame(delta_agg).to_csv(
        analysis / "complete_ablation_delta_aggregate.csv", index=False, encoding="utf-8-sig"
    )
    mask_frames = []
    for seed in protocol["seeds"]:
        for stratum in protocol["evaluation_strata"]:
            path = root / f"seed_{seed}" / "eval" / stratum / "no_uncertainty_risk_mask" / "episode_records.csv"
            mask_frames.append(pd.read_csv(path))
    mask = pd.concat(mask_frames, ignore_index=True)
    mask_audit = {
        "episodes": int(len(mask)),
        "mean_changed_nodes": float(mask["risk_mask_changed_node_count"].mean()),
        "mean_changed_ratio": float(mask["risk_mask_changed_node_ratio"].mean()),
        "any_changed_episode_rate": float((mask["risk_mask_changed_node_count"] > 0).mean()),
    }
    result = {
        "root": str(root.resolve()),
        "full_reference_root": str(reference.resolve()),
        "seeds": protocol["seeds"],
        "pairing_exact": bool((pairing["task_pairing_exact"] == 1).all()),
        "risk_mask_audit": mask_audit,
    }
    (analysis / "complete_ablation_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result

