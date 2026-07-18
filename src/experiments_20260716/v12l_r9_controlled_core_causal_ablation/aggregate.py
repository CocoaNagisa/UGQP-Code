from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments_20260716.v12g_primary_stable_graph_queue.aggregate import (
    HISTORICAL_EFFECTS,
    MODULE_THRESHOLDS,
    PRACTICAL_THRESHOLDS,
    UTILITY_BOUNDS,
    UTILITY_WEIGHTS,
    aggregate_root as aggregate_v12g_schema,
)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _hierarchical_audit(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    protocol = json.loads(
        (root / "experiment_protocol.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    for seed in protocol["seeds"]:
        for stratum in protocol["evaluation_strata"]:
            base = root / f"seed_{seed}" / "eval" / stratum
            full = _read(base / "full" / "action_records.csv")
            blind = _read(base / "no_queue" / "action_records.csv")
            full_episodes = _read(base / "full" / "episode_records.csv")
            blind_episodes = _read(base / "no_queue" / "episode_records.csv")
            joined = full.merge(
                blind,
                on=["episode", "step"],
                suffixes=("_full", "_no_queue"),
                validate="one_to_one",
            )
            tier_match = joined["tier_action_full"].astype(str) == joined[
                "tier_action_no_queue"
            ].astype(str)
            quality_difference = np.abs(
                joined["quality_full"].astype(float)
                - joined["quality_no_queue"].astype(float)
            )
            cost_difference = np.abs(
                joined["cost_full"].astype(float)
                - joined["cost_no_queue"].astype(float)
            )
            latency_advantage = (
                joined["latency_no_queue"].astype(float)
                - joined["latency_full"].astype(float)
            )
            twin = joined[
                joined["router_pair_no_queue"].astype(str).str.contains(
                    "|", regex=False
                )
            ]
            episode_joined = full_episodes.merge(
                blind_episodes,
                on="episode",
                suffixes=("_full", "_no_queue"),
                validate="one_to_one",
            )
            episode_sequence_match = (
                joined.groupby("episode", sort=True)
                .apply(
                    lambda group: bool(
                        (
                            group["tier_action_full"].astype(str).to_numpy()
                            == group["tier_action_no_queue"].astype(str).to_numpy()
                        ).all()
                    ),
                    include_groups=False,
                )
                .astype(bool)
            )
            rows.append(
                {
                    "seed": int(seed),
                    "stratum": stratum,
                    "steps": len(joined),
                    "tier_action_match_rate": float(tier_match.mean()),
                    "episode_tier_sequence_match_rate": float(
                        episode_sequence_match.mean()
                    ),
                    "max_abs_quality_difference": float(quality_difference.max()),
                    "max_abs_cost_difference": float(cost_difference.max()),
                    "max_abs_episode_quality_difference": float(
                        np.abs(
                            episode_joined["avg_quality_full"].astype(float)
                            - episode_joined["avg_quality_no_queue"].astype(float)
                        ).max()
                    ),
                    "max_abs_episode_cost_difference": float(
                        np.abs(
                            episode_joined["total_cost_full"].astype(float)
                            - episode_joined["total_cost_no_queue"].astype(float)
                        ).max()
                    ),
                    "full_latency_le_no_queue_rate": float(
                        (latency_advantage >= -1e-10).mean()
                    ),
                    "mean_latency_advantage": float(latency_advantage.mean()),
                    "full_mean_router_regret": float(
                        joined["router_queue_regret_full"].astype(float).mean()
                    ),
                    "no_queue_mean_router_regret": float(
                        joined["router_queue_regret_no_queue"].astype(float).mean()
                    ),
                    "blind_twin_router_steps": int(len(twin)),
                    "blind_replica_b_rate": float(
                        twin["routed_action_no_queue"]
                        .astype(str)
                        .str.endswith("_b")
                        .mean()
                    )
                    if len(twin)
                    else float("nan"),
                }
            )
    frame = pd.DataFrame(rows)
    exact_control = bool(
        (frame["tier_action_match_rate"] == 1.0).all()
        and (frame["episode_tier_sequence_match_rate"] == 1.0).all()
        and (frame["max_abs_quality_difference"] <= 1e-12).all()
        and (frame["max_abs_cost_difference"] <= 1e-12).all()
        and (frame["max_abs_episode_quality_difference"] <= 1e-12).all()
        and (frame["max_abs_episode_cost_difference"] <= 1e-12).all()
        and (frame["full_latency_le_no_queue_rate"] == 1.0).all()
        and (frame["full_mean_router_regret"].abs() <= 1e-12).all()
    )
    summary = {
        "controller_intervention_isolated": exact_control,
        "tier_action_match_rate_min": float(
            frame["tier_action_match_rate"].min()
        ),
        "episode_tier_sequence_match_rate_min": float(
            frame["episode_tier_sequence_match_rate"].min()
        ),
        "max_abs_quality_difference": float(
            frame["max_abs_quality_difference"].max()
        ),
        "max_abs_cost_difference": float(
            frame["max_abs_cost_difference"].max()
        ),
        "max_abs_episode_quality_difference": float(
            frame["max_abs_episode_quality_difference"].max()
        ),
        "max_abs_episode_cost_difference": float(
            frame["max_abs_episode_cost_difference"].max()
        ),
        "full_latency_le_no_queue_rate_min": float(
            frame["full_latency_le_no_queue_rate"].min()
        ),
        "mean_latency_advantage": float(
            np.average(frame["mean_latency_advantage"], weights=frame["steps"])
        ),
        "full_mean_router_regret": float(
            np.average(frame["full_mean_router_regret"], weights=frame["steps"])
        ),
        "no_queue_mean_router_regret": float(
            np.average(
                frame["no_queue_mean_router_regret"], weights=frame["steps"]
            )
        ),
        "interpretation": (
            "Full and no_queue share one queue-free tier controller.  Their "
            "quality/cost/tier sequence must be identical; only the same-tier "
            "replica router may change latency and deadline outcomes."
        ),
    }
    return frame, summary


def _graph_training_audit(root: Path) -> dict[str, Any]:
    protocol = json.loads(
        (root / "experiment_protocol.json").read_text(encoding="utf-8")
    )
    quota_rows: list[dict[str, Any]] = []
    for seed in protocol["seeds"]:
        for stratum in protocol["evaluation_strata"]:
            for variant in ("full", "no_graph", "no_queue"):
                path = (
                    root
                    / f"seed_{seed}"
                    / "eval"
                    / stratum
                    / variant
                    / "episode_records.csv"
                )
                frame = _read(path)
                uses = frame["high_capacity_tier_uses"].astype(int)
                quota_rows.append(
                    {
                        "seed": int(seed),
                        "stratum": stratum,
                        "variant": variant,
                        "min_uses": int(uses.min()),
                        "max_uses": int(uses.max()),
                        "all_exactly_two": bool((uses == 2).all()),
                    }
                )

    checkpoint_rows: list[dict[str, Any]] = []
    for seed in protocol["seeds"]:
        full_cfg = json.loads(
            (root / f"seed_{seed}" / "models" / "full" / "train_config.json")
            .read_text(encoding="utf-8")
        )
        graph_cfg = json.loads(
            (
                root
                / f"seed_{seed}"
                / "models"
                / "no_graph"
                / "train_config.json"
            ).read_text(encoding="utf-8")
        )
        queue_cfg = json.loads(
            (
                root
                / f"seed_{seed}"
                / "models"
                / "no_queue"
                / "shared_controller.json"
            ).read_text(encoding="utf-8")
        )
        checkpoint_rows.append(
            {
                "seed": int(seed),
                "full_no_graph_distinct": (
                    full_cfg["model_sha256"] != graph_cfg["model_sha256"]
                ),
                "no_queue_reuses_full": (
                    queue_cfg["source_sha256"] == full_cfg["model_sha256"]
                ),
            }
        )
    return {
        "quota_rows": quota_rows,
        "checkpoint_rows": checkpoint_rows,
        "quota_exactly_two_all": bool(
            quota_rows and all(row["all_exactly_two"] for row in quota_rows)
        ),
        "full_no_graph_independent_all": bool(
            checkpoint_rows
            and all(row["full_no_graph_distinct"] for row in checkpoint_rows)
        ),
        "no_queue_reuses_full_all": bool(
            checkpoint_rows
            and all(row["no_queue_reuses_full"] for row in checkpoint_rows)
        ),
    }


def _r9_adoption_gate(root: Path) -> dict[str, Any]:
    deltas = _read(root / "analysis" / "table6_delta_per_seed.csv")
    graph = deltas[deltas["comparison"] == "full_minus_no_graph"]
    queue = deltas[deltas["comparison"] == "full_minus_no_queue"]
    graph_rule = {
        "avg_quality_positive_each_seed": bool((graph["avg_quality"] > 0).all()),
        "success_nonnegative_each_seed": bool((graph["success"] >= 0).all()),
        "reward_positive_each_seed": bool((graph["reward"] > 0).all()),
        "utility_positive_each_seed": bool(
            (graph["composite_utility"] > 0).all()
        ),
        "mean_avg_quality_gain_ge_0_005": bool(
            graph["avg_quality"].mean() >= 0.005
        ),
        "mean_success_gain_ge_0_03": bool(graph["success"].mean() >= 0.03),
    }
    queue_rule = {
        "quality_exact": bool((queue["avg_quality"].abs() <= 1e-12).all()),
        "cost_exact": bool((queue["total_cost"].abs() <= 1e-12).all()),
        "success_nonnegative_each_seed": bool((queue["success"] >= 0).all()),
        "mean_success_gain_ge_0_03": bool(queue["success"].mean() >= 0.03),
        "reward_positive_each_seed": bool((queue["reward"] > 0).all()),
        "utility_positive_each_seed": bool(
            (queue["composite_utility"] > 0).all()
        ),
        "latency_lower_each_seed": bool((queue["total_latency"] < 0).all()),
        "queue_wait_lower_each_seed": bool(
            (queue["mean_chosen_queue_wait"] < 0).all()
        ),
        "deadline_nonworse_each_seed": bool(
            (queue["deadline_violation"] <= 0).all()
        ),
    }
    return {
        "graph_rule": graph_rule,
        "queue_rule": queue_rule,
        "graph_pass": bool(all(graph_rule.values())),
        "queue_pass": bool(all(queue_rule.values())),
        "historical_effects_are_diagnostic_only": True,
        "queue_interpretation": (
            "quality/cost non-inferiority is exact by construction; the queue "
            "module is judged on success, Reward, utility and direct latency/queue outcomes"
        ),
    }


def aggregate_root(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    result = aggregate_v12g_schema(root)
    audit_frame, hierarchical = _hierarchical_audit(root)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(
        analysis / "v12h_hierarchical_queue_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    episode_files = list(root.glob("seed_*/eval/*/*/episode_records.csv"))
    graph_gate_rows = []
    for path in episode_files:
        frame = pd.read_csv(path)
        graph_gate_rows.append(
            {
                "path": str(path),
                "all_pass": bool(
                    (frame["graph_role_swap_gate_pass"].astype(int) == 1).all()
                ),
                "max_correlation": float(
                    frame["graph_exposure_correlation_ab"].astype(float).max()
                ),
                "min_mad": float(
                    frame["graph_exposure_mad_ab"].astype(float).min()
                ),
                "max_top3_overlap": float(
                    frame["graph_top3_overlap_fraction_ab"].astype(float).max()
                ),
            }
        )
    graph_gate = {
        "files": len(graph_gate_rows),
        "all_pass": bool(
            graph_gate_rows and all(row["all_pass"] for row in graph_gate_rows)
        ),
        "max_correlation": max(
            (row["max_correlation"] for row in graph_gate_rows), default=float("nan")
        ),
        "min_mad": min(
            (row["min_mad"] for row in graph_gate_rows), default=float("nan")
        ),
        "max_top3_overlap": max(
            (row["max_top3_overlap"] for row in graph_gate_rows),
            default=float("nan"),
        ),
    }
    result["v12h_hierarchical_queue_audit"] = hierarchical
    result["v12h_role_swap_graph_gate"] = graph_gate
    graph_training = _graph_training_audit(root)
    result["v12h_r9_graph_training_audit"] = graph_training
    result["v12h_r9_adoption_gate"] = _r9_adoption_gate(root)
    result["v12h_hard_invariants_pass"] = bool(
        hierarchical["controller_intervention_isolated"]
        and graph_gate["all_pass"]
        and graph_training["quota_exactly_two_all"]
        and graph_training["full_no_graph_independent_all"]
        and graph_training["no_queue_reuses_full_all"]
    )
    (analysis / "v12h_gate_decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
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
