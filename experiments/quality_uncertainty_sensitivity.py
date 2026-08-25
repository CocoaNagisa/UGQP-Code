"""Evaluate frozen policies under bounded quality and uncertainty perturbations."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO


ROOT = Path(__file__).resolve().parent
CORE = ROOT.parent / "src"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import llm_offload_env.env as env_module
from llm_offload_env.config import make_config
from llm_offload_env.env import LLMAgentOffloadEnv
from llm_offload_env.gnn_policy import (
    DAGMessagePassingExtractor,
    DTODRLGATExtractor,
)
from llm_offload_env.taskgen import compute_graph_features


PAPER_MODEL_ROOT = ROOT.parent.parent / "data" / "main_comparison"
SEEDS = [2026, 2027, 2028]
SCALES = [0.80, 1.00, 1.20]
EPISODES = 400
EVALUATION_SEED = 20_000


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries as a UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    """Return the arithmetic mean of a numeric field."""
    return statistics.fmean(float(row[key]) for row in rows)


def _paper_model(seed: int) -> Path:
    """Return the baseline policy for one seed."""
    return PAPER_MODEL_ROOT / f"seed_{seed}" / "ugqp_gnn" / "model" / "ppo_full.zip"


def _quality_with_scale(node, profile, scale: float) -> float:
    """Recompute quality while scaling non-uncertainty manual corrections."""
    complexity_penalty = 0.11 if node.kind in {"plan", "generate"} else 0.07 if node.kind == "verify" else 0.035
    uncertainty_penalty = node.uncertainty * (
        0.40 if profile.name == "local" else 0.17 if profile.name.startswith("edge") else 0.10
    )
    tool_penalty = 0.04 if node.kind == "tool_call" and profile.name == "local" else 0.0
    privacy_bonus = 0.025 if node.privacy == 3 and profile.name == "local" else 0.0
    value = (
        profile.quality_base
        - scale * complexity_penalty
        - uncertainty_penalty
        - scale * tool_penalty
        + scale * privacy_bonus
    )
    return min(1.0, max(0.0, value))


def _make_env(seed: int) -> LLMAgentOffloadEnv:
    """Create the main-protocol evaluation environment."""
    return LLMAgentOffloadEnv(
        make_config(
            seed=seed,
            variant="full",
            arrival_rate=0.7,
            network_scale=1.0,
            deadline_scale=1.55,
            queue_mode="bursty",
            queue_scale=1.5,
            profile_mode="tradeoff_v2",
            min_nodes=5,
            max_nodes=12,
            task_graph_mode="standard",
            quality_propagation_strength=1.50,
        )
    )


def _scale_fused_uncertainty(env: LLMAgentOffloadEnv, scale: float) -> None:
    """Scale the already-fused uncertainty state and its derived risk fields."""
    assert env.task is not None
    for node in env.task.nodes:
        node.uncertainty = min(1.0, max(0.0, node.uncertainty * scale))
        node.risk = min(1.0, max(0.0, node.risk * scale))
        node.token_uncertainty = min(1.0, max(0.0, node.token_uncertainty * scale))
    compute_graph_features(env.task.nodes)


def _evaluate_condition(family: str, scale: float, seed: int) -> list[dict[str, object]]:
    """Evaluate one family/scale/model condition on shared tasks."""
    model = MaskablePPO.load(_paper_model(seed), device="cpu")
    rows: list[dict[str, object]] = []
    original_quality = env_module.quality
    try:
        if family == "quality_coefficients":
            env_module.quality = lambda node, profile: _quality_with_scale(node, profile, scale)
        else:
            env_module.quality = lambda node, profile: _quality_with_scale(node, profile, 1.0)

        for episode in range(EPISODES):
            env = _make_env(EVALUATION_SEED + episode)
            observation, _ = env.reset()
            if family == "fused_uncertainty":
                _scale_fused_uncertainty(env, scale)
                observation = env._obs()
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
                observation, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
            rows.append(
                {
                    "sensitivity_family": family,
                    "scale": scale,
                    "training_seed": seed,
                    **env.episode_summary(f"UGQP_{family}_{scale:.2f}_seed{seed}", episode),
                }
            )
    finally:
        env_module.quality = original_quality
    return rows


def main() -> None:
    """Run frozen-policy quality/uncertainty sensitivity and aggregate results."""
    output = ROOT.parent / "outputs" / "sensitivity_evidence" / "quality_uncertainty"
    output.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict[str, object]] = []
    for family in ("quality_coefficients", "fused_uncertainty"):
        for scale in SCALES:
            for seed in SEEDS:
                rows = _evaluate_condition(family, scale, seed)
                episode_rows.extend(rows)
                print(
                    f"family={family} scale={scale:.2f} seed={seed} "
                    f"quality={_mean(rows, 'avg_quality'):.4f} "
                    f"DSR={1.0 - _mean(rows, 'deadline_violation'):.4f}"
                )

    aggregate_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for family in ("quality_coefficients", "fused_uncertainty"):
        for scale in SCALES:
            for seed in SEEDS:
                subset = [
                    row
                    for row in episode_rows
                    if row["sensitivity_family"] == family
                    and float(row["scale"]) == scale
                    and int(row["training_seed"]) == seed
                ]
                seed_rows.append(
                    {
                        "sensitivity_family": family,
                        "scale": scale,
                        "training_seed": seed,
                        "episodes": len(subset),
                        "mean_total_latency": _mean(subset, "total_latency"),
                        "deadline_satisfaction_rate": 1.0 - _mean(subset, "deadline_violation"),
                        "mean_avg_quality": _mean(subset, "avg_quality"),
                        "success_rate": _mean(subset, "success"),
                        "mean_total_cost": _mean(subset, "total_cost"),
                        "mean_high_risk_local_ratio": _mean(subset, "high_risk_local_ratio"),
                    }
                )
            pooled = [
                row
                for row in episode_rows
                if row["sensitivity_family"] == family and float(row["scale"]) == scale
            ]
            aggregate_rows.append(
                {
                    "sensitivity_family": family,
                    "scale": scale,
                    "training_seeds": len(SEEDS),
                    "episodes": len(pooled),
                    "mean_total_latency": _mean(pooled, "total_latency"),
                    "deadline_satisfaction_rate": 1.0 - _mean(pooled, "deadline_violation"),
                    "mean_avg_quality": _mean(pooled, "avg_quality"),
                    "success_rate": _mean(pooled, "success"),
                    "mean_total_cost": _mean(pooled, "total_cost"),
                    "mean_high_risk_local_ratio": _mean(pooled, "high_risk_local_ratio"),
                }
            )

    _write_csv(output / "quality_uncertainty_episode_records.csv", episode_rows)
    _write_csv(output / "quality_uncertainty_seed_summary.csv", seed_rows)
    _write_csv(output / "quality_uncertainty_aggregate.csv", aggregate_rows)
    manifest = {
        "purpose": "Frozen-policy quality-coefficient and fused-uncertainty sensitivity",
        "families": {
            "quality_coefficients": (
                "Co-scale manually configured complexity, tool, and privacy quality corrections by ±20%; "
                "hold the uncertainty penalty coefficient fixed."
            ),
            "fused_uncertainty": (
                "Scale the already-fused node uncertainty scalar and its derived risk/token-uncertainty fields by ±20%."
            ),
        },
        "scales": SCALES,
        "training_seeds": SEEDS,
        "episodes_per_seed_condition": EPISODES,
        "evaluation_seed": EVALUATION_SEED,
        "current_uncertainty_implementation": (
            "The simulator directly samples an already-fused scalar u_v. It does not instantiate the five separate "
            "source signals or an active lambda vector in the reported runs."
        ),
        "scope": (
            "Frozen-policy evaluation sensitivity. These runs test decision/output robustness to bounded measurement-model "
            "perturbations; they are not retraining sensitivity and do not identify source-specific uncertainty weights."
        ),
    }
    (output / "quality_uncertainty_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
