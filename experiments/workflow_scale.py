"""Train and evaluate scale-specific UGQP policies for larger workflows.

The 12-node row reuses the three existing paper policies and evaluates them on
fixed 12-node tasks.  The 20- and 32-node rows train new policies with the same
60,000-step PPO budget and other main settings, then evaluate all seeds on a
common 400-task stream.  Candidate sites remain local + three edges + cloud;
additional-server scalability is addressed only by the separate computational
shape stress because the released action/profile schema is fixed to five sites.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CORE = ROOT.parent / "src"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from infer_ppo import evaluate_model
from train_ppo import train


PAPER_MODEL_ROOT = ROOT.parent.parent / "data" / "main_comparison"
SEEDS = [2026, 2027, 2028]
NODE_BUDGETS = [12, 20, 32]
TRAINING_STEPS = 60_000
EVALUATION_EPISODES = 400
EVALUATION_SEED = 20_000


def _sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest.

    :param path: File to hash.
    :returns: SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file.

    :param path: CSV input.
    :returns: Data rows.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries as a UTF-8 CSV file.

    :param path: CSV output.
    :param rows: Non-empty data rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, str]], key: str) -> float:
    """Return the arithmetic mean of a numeric CSV column."""
    return statistics.fmean(float(row[key]) for row in rows)


def _std(rows: list[dict[str, str]], key: str) -> float:
    """Return the population standard deviation of a numeric CSV column."""
    return statistics.pstdev(float(row[key]) for row in rows)


def _paper_model(seed: int) -> Path:
    """Return the baseline policy path for one seed."""
    return PAPER_MODEL_ROOT / f"seed_{seed}" / "ugqp_gnn" / "model" / "ppo_full.zip"


def _scaled_model(output_root: Path, nodes: int, seed: int) -> tuple[Path, float]:
    """Train or reuse one exact-config scale-specific model.

    :param output_root: Effectiveness evidence root.
    :param nodes: Fixed workflow size.
    :param seed: Training seed.
    :returns: Model path and new-run wall time in seconds.
    """
    model_root = output_root / f"node_{nodes}" / f"seed_{seed}" / "model"
    model_path = model_root / "ppo_full.zip"
    config_path = model_root / "train_config.json"
    if model_path.exists() and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {
            "timesteps": TRAINING_STEPS,
            "seed": seed,
            "min_nodes": nodes,
            "max_nodes": nodes,
            "deadline_scale": 1.55,
            "queue_scale": 1.5,
        }
        if all(config.get(key) == value for key, value in expected.items()):
            return model_path, 0.0
        raise RuntimeError(f"Existing scale model has a mismatched config: {config_path}")

    started = time.perf_counter()
    trained_path = train(
        output=model_root,
        timesteps=TRAINING_STEPS,
        seed=seed,
        variant="full",
        arrival_rate=0.7,
        network_scale=1.0,
        deadline_scale=1.55,
        encoder="gnn",
        queue_mode="bursty",
        queue_scale=1.5,
        ent_coef=0.05,
        profile_mode="tradeoff_v2",
        n_envs=4,
        min_nodes=nodes,
        max_nodes=nodes,
        task_graph_mode="standard",
        quality_propagation_strength=1.50,
    )
    return trained_path, time.perf_counter() - started


def main() -> None:
    """Run the larger-workflow effectiveness experiment and aggregate results."""
    output_root = ROOT.parent / "outputs" / "scale_evidence" / "effectiveness"
    output_root.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, object]] = []
    all_episode_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    for nodes in NODE_BUDGETS:
        for seed in SEEDS:
            if nodes == 12:
                model_path = _paper_model(seed)
                training_seconds = 0.0
                model_origin = "12-node policy trained on 5--12 nodes"
            else:
                model_path, training_seconds = _scaled_model(output_root, nodes, seed)
                model_origin = f"scale-specific policy trained on fixed {nodes}-node tasks"
            if not model_path.exists():
                raise FileNotFoundError(model_path)

            evaluation_root = output_root / f"node_{nodes}" / f"seed_{seed}" / "evaluation"
            evaluate_model(
                model_path=model_path,
                output=evaluation_root,
                episodes=EVALUATION_EPISODES,
                seed=EVALUATION_SEED,
                variant="full",
                policy_name=f"UGQP_GNN_n{nodes}_seed{seed}",
                arrival_rate=0.7,
                network_scale=1.0,
                deadline_scale=1.55,
                queue_mode="bursty",
                queue_scale=1.5,
                profile_mode="tradeoff_v2",
                min_nodes=nodes,
                max_nodes=nodes,
                permute_node_order=False,
                task_graph_mode="standard",
                quality_propagation_strength=1.50,
            )
            episodes = _read_csv(evaluation_root / "episode_records.csv")
            for episode in episodes:
                all_episode_rows.append({"node_budget": nodes, "training_seed": seed, **episode})
            possible_edges = nodes * (nodes - 1) / 2
            seed_rows.append(
                {
                    "node_budget": nodes,
                    "training_seed": seed,
                    "episodes": len(episodes),
                    "mean_edges": _mean(episodes, "edges"),
                    "mean_graph_density": _mean(episodes, "edges") / possible_edges,
                    "mean_total_latency": _mean(episodes, "total_latency"),
                    "mean_latency_per_node": _mean(episodes, "total_latency") / nodes,
                    "deadline_satisfaction_rate": 1.0 - _mean(episodes, "deadline_violation"),
                    "mean_avg_quality": _mean(episodes, "avg_quality"),
                    "success_rate": _mean(episodes, "success"),
                    "mean_total_cost": _mean(episodes, "total_cost"),
                    "mean_cost_per_node": _mean(episodes, "total_cost") / nodes,
                    "mean_high_risk_local_ratio": _mean(episodes, "high_risk_local_ratio"),
                    "mean_offload_ratio": _mean(episodes, "offload_ratio"),
                }
            )
            model_rows.append(
                {
                    "node_budget": nodes,
                    "training_seed": seed,
                    "model_origin": model_origin,
                    "model_path": model_path.name,
                    "model_sha256": _sha256(model_path),
                    "new_training_wall_seconds": training_seconds,
                }
            )
            print(
                f"nodes={nodes} seed={seed} quality={_mean(episodes, 'avg_quality'):.4f} "
                f"DSR={1.0 - _mean(episodes, 'deadline_violation'):.4f} "
                f"success={_mean(episodes, 'success'):.4f}"
            )

    aggregate_rows: list[dict[str, object]] = []
    for nodes in NODE_BUDGETS:
        subset = [row for row in all_episode_rows if int(row["node_budget"]) == nodes]
        possible_edges = nodes * (nodes - 1) / 2
        aggregate_rows.append(
            {
                "node_budget": nodes,
                "training_seeds": len(SEEDS),
                "episodes": len(subset),
                "mean_edges": _mean(subset, "edges"),
                "mean_graph_density": _mean(subset, "edges") / possible_edges,
                "mean_total_latency": _mean(subset, "total_latency"),
                "std_total_latency": _std(subset, "total_latency"),
                "mean_latency_per_node": _mean(subset, "total_latency") / nodes,
                "deadline_satisfaction_rate": 1.0 - _mean(subset, "deadline_violation"),
                "mean_avg_quality": _mean(subset, "avg_quality"),
                "std_avg_quality": _std(subset, "avg_quality"),
                "success_rate": _mean(subset, "success"),
                "mean_total_cost": _mean(subset, "total_cost"),
                "mean_cost_per_node": _mean(subset, "total_cost") / nodes,
                "mean_high_risk_local_ratio": _mean(subset, "high_risk_local_ratio"),
                "mean_offload_ratio": _mean(subset, "offload_ratio"),
            }
        )

    _write_csv(output_root / "large_workflow_seed_summary.csv", seed_rows)
    _write_csv(output_root / "large_workflow_aggregate.csv", aggregate_rows)
    _write_csv(output_root / "large_workflow_episode_records.csv", all_episode_rows)
    _write_csv(output_root / "large_workflow_models.csv", model_rows)
    manifest = {
        "purpose": "Scale-specific UGQP effectiveness on larger workflows",
        "node_budgets": NODE_BUDGETS,
        "training_seeds": SEEDS,
        "training_steps_per_new_model": TRAINING_STEPS,
        "evaluation_episodes_per_seed": EVALUATION_EPISODES,
        "evaluation_seed": EVALUATION_SEED,
        "new_models": 6,
        "reused_main_models": 3,
        "main_settings": {
            "arrival_rate": 0.7,
            "network_scale": 1.0,
            "deadline_scale": 1.55,
            "queue_mode": "bursty",
            "queue_scale": 1.5,
            "profile_mode": "tradeoff_v2",
            "edge_servers": 3,
            "action_count": 5,
            "quality_propagation_strength": 1.50,
        },
        "scope": (
            "The 20- and 32-node results are scale-specific retraining/evaluation results with three edge servers. "
            "They do not establish transfer of the original 12-node policy or effectiveness with additional servers."
        ),
    }
    (output_root / "large_workflow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
