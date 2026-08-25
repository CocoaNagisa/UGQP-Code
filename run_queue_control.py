"""Run the queue-observation control experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from infer_ppo import evaluate_model
from train_ppo import train


SEEDS = (2024, 2025, 2026, 2027, 2028)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "queue_observation_control")
    parser.add_argument("--timesteps", type=int, default=60000)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    checkpoint = ROOT / "models" / "dtodrl_gat_pretrained.pt"
    for seed in SEEDS:
        model_dir = args.output / "models" / f"seed_{seed}"
        model_path = model_dir / "ppo_no_queue.zip"
        if not (args.resume and model_path.exists()):
            model_path = train(
                model_dir,
                args.timesteps,
                seed=seed,
                variant="no_queue",
                arrival_rate=0.7,
                network_scale=1.0,
                deadline_scale=1.55,
                encoder="gnn",
                queue_mode="bursty",
                queue_scale=1.5,
                ent_coef=0.05,
                profile_mode="tradeoff_v2",
                n_envs=4,
                min_nodes=5,
                max_nodes=12,
                task_graph_mode="standard",
                quality_propagation_strength=1.5,
            )
        evaluation_dir = args.output / "evaluation" / f"seed_{seed}"
        if not (args.resume and (evaluation_dir / "summary.csv").exists()):
            evaluate_model(
                model_path,
                evaluation_dir,
                episodes=args.episodes,
                seed=20000,
                variant="no_queue",
                policy_name="UGQP-GNN no-queue observation",
                arrival_rate=0.7,
                network_scale=1.0,
                deadline_scale=1.55,
                queue_mode="bursty",
                queue_scale=1.5,
                profile_mode="tradeoff_v2",
                min_nodes=5,
                max_nodes=12,
                task_graph_mode="standard",
                quality_propagation_strength=1.5,
            )


if __name__ == "__main__":
    main()
