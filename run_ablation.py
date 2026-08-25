"""Run the component ablation protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from infer_ppo import evaluate_model
from train_ppo import train


SEEDS = (2082, 2083, 2084)
VARIANTS = {
    "full": ("gnn", "UGQP-GNN"),
    "ppo_mlp": ("mlp", "PPO-MLP"),
    "no_graph": ("mlp", "w/o graph information"),
    "no_uncertainty_risk_mask": ("gnn", "w/o uncertainty and risk mask"),
    "no_queue": ("gnn", "queue-blind within-tier router"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "load_stratified_ablation")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for seed in SEEDS:
        for variant in [item.strip() for item in args.variants.split(",") if item.strip()]:
            encoder, label = VARIANTS[variant]
            model_dir = args.output / "models" / f"seed_{seed}" / variant
            model_path = model_dir / f"ppo_{variant}.zip"
            if not (args.resume and model_path.exists()):
                model_path = train(
                    model_dir,
                    args.timesteps,
                    seed=seed,
                    variant=variant,
                    arrival_rate=0.7,
                    network_scale=1.0,
                    deadline_scale=1.25,
                    encoder=encoder,
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
            evaluation_dir = args.output / "evaluation" / f"seed_{seed}" / variant
            if not (args.resume and (evaluation_dir / "summary.csv").exists()):
                evaluate_model(
                    model_path,
                    evaluation_dir,
                    episodes=args.episodes,
                    seed=20000,
                    variant=variant,
                    policy_name=label,
                    arrival_rate=0.7,
                    network_scale=1.0,
                    deadline_scale=1.25,
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
