from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from run_external_baselines import TRAIN_SEEDS, run as run_external
from train_ppo import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/main_comparison"))
    parser.add_argument("--timesteps", type=int, default=60000)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_models = args.output / "ugqp_models"
    results = args.output / "results"
    for seed in TRAIN_SEEDS:
        model = source_models / f"seed_{seed}" / "models" / "full" / "ppo_full.zip"
        if args.resume and model.exists():
            continue
        train(
            model.parent,
            args.timesteps,
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
            min_nodes=5,
            max_nodes=12,
            task_graph_mode="standard",
            quality_propagation_strength=1.5,
        )
    run_external(
        results,
        source_models,
        timesteps=args.timesteps,
        episodes=args.episodes,
        resume=args.resume,
        deadline_scale=1.55,
    )


if __name__ == "__main__":
    main()

