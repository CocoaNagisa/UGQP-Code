"""Run the four learned policies used in the main comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from infer_ppo import evaluate_model
from train_ppo import train


SEEDS = (2024, 2025, 2026, 2027, 2028)
METHODS = {
    "ugqp_gnn": {"encoder": "gnn", "freeze": True, "label": "UGQP-GNN"},
    "dtodrl_frozen": {"encoder": "dtodrl", "freeze": True, "label": "DTODRL-frozen"},
    "dtodrl_joint": {"encoder": "dtodrl", "freeze": False, "label": "DTODRL-joint"},
    "ppo_mlp": {"encoder": "mlp", "freeze": True, "label": "PPO-MLP"},
}


def run_method(
    output: Path,
    method_key: str,
    seed: int,
    timesteps: int,
    episodes: int,
    checkpoint: Path,
    resume: bool,
) -> None:
    method = METHODS[method_key]
    model_dir = output / "models" / f"seed_{seed}" / method_key
    model_path = model_dir / "ppo_full.zip"
    if not (resume and model_path.exists()):
        model_path = train(
            model_dir,
            timesteps,
            seed=seed,
            variant="full",
            arrival_rate=0.7,
            network_scale=1.0,
            deadline_scale=1.55,
            encoder=method["encoder"],
            queue_mode="bursty",
            queue_scale=1.5,
            ent_coef=0.05,
            profile_mode="tradeoff_v2",
            n_envs=4,
            min_nodes=5,
            max_nodes=12,
            task_graph_mode="standard",
            quality_propagation_strength=1.5,
            dtodrl_checkpoint=checkpoint if method["encoder"] == "dtodrl" else None,
            dtodrl_freeze_gat=method["freeze"],
        )
    evaluation_dir = output / "evaluation" / f"seed_{seed}" / method_key
    if not (resume and (evaluation_dir / "summary.csv").exists()):
        evaluate_model(
            model_path,
            evaluation_dir,
            episodes=episodes,
            seed=20000,
            variant="full",
            policy_name=method["label"],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "main_comparison")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--timesteps", type=int, default=60000)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    checkpoint = ROOT / "models" / "dtodrl_gat_pretrained.pt"
    selected = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        raise SystemExit(f"Unknown method: {', '.join(unknown)}")
    if any(METHODS[name]["encoder"] == "dtodrl" for name in selected) and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    for seed in SEEDS:
        for method_key in selected:
            run_method(args.output, method_key, seed, args.timesteps, args.episodes, checkpoint, args.resume)


if __name__ == "__main__":
    main()
