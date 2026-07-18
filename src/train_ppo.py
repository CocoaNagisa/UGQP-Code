from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from common_io import ensure_dir, write_json
from llm_offload_env.config import ACTIONS, make_config
from llm_offload_env.env import LLMAgentOffloadEnv
from llm_offload_env.gnn_policy import DAGMessagePassingExtractor, DTODRLGATExtractor


def make_env(
    seed: int,
    variant: str,
    arrival_rate: float,
    network_scale: float,
    deadline_scale: float,
    queue_mode: str = "bursty",
    queue_scale: float = 1.0,
    profile_mode: str = "tradeoff_v2",
    min_nodes: int = 5,
    max_nodes: int = 12,
    task_graph_mode: str = "standard",
    quality_propagation_strength: float = 1.50,
):
                                                
    def _factory():
        cfg = make_config(
            seed=seed,
            variant=variant,
            arrival_rate=arrival_rate,
            network_scale=network_scale,
            deadline_scale=deadline_scale,
            queue_mode=queue_mode,
            queue_scale=queue_scale,
            profile_mode=profile_mode,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            task_graph_mode=task_graph_mode,
            quality_propagation_strength=quality_propagation_strength,
        )
        return LLMAgentOffloadEnv(cfg)

    return _factory


def _policy_kwargs(
    env: DummyVecEnv,
    encoder: str,
    dtodrl_checkpoint: Path | None = None,
) -> dict | None:
                                                                   
    if encoder == "mlp":
        return None

    raw_env = env.envs[0]
    if encoder == "dtodrl":
        if dtodrl_checkpoint is None or not dtodrl_checkpoint.exists():
            raise FileNotFoundError(
                "encoder=dtodrl requires a pretrained --dtodrl-checkpoint"
            )
        return {
            "features_extractor_class": DTODRLGATExtractor,
            "features_extractor_kwargs": {
                "max_nodes": raw_env.config.max_nodes,
                "node_feature_dim": raw_env.node_feature_dim,
                "action_count": len(ACTIONS),
                "global_feature_dim": 6,
                "hidden_dim": 32,
                "features_dim": 128,
                "checkpoint_path": str(dtodrl_checkpoint.resolve()),
                "freeze_gat": True,
            },
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
        }
    return {
        "features_extractor_class": DAGMessagePassingExtractor,
        "features_extractor_kwargs": {
            "max_nodes": raw_env.config.max_nodes,
            "node_feature_dim": raw_env.node_feature_dim,
            "action_count": len(ACTIONS),
            "global_feature_dim": 6,
            "hidden_dim": 64,
            "message_steps": 2,
            "features_dim": 128,
        },
        "net_arch": {"pi": [128, 64], "vf": [128, 64]},
    }


def train(
    output: Path,
    timesteps: int,
    seed: int,
    variant: str,
    arrival_rate: float,
    network_scale: float,
    deadline_scale: float,
    encoder: str = "gnn",
    queue_mode: str = "bursty",
    queue_scale: float = 1.0,
    ent_coef: float = 0.05,
    profile_mode: str = "tradeoff_v2",
    n_envs: int = 4,
    min_nodes: int = 5,
    max_nodes: int = 12,
    task_graph_mode: str = "standard",
    quality_propagation_strength: float = 1.50,
    dtodrl_checkpoint: Path | None = None,
) -> Path:
               
    ensure_dir(output)
                                
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1")
    env = DummyVecEnv([
        make_env(
            seed + 1009 * env_idx,
            variant,
            arrival_rate,
            network_scale,
            deadline_scale,
            queue_mode,
            queue_scale,
            profile_mode,
            min_nodes,
            max_nodes,
            task_graph_mode,
            quality_propagation_strength,
        )
        for env_idx in range(n_envs)
    ])
    policy_kwargs = _policy_kwargs(env, encoder, dtodrl_checkpoint)

                                                         
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        n_steps=128,
        batch_size=64,
        gamma=0.96,
        learning_rate=3e-4,
        ent_coef=ent_coef,
        clip_range=0.2,
        tensorboard_log=str(output / "tb"),
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)

    model_path = output / f"ppo_{variant}.zip"
    model.save(model_path)
    write_json(output / "train_config.json", {
        "timesteps": timesteps,
        "seed": seed,
        "variant": variant,
        "arrival_rate": arrival_rate,
        "network_scale": network_scale,
        "deadline_scale": deadline_scale,
        "encoder": encoder,
        "queue_mode": queue_mode,
        "queue_scale": queue_scale,
        "ent_coef": ent_coef,
        "profile_mode": profile_mode,
        "n_envs": n_envs,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
        "task_graph_mode": task_graph_mode,
        "quality_propagation_strength": quality_propagation_strength,
        "dtodrl_checkpoint": str(dtodrl_checkpoint) if dtodrl_checkpoint else None,
        "environment_seeds": [seed + 1009 * idx for idx in range(n_envs)],
        "model_path": str(model_path),
    })
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/train/full"))
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--variant", choices=["full", "no_uncertainty", "no_graph", "no_queue", "no_message_passing"], default="full")
    parser.add_argument("--arrival-rate", type=float, default=0.7)
    parser.add_argument("--network-scale", type=float, default=1.0)
    parser.add_argument("--deadline-scale", type=float, default=1.0)
    parser.add_argument("--queue-mode", choices=["legacy", "bursty"], default="bursty")
    parser.add_argument("--queue-scale", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--profile-mode", choices=["legacy_v1", "tradeoff_v2"], default="tradeoff_v2")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--encoder", choices=["gnn", "mlp", "dtodrl"], default="gnn")
    parser.add_argument("--dtodrl-checkpoint", type=Path)
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--task-graph-mode", choices=["standard", "dependency_stress"], default="standard")
    parser.add_argument("--quality-propagation-strength", type=float, default=1.50)
    args = parser.parse_args()

    path = train(
        args.output,
        args.timesteps,
        args.seed,
        args.variant,
        args.arrival_rate,
        args.network_scale,
        args.deadline_scale,
        args.encoder,
        args.queue_mode,
        args.queue_scale,
        args.ent_coef,
        args.profile_mode,
        args.n_envs,
        args.min_nodes,
        args.max_nodes,
        args.task_graph_mode,
        args.quality_propagation_strength,
        args.dtodrl_checkpoint,
    )
    print(f"saved model: {path}")


if __name__ == "__main__":
    main()
