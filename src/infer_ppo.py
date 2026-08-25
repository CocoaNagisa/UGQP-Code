from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO

from common_io import summarize_episode_rows, write_csv, write_json
from llm_offload_env.config import make_config
from llm_offload_env.env import LLMAgentOffloadEnv
from llm_offload_env.gnn_policy import (
    DAGMessagePassingExtractor,
    DTODRLGATExtractor,
)


def evaluate_model(
    model_path: Path,
    output: Path,
    episodes: int,
    seed: int,
    variant: str,
    policy_name: str,
    arrival_rate: float,
    network_scale: float,
    deadline_scale: float,
    queue_mode: str = "bursty",
    queue_scale: float = 1.0,
    profile_mode: str = "tradeoff_v2",
    min_nodes: int = 5,
    max_nodes: int = 12,
    permute_node_order: bool = False,
    task_graph_mode: str = "standard",
    quality_propagation_strength: float = 1.50,
):
    output.mkdir(parents=True, exist_ok=True)
    model = MaskablePPO.load(model_path)
    episode_rows = []
    action_rows = []
    for ep in range(episodes):
        env = LLMAgentOffloadEnv(make_config(
            seed=seed + ep,
            variant=variant,
            arrival_rate=arrival_rate,
            network_scale=network_scale,
            deadline_scale=deadline_scale,
            queue_mode=queue_mode,
            queue_scale=queue_scale,
            profile_mode=profile_mode,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            permute_node_order=permute_node_order,
            task_graph_mode=task_graph_mode,
            quality_propagation_strength=quality_propagation_strength,
        ))
        obs, _ = env.reset()
        done = False
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        episode_rows.append(env.episode_summary(policy_name, ep))
        action_rows.extend(env.action_records(policy_name, ep))
    write_csv(output / "episode_records.csv", episode_rows)
    write_csv(output / "action_records.csv", action_rows)
    write_csv(output / "summary.csv", summarize_episode_rows(episode_rows))
    code_root = Path(__file__).resolve().parents[1]
    try:
        portable_model = model_path.resolve().relative_to(code_root.resolve()).as_posix()
    except ValueError:
        portable_model = model_path.name
    write_json(output / "infer_config.json", {
        "model_path": portable_model,
        "episodes": episodes,
        "seed": seed,
        "variant": variant,
        "policy_name": policy_name,
        "arrival_rate": arrival_rate,
        "network_scale": network_scale,
        "deadline_scale": deadline_scale,
        "queue_mode": queue_mode,
        "queue_scale": queue_scale,
        "profile_mode": profile_mode,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
        "permute_node_order": permute_node_order,
        "task_graph_mode": task_graph_mode,
        "quality_propagation_strength": quality_propagation_strength,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/infer"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--variant", choices=["full", "no_uncertainty", "no_uncertainty_risk_mask", "no_graph", "no_queue", "no_message_passing"], default="full")
    parser.add_argument("--policy-name", default="ppo_full")
    parser.add_argument("--arrival-rate", type=float, default=0.7)
    parser.add_argument("--network-scale", type=float, default=1.0)
    parser.add_argument("--deadline-scale", type=float, default=1.0)
    parser.add_argument("--queue-mode", choices=["legacy", "bursty"], default="bursty")
    parser.add_argument("--queue-scale", type=float, default=1.0)
    parser.add_argument("--profile-mode", choices=["legacy_v1", "tradeoff_v2"], default="tradeoff_v2")
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--permute-node-order", action="store_true")
    parser.add_argument("--task-graph-mode", choices=["standard", "dependency_stress"], default="standard")
    parser.add_argument("--quality-propagation-strength", type=float, default=1.50)
    args = parser.parse_args()
    evaluate_model(
        args.model,
        args.output,
        args.episodes,
        args.seed,
        args.variant,
        args.policy_name,
        args.arrival_rate,
        args.network_scale,
        args.deadline_scale,
        args.queue_mode,
        args.queue_scale,
        args.profile_mode,
        args.min_nodes,
        args.max_nodes,
        args.permute_node_order,
        args.task_graph_mode,
        args.quality_propagation_strength,
    )
    print(f"saved inference results: {args.output}")


if __name__ == "__main__":
    main()
