from __future__ import annotations

import argparse
import random
from pathlib import Path

from common_io import summarize_episode_rows, write_csv, write_json
from llm_offload_env.baselines import BASELINES, select_action
from llm_offload_env.config import make_config
from llm_offload_env.env import LLMAgentOffloadEnv


def evaluate_baselines(
    output: Path,
    episodes: int,
    seed: int,
    policies: list[str] | None,
    arrival_rate: float,
    network_scale: float,
    deadline_scale: float,
    queue_mode: str = "bursty",
    queue_scale: float = 1.0,
    profile_mode: str = "tradeoff_v2",
    min_nodes: int = 5,
    max_nodes: int = 12,
):
                                               
    output.mkdir(parents=True, exist_ok=True)
                               
    policies = policies or BASELINES
    episode_rows = []
    action_rows = []
    for policy in policies:
                                                               
        rng = random.Random(seed + len(policy))
        for ep in range(episodes):
                                                   
            env = LLMAgentOffloadEnv(make_config(
                seed=seed + ep,
                variant="full",
                arrival_rate=arrival_rate,
                network_scale=network_scale,
                deadline_scale=deadline_scale,
                queue_mode=queue_mode,
                queue_scale=queue_scale,
                profile_mode=profile_mode,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
            ))
            _, _ = env.reset()
            done = False
            while not done:
                                                       
                action = select_action(env, policy, rng)
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                                   
            episode_rows.append(env.episode_summary(policy, ep))
            action_rows.extend(env.action_records(policy, ep))
                        
    write_csv(output / "episode_records.csv", episode_rows)
    write_csv(output / "action_records.csv", action_rows)
    write_csv(output / "baseline_summary.csv", summarize_episode_rows(episode_rows))
                       
    write_json(output / "baseline_config.json", {
        "episodes": episodes,
        "seed": seed,
        "policies": policies,
        "arrival_rate": arrival_rate,
        "network_scale": network_scale,
        "deadline_scale": deadline_scale,
        "queue_mode": queue_mode,
        "queue_scale": queue_scale,
        "profile_mode": profile_mode,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
    })


def main() -> None:
                            
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--arrival-rate", type=float, default=0.7)
    parser.add_argument("--network-scale", type=float, default=1.0)
    parser.add_argument("--deadline-scale", type=float, default=1.0)
    parser.add_argument("--queue-mode", choices=["legacy", "bursty"], default="bursty")
    parser.add_argument("--queue-scale", type=float, default=1.0)
    parser.add_argument("--profile-mode", choices=["legacy_v1", "tradeoff_v2"], default="tradeoff_v2")
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=12)
    args = parser.parse_args()
                    
    evaluate_baselines(
        args.output,
        args.episodes,
        args.seed,
        None,
        args.arrival_rate,
        args.network_scale,
        args.deadline_scale,
        args.queue_mode,
        args.queue_scale,
        args.profile_mode,
        args.min_nodes,
        args.max_nodes,
    )
    print(f"saved baseline results: {args.output}")


if __name__ == "__main__":
    main()
