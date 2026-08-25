"""Evaluate trained UGQP policies under controlled DAG-density perturbations."""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO


ROOT = Path(__file__).resolve().parent
CORE = ROOT.parent / "src"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from llm_offload_env.config import make_config
from llm_offload_env.env import LLMAgentOffloadEnv
from llm_offload_env.gnn_policy import (
    DAGMessagePassingExtractor,
    DTODRLGATExtractor,
)
from llm_offload_env.taskgen import add_edge, compute_graph_features


SEEDS = [2026, 2027, 2028]
NODE_BUDGETS = [12, 20, 32]
DENSITIES = [("sparse", 0.20), ("medium", 0.40), ("dense", 0.50)]
EPISODES = 200
EVALUATION_SEED = 30_000


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _set_density(env: LLMAgentOffloadEnv, target_density: float, episode_seed: int) -> None:
    """Replace task edges with a connected, deterministic, nested-density DAG.

    :param env: Reset environment whose node attributes are retained.
    :param target_density: Fraction of possible upper-triangular edges.
    :param episode_seed: Seed used to rank optional edges.
    """
    assert env.task is not None
    nodes = env.task.nodes
    node_count = len(nodes)
    for node in nodes:
        node.predecessors.clear()
        node.successors.clear()
    rng = random.Random(episode_seed)
    mandatory = [(rng.randrange(target), target) for target in range(1, node_count)]
    mandatory_set = set(mandatory)
    optional = [
        (source, target)
        for source in range(node_count)
        for target in range(source + 1, node_count)
        if (source, target) not in mandatory_set
    ]
    rng.shuffle(optional)
    possible_count = node_count * (node_count - 1) // 2
    target_count = max(len(mandatory), round(target_density * possible_count))
    for source, target in mandatory + optional[: target_count - len(mandatory)]:
        add_edge(nodes, source, target)
    compute_graph_features(nodes)


def _model_paths() -> dict[tuple[int, int], Path]:
    """Load model locations written by the larger-workflow experiment."""
    data_root = ROOT.parent.parent / "data"
    paths = {}
    for nodes in NODE_BUDGETS:
        for seed in SEEDS:
            if nodes == 12:
                path = data_root / "main_comparison" / f"seed_{seed}" / "ugqp_gnn" / "model" / "ppo_full.zip"
            else:
                path = data_root / "scalability" / "effectiveness" / f"node_{nodes}" / f"seed_{seed}" / "model" / "ppo_full.zip"
            paths[(nodes, seed)] = path
    return paths


def main() -> None:
    """Run controlled-density evaluation and write raw and aggregate evidence."""
    output = ROOT.parent.parent / "data" / "scalability" / "density_effectiveness"
    output.mkdir(parents=True, exist_ok=True)
    model_paths = _model_paths()
    episode_rows: list[dict[str, object]] = []

    for nodes in NODE_BUDGETS:
        for seed in SEEDS:
            model_path = model_paths[(nodes, seed)]
            model = MaskablePPO.load(model_path, device="cpu")
            for density_label, target_density in DENSITIES:
                condition_rows: list[dict[str, object]] = []
                for episode in range(EPISODES):
                    task_seed = EVALUATION_SEED + episode
                    env = LLMAgentOffloadEnv(
                        make_config(
                            seed=task_seed,
                            variant="full",
                            arrival_rate=0.7,
                            network_scale=1.0,
                            deadline_scale=1.55,
                            queue_mode="bursty",
                            queue_scale=1.5,
                            profile_mode="tradeoff_v2",
                            min_nodes=nodes,
                            max_nodes=nodes,
                            task_graph_mode="standard",
                            quality_propagation_strength=1.50,
                        )
                    )
                    observation, _ = env.reset()
                    _set_density(env, target_density, episode_seed=60_000 + episode)
                    observation = env._obs()
                    done = False
                    while not done:
                        action, _ = model.predict(
                            observation,
                            deterministic=True,
                            action_masks=env.action_masks(),
                        )
                        observation, _, terminated, truncated, _ = env.step(int(action))
                        done = terminated or truncated
                    summary = env.episode_summary(f"UGQP_GNN_n{nodes}_seed{seed}", episode)
                    possible_edges = nodes * (nodes - 1) / 2
                    row = {
                        "node_budget": nodes,
                        "training_seed": seed,
                        "density_label": density_label,
                        "target_density": target_density,
                        "realized_density": float(summary["edges"]) / possible_edges,
                        **summary,
                    }
                    episode_rows.append(row)
                    condition_rows.append(row)
                print(
                    f"nodes={nodes} seed={seed} density={density_label} "
                    f"quality={_mean(condition_rows, 'avg_quality'):.4f} "
                    f"DSR={1.0 - _mean(condition_rows, 'deadline_violation'):.4f}"
                )

    aggregate_rows: list[dict[str, object]] = []
    for nodes in NODE_BUDGETS:
        for density_label, target_density in DENSITIES:
            subset = [
                row
                for row in episode_rows
                if int(row["node_budget"]) == nodes and row["density_label"] == density_label
            ]
            aggregate_rows.append(
                {
                    "node_budget": nodes,
                    "density_label": density_label,
                    "target_density": target_density,
                    "training_seeds": len(SEEDS),
                    "episodes": len(subset),
                    "mean_realized_density": _mean(subset, "realized_density"),
                    "mean_total_latency": _mean(subset, "total_latency"),
                    "mean_latency_per_node": _mean(subset, "total_latency") / nodes,
                    "deadline_satisfaction_rate": 1.0 - _mean(subset, "deadline_violation"),
                    "mean_avg_quality": _mean(subset, "avg_quality"),
                    "success_rate": _mean(subset, "success"),
                    "mean_cost_per_node": _mean(subset, "total_cost") / nodes,
                    "mean_high_risk_local_ratio": _mean(subset, "high_risk_local_ratio"),
                }
            )

    _write_csv(output / "density_episode_records.csv", episode_rows)
    _write_csv(output / "density_aggregate.csv", aggregate_rows)
    manifest = {
        "purpose": "Controlled graph-density effectiveness evaluation",
        "node_budgets": NODE_BUDGETS,
        "training_seeds": SEEDS,
        "densities": {label: density for label, density in DENSITIES},
        "episodes_per_seed_condition": EPISODES,
        "evaluation_seed": EVALUATION_SEED,
        "edge_construction": (
            "Connected upper-triangular DAGs. Each non-root node receives one deterministic random parent, matching "
            "the main generator's skeleton; optional edges use one per-episode ranking, so density edge sets are nested."
        ),
        "scope": (
            "Evaluation-only graph-density perturbation of trained policies; no density-specific retraining. "
            "Node attributes, serving settings, and evaluation seeds are held fixed across density levels."
        ),
    }
    (output / "density_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
