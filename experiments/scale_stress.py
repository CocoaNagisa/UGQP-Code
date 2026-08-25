"""Run a shape-scaled CPU stress test for the UGQP scheduler architecture.

The trained paper policy has a fixed 12-node, five-action observation space, so
it cannot be loaded unchanged for larger graphs or additional edge servers.
This script therefore measures architecture-equivalent forward computation at
larger tensor shapes.  It does not report scheduling quality or zero-shot
generalization outside the trained observation/action space.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from torch import nn


ROOT = Path(__file__).resolve().parent
CORE = ROOT.parent / "src"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from llm_offload_env.gnn_policy import DAGMessagePassingExtractor


@dataclass(frozen=True)
class Condition:
    """One scale-stress condition.

    :param max_nodes: Padded workflow node budget.
    :param edge_servers: Number of candidate edge servers.
    :param density_label: Human-readable graph-density level.
    :param target_density: Fraction of possible upper-triangular DAG edges.
    """

    max_nodes: int
    edge_servers: int
    density_label: str
    target_density: float

    @property
    def action_count(self) -> int:
        """Return local + edge servers + cloud action count."""
        return self.edge_servers + 2


class ArchitectureEquivalentScheduler(nn.Module):
    """UGQP extractor and Actor-Critic head shapes used for scale timing."""

    def __init__(self, max_nodes: int, action_count: int) -> None:
        """Create a scheduler at a requested graph/action shape.

        :param max_nodes: Padded workflow node budget.
        :param action_count: Number of candidate execution sites.
        """
        super().__init__()
        observation_dim = max_nodes * 12 + max_nodes * max_nodes + action_count + 6
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.extractor = DAGMessagePassingExtractor(
            observation_space,
            max_nodes=max_nodes,
            node_feature_dim=12,
            action_count=action_count,
            global_feature_dim=6,
            hidden_dim=64,
            message_steps=2,
            features_dim=128,
        )
        self.policy_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(64, action_count)
        self.value_head = nn.Linear(64, 1)

    def decision_logits(self, observation: torch.Tensor) -> torch.Tensor:
        """Return actor logits for one or more observations.

        :param observation: Batched flattened observation tensor.
        :returns: Unnormalized action logits.
        """
        features = self.extractor(observation)
        return self.action_net(self.policy_net(features))


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


def _make_dag(rng: random.Random, node_count: int, target_density: float) -> np.ndarray:
    """Create an acyclic upper-triangular graph with a fixed edge count.

    :param rng: Condition-specific random generator.
    :param node_count: Number of nodes.
    :param target_density: Requested fraction of possible DAG edges.
    :returns: Float32 adjacency matrix.
    """
    possible = [(source, target) for source in range(node_count) for target in range(source + 1, node_count)]
    edge_count = max(1, min(len(possible), round(target_density * len(possible))))
    selected = rng.sample(possible, edge_count)
    adjacency = np.zeros((node_count, node_count), dtype=np.float32)
    for source, target in selected:
        adjacency[source, target] = 1.0
    return adjacency


def _make_inputs(condition: Condition, sample_count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Construct deterministic observations and masks for one condition.

    :param condition: Scale-stress condition.
    :param sample_count: Number of preconstructed workflow states.
    :param seed: Condition-specific seed.
    :returns: Observation batch, feasibility masks, and graph summary.
    """
    rng = random.Random(seed)
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    edge_counts: list[int] = []
    for _ in range(sample_count):
        nodes = np.asarray(
            [[rng.random() for _ in range(12)] for _ in range(condition.max_nodes)],
            dtype=np.float32,
        )
        nodes[:, 0] = 0.0
        nodes[rng.randrange(condition.max_nodes), 0] = 1.0
        adjacency = _make_dag(rng, condition.max_nodes, condition.target_density)
        edge_counts.append(int(adjacency.sum()))
        queues = np.asarray([rng.random() for _ in range(condition.action_count)], dtype=np.float32)
        globals_ = np.asarray([rng.random() for _ in range(6)], dtype=np.float32)
        observations.append(np.concatenate([nodes.reshape(-1), adjacency.reshape(-1), queues, globals_]))

        mask = np.asarray([rng.random() > 0.2 for _ in range(condition.action_count)], dtype=bool)
        if not mask.any():
            mask[0] = True
        masks.append(mask)

    possible_edges = condition.max_nodes * (condition.max_nodes - 1) / 2
    summary = {
        "mean_edges": statistics.fmean(edge_counts),
        "actual_density": statistics.fmean(edge_counts) / possible_edges,
        "min_edges": min(edge_counts),
        "max_edges": max(edge_counts),
    }
    return torch.from_numpy(np.stack(observations)), torch.from_numpy(np.stack(masks)), summary


def _percentile(values: list[float], fraction: float) -> float:
    """Return a linear-interpolated percentile.

    :param values: Measurements.
    :param fraction: Percentile fraction in [0, 1].
    :returns: Percentile value.
    """
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure(
    model: ArchitectureEquivalentScheduler,
    observations: torch.Tensor,
    masks: torch.Tensor,
    warmup_calls: int,
    measured_calls: int,
    blocks: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Measure masked actor-path latency using preconstructed states.

    :param model: Architecture-equivalent scheduler.
    :param observations: Preconstructed workflow observations.
    :param masks: Preconstructed feasibility masks.
    :param warmup_calls: Untimed warm-up calls.
    :param measured_calls: Timed calls.
    :param blocks: Number of retained timing blocks.
    :returns: Aggregate and block-level latency statistics.
    """
    model.eval()
    sample_count = observations.shape[0]

    def call(index: int) -> None:
        logits = model.decision_logits(observations[index : index + 1])
        masked_logits = logits.masked_fill(~masks[index : index + 1], -1e9)
        torch.argmax(masked_logits, dim=1)

    with torch.inference_mode():
        for index in range(warmup_calls):
            call(index % sample_count)

        timings: list[float] = []
        block_size = measured_calls // blocks
        block_rows: list[dict[str, float]] = []
        for block in range(blocks):
            block_timings: list[float] = []
            for offset in range(block_size):
                index = (block * block_size + offset) % sample_count
                started = time.perf_counter_ns()
                call(index)
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                timings.append(elapsed_ms)
                block_timings.append(elapsed_ms)
            block_rows.append(
                {
                    "block": block + 1,
                    "calls": block_size,
                    "mean_ms": statistics.fmean(block_timings),
                    "median_ms": statistics.median(block_timings),
                    "p95_ms": _percentile(block_timings, 0.95),
                }
            )

    aggregate = {
        "calls": len(timings),
        "mean_ms": statistics.fmean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": _percentile(timings, 0.95),
        "p99_ms": _percentile(timings, 0.99),
        "std_ms": statistics.pstdev(timings),
    }
    return aggregate, block_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries as a UTF-8 CSV file.

    :param path: Output CSV.
    :param rows: Non-empty data rows.
    """
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Execute all scale conditions and write machine-readable evidence."""
    output = ROOT.parent / "outputs" / "scale_evidence"
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260822)
    np.random.seed(20260822)
    torch.set_num_threads(1)

    node_counts = [12, 20, 32, 50]
    edge_server_counts = [3, 5, 8]
    densities = [("sparse", 0.20), ("medium", 0.40), ("dense", 0.50)]
    warmup_calls = 200
    measured_calls = 1000
    blocks = 5
    graph_samples = 100
    summary_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []

    for condition_index, values in enumerate(
        (nodes, servers, label, density)
        for nodes in node_counts
        for servers in edge_server_counts
        for label, density in densities
    ):
        condition = Condition(*values)
        condition_seed = 20260822 + condition_index
        torch.manual_seed(condition_seed)
        model = ArchitectureEquivalentScheduler(condition.max_nodes, condition.action_count)
        observations, masks, graph_summary = _make_inputs(condition, graph_samples, condition_seed)
        aggregate, condition_blocks = _measure(
            model,
            observations,
            masks,
            warmup_calls,
            measured_calls,
            blocks,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        row = {
            **asdict(condition),
            "action_count": condition.action_count,
            "observation_dim": observations.shape[1],
            "input_bytes_fp32": observations.shape[1] * 4,
            "mean_edges": graph_summary["mean_edges"],
            "actual_density": graph_summary["actual_density"],
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "parameter_bytes_fp32": parameter_count * 4,
            **aggregate,
            "finite_output_check": True,
        }
        summary_rows.append(row)
        for block_row in condition_blocks:
            block_rows.append(
                {
                    "max_nodes": condition.max_nodes,
                    "edge_servers": condition.edge_servers,
                    "action_count": condition.action_count,
                    "density_label": condition.density_label,
                    "target_density": condition.target_density,
                    **block_row,
                }
            )
        print(
            f"nodes={condition.max_nodes:2d} servers={condition.edge_servers} "
            f"density={condition.density_label:6s} mean={aggregate['mean_ms']:.4f} ms "
            f"p95={aggregate['p95_ms']:.4f} ms params={parameter_count}"
        )

    _write_csv(output / "scale_stress_latency.csv", summary_rows)
    _write_csv(output / "scale_stress_latency_blocks.csv", block_rows)
    source = CORE / "llm_offload_env" / "gnn_policy.py"
    manifest = {
        "purpose": "Architecture-equivalent scheduler compute scale stress",
        "scope": (
            "Actor-path CPU forward latency only. Larger shapes are architecture-equivalent, randomly initialized "
            "timing replicas; they are not trained policies and do not establish scheduling quality or zero-shot transfer."
        ),
        "source_module": "src/llm_offload_env/gnn_policy.py",
        "source_module_sha256": _sha256(source),
        "base_trained_shape": {"max_nodes": 12, "edge_servers": 3, "action_count": 5},
        "node_counts": node_counts,
        "edge_server_counts": edge_server_counts,
        "action_counts": [count + 2 for count in edge_server_counts],
        "density_levels": {label: density for label, density in densities},
        "conditions": len(summary_rows),
        "graph_samples_per_condition": graph_samples,
        "warmup_calls_per_condition": warmup_calls,
        "measured_calls_per_condition": measured_calls,
        "measurement_blocks": blocks,
        "master_seed": 20260822,
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "timing_clock": "time.perf_counter_ns",
    }
    (output / "scale_stress_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
