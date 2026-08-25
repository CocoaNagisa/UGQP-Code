from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import ACTIONS, ID_TO_ACTION, EnvConfig, NODE_TYPES
from .serving import (
    advance_background_queues,
    cost,
    default_profiles,
    feasible,
    initialize_background_queues,
    latency,
    quality,
    update_queue,
)
from .taskgen import Task, generate_task


class LLMAgentOffloadEnv(gym.Env):
    """Gymnasium environment for node-level workflow placement."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.rng = random.Random(self.config.seed)
        self.queue_rng = random.Random(self.config.seed + 1_000_003)
        self.observation_rng = random.Random(self.config.seed + 2_000_003)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.node_feature_dim = 12
        self.adjacency_dim = self.config.max_nodes * self.config.max_nodes
        obs_dim = self.config.max_nodes * self.node_feature_dim + self.adjacency_dim + len(ACTIONS) + 6
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.task: Task | None = None
        self.profiles = default_profiles(self.config.profile_mode)
        self.ready_order: list[int] = []
        self.current_pos = 0
        self.completion: dict[int, float] = {}
        self.realized_quality: dict[int, float] = {}
        self.node_results: list[dict[str, Any]] = []
        self.episode_id = 0
        self.total_reward = 0.0
        self.queue_burst_remaining: dict[str, int] = {}
        self.node_permutation = list(range(self.config.max_nodes))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng.seed(seed)
            self.queue_rng.seed(seed + 1_000_003)
            self.observation_rng.seed(seed + 2_000_003)
        self.profiles = default_profiles(self.config.profile_mode)
        self.queue_burst_remaining = initialize_background_queues(
            self.profiles,
            self.queue_rng,
            self.config.arrival_rate,
            self.config.queue_mode,
            self.config.queue_scale,
        )
        self.task = generate_task(
            self.rng,
            task_id=self.episode_id,
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            deadline_scale=self.config.deadline_scale,
            graph_mode=self.config.task_graph_mode,
        )
        self.ready_order = list(range(len(self.task.nodes)))
        self.node_permutation = list(range(self.config.max_nodes))
        if self.config.permute_node_order:
            self.observation_rng.shuffle(self.node_permutation)
        self.current_pos = 0
        self.completion = {}
        self.realized_quality = {}
        self.node_results = []
        self.total_reward = 0.0
        self.episode_id += 1
        return self._obs(), {}

    def step(self, action: int):
        assert self.task is not None
        node = self.task.nodes[self.ready_order[self.current_pos]]
        mask = self.action_masks()

        invalid = mask[action] == 0
        if invalid:
            feasible_ids = [idx for idx, ok in enumerate(mask) if ok]
            action = feasible_ids[0]

        action_name = ID_TO_ACTION[int(action)]
        profile = self.profiles[action_name]
        queue_snapshot = {name: p.queue_load for name, p in self.profiles.items()}

        step_latency = latency(node, profile, self.rng, self.config.network_scale)
        base_step_quality = quality(node, profile)
        predecessor_quality = float(np.mean([
            self.realized_quality[pred]
            for pred in node.predecessors
            if pred in self.realized_quality
        ])) if node.predecessors else 1.0
        recovery_factor = max(0.0, 1.0 - profile.capability)
        propagation_penalty = (
            self.config.quality_propagation_strength
            * (1.0 - predecessor_quality)
            * node.downstream_risk
            * recovery_factor
        )
        step_quality = min(1.0, max(0.0, base_step_quality - propagation_penalty))
        step_cost = cost(node, profile)

        start_time = max([self.completion[p] for p in node.predecessors], default=0.0)
        finish_time = start_time + step_latency
        self.completion[node.idx] = finish_time
        self.realized_quality[node.idx] = step_quality

        update_queue(profile, node, step_latency, self.config.arrival_rate)
        advance_background_queues(
            self.profiles,
            self.queue_rng,
            self.config.arrival_rate,
            self.config.queue_mode,
            self.config.queue_scale,
            self.config.queue_burst_probability,
            self.config.queue_burst_multiplier,
            self.config.queue_burst_min_steps,
            self.config.queue_burst_max_steps,
            self.queue_burst_remaining,
        )

        high_risk_local = 1 if action_name == "local" and node.uncertainty > 0.60 else 0
        load_var = self._load_variance()
        latency_norm = max(self.config.reward_latency_normalizer, 1e-6)
        reward = (
            -self.config.reward_latency_weight * (step_latency / latency_norm)
            -self.config.reward_cost_weight * step_cost
            +self.config.reward_quality_weight * (step_quality - 0.70)
            -self.config.reward_balance_weight * load_var
            -1.25 * high_risk_local
        )
        if invalid:
            reward -= self.config.invalid_action_penalty

        self.node_results.append({
            "step": self.current_pos,
            "node_id": node.idx,
            "node_kind": node.kind,
            "action": action_name,
            "feasible_actions": ",".join(ACTIONS[i] for i, ok in enumerate(mask) if ok),
            "latency": step_latency,
            "quality": step_quality,
            "base_quality": base_step_quality,
            "predecessor_quality": predecessor_quality,
            "propagation_penalty": propagation_penalty,
            "recovery_factor": recovery_factor,
            "cost": step_cost,
            "uncertainty": node.uncertainty,
            "first_token_p1": node.first_token_p1,
            "first_token_p2": node.first_token_p2,
            "token_uncertainty": node.token_uncertainty,
            "criticality": node.criticality,
            "downstream_risk": node.downstream_risk,
            "start_time": start_time,
            "finish_time": finish_time,
            "high_risk_local": high_risk_local,
            "chosen_queue_wait": queue_snapshot[action_name],
            "queue_local": queue_snapshot["local"],
            "queue_edge_s": queue_snapshot["edge_s"],
            "queue_edge_m": queue_snapshot["edge_m"],
            "queue_edge_l": queue_snapshot["edge_l"],
            "queue_cloud": queue_snapshot["cloud"],
        })

        self.current_pos += 1
        terminated = self.current_pos >= len(self.task.nodes)
        if terminated:
            makespan = max(self.completion.values()) if self.completion else 0.0
            if makespan > self.task.deadline:
                reward -= self.config.reward_deadline_weight * (
                    (makespan - self.task.deadline) / latency_norm
                )
        self.total_reward += reward
        return self._obs(), float(reward), terminated, False, {}

    def action_masks(self) -> np.ndarray:
        assert self.task is not None
        if self.current_pos >= len(self.task.nodes):
            return np.ones(len(ACTIONS), dtype=bool)

        node = self.task.nodes[self.ready_order[self.current_pos]]
        if self.config.variant in {"no_uncertainty", "no_uncertainty_risk_mask"}:
            node = replace(node, uncertainty=0.0, risk=0.0)
        mask = np.zeros(len(ACTIONS), dtype=bool)
        for idx, action in ID_TO_ACTION.items():
            mask[idx] = feasible(node, self.profiles[action])
        if not mask.any():
            mask[-1] = True
        return mask

    def episode_summary(self, policy: str, episode: int) -> dict[str, Any]:
        assert self.task is not None
        total_latency = max(self.completion.values()) if self.completion else 0.0
        avg_quality = float(np.mean([r["quality"] for r in self.node_results])) if self.node_results else 0.0
        total_cost = float(np.sum([r["cost"] for r in self.node_results])) if self.node_results else 0.0
        high_risk_local = float(np.mean([r["high_risk_local"] for r in self.node_results])) if self.node_results else 0.0
        offload_ratio = float(np.mean([1 if r["action"] != "local" else 0 for r in self.node_results])) if self.node_results else 0.0
        deadline_violation = int(total_latency > self.task.deadline)
        success = int(avg_quality >= 0.72 and high_risk_local <= 0.05 and not deadline_violation)
        chosen_queues = [float(r["chosen_queue_wait"]) for r in self.node_results]
        remote_queues = [
            float(r[key])
            for r in self.node_results
            for key in ("queue_edge_s", "queue_edge_m", "queue_edge_l", "queue_cloud")
        ]
        return {
            "policy": policy,
            "episode": episode,
            "scenario": self.task.scenario,
            "nodes": len(self.task.nodes),
            "edges": sum(len(node.successors) for node in self.task.nodes),
            "max_indegree": max((len(node.predecessors) for node in self.task.nodes), default=0),
            "graph_depth": self._graph_depth(),
            "total_latency": total_latency,
            "deadline": self.task.deadline,
            "deadline_violation": deadline_violation,
            "avg_quality": avg_quality,
            "success": success,
            "total_cost": total_cost,
            "high_risk_local_ratio": high_risk_local,
            "offload_ratio": offload_ratio,
            "load_variance": self._load_variance(),
            "reward": self.total_reward,
            "mean_chosen_queue_wait": float(np.mean(chosen_queues)) if chosen_queues else 0.0,
            "mean_observed_remote_queue": float(np.mean(remote_queues)) if remote_queues else 0.0,
            "max_observed_remote_queue": float(np.max(remote_queues)) if remote_queues else 0.0,
        }

    def action_records(self, policy: str, episode: int) -> list[dict[str, Any]]:
        return [{"policy": policy, "episode": episode, **row} for row in self.node_results]

    def _obs(self) -> np.ndarray:
        assert self.task is not None
        node_rows: list[list[float]] = []
        current_node_id = self.ready_order[self.current_pos] if self.current_pos < len(self.ready_order) else -1

        for idx in range(self.config.max_nodes):
            if idx < len(self.task.nodes):
                node = self.task.nodes[idx]
                is_current = 1.0 if idx == current_node_id else 0.0
                executed = 1.0 if idx in self.completion else 0.0
                risk_blind = self.config.variant in {"no_uncertainty", "no_uncertainty_risk_mask"}
                uncertainty = 0.0 if risk_blind else node.uncertainty
                risk = 0.0 if risk_blind else node.risk
                criticality = 0.0 if self.config.variant == "no_graph" else node.criticality
                downstream = 0.0 if self.config.variant in {"no_graph", "no_uncertainty", "no_uncertainty_risk_mask"} else node.downstream_risk
                pred_count = 0.0 if self.config.variant == "no_graph" else min(len(node.predecessors) / 4.0, 1.0)
                realized_quality = self.realized_quality.get(idx, 0.0)
                row = [
                    is_current,
                    executed,
                    min(node.kind_id() / max(len(NODE_TYPES) - 1, 1), 1.0),
                    min(node.input_tokens / 1200.0, 1.0),
                    min(node.output_tokens / 1300.0, 1.0),
                    uncertainty,
                    node.privacy / 3.0,
                    risk,
                    criticality,
                    downstream,
                    pred_count,
                    realized_quality,
                ]
            else:
                row = [0.0] * self.node_feature_dim
            node_rows.append(row)

        adjacency_matrix = np.asarray(
            self._adjacency_features(), dtype=np.float32
        ).reshape(self.config.max_nodes, self.config.max_nodes)
        if self.config.permute_node_order:
            permutation = np.asarray(self.node_permutation, dtype=int)
            node_rows = [node_rows[idx] for idx in permutation]
            adjacency_matrix = adjacency_matrix[np.ix_(permutation, permutation)]
        rows = [value for row in node_rows for value in row]
        adjacency = adjacency_matrix.reshape(-1).tolist()
        queues = []
        for action in ACTIONS:
            q = 0.0 if self.config.variant == "no_queue" else self.profiles[action].queue_load
            queues.append(min(q / 8.0, 1.0))

        progress = self.current_pos / max(len(self.task.nodes), 1)
        deadline_used = (max(self.completion.values()) if self.completion else 0.0) / max(self.task.deadline, 1e-6)
        global_load_variance = 0.0 if self.config.variant == "no_queue" else self._load_variance_norm()
        globals_ = [
            progress,
            min(deadline_used, 1.0),
            min(self.config.arrival_rate / 2.0, 1.0),
            min(self.config.network_scale / 2.0, 1.0),
            len(self.task.nodes) / self.config.max_nodes,
            global_load_variance,
        ]
        return np.asarray(rows + adjacency + queues + globals_, dtype=np.float32)

    def _adjacency_features(self) -> list[float]:
        assert self.task is not None
        if self.config.variant in {"no_graph", "no_message_passing"}:
            return [0.0] * self.adjacency_dim

        values: list[float] = []
        node_count = len(self.task.nodes)
        for src in range(self.config.max_nodes):
            for dst in range(self.config.max_nodes):
                if src < node_count and dst < node_count and dst in self.task.nodes[src].successors:
                    values.append(1.0)
                else:
                    values.append(0.0)
        return values

    def _load_variance(self) -> float:
        loads = [p.queue_load for name, p in self.profiles.items() if name.startswith("edge")]
        if not loads:
            return 0.0
        mean = sum(loads) / len(loads)
        return sum((x - mean) ** 2 for x in loads) / len(loads)

    def _graph_depth(self) -> int:
        assert self.task is not None
        depth = [1] * len(self.task.nodes)
        for node in self.task.nodes:
            if node.predecessors:
                depth[node.idx] = 1 + max(depth[pred] for pred in node.predecessors)
        return max(depth, default=0)

    def _load_variance_norm(self) -> float:
        return min(self._load_variance() / 8.0, 1.0)
