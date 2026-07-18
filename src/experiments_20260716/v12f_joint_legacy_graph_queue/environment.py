from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from experiments_20260715.v12e_queue_multiserver_hotspot import (
    EQUIVALENT_PAIRS,
    INTENSITY_FACTOR,
    LOAD_INTENSITIES,
    LOAD_PATTERNS,
    LOAD_STRATA,
    PROFILE_FEATURE_DIM_V12E,
    SERVER_REGION,
    V12E_ACTIONS,
    V12E_ACTION_TO_ID,
    V12E_EDGE_ACTIONS,
    V12E_ID_TO_ACTION,
    V12E_REMOTE_ACTIONS,
    _static_profile_features,
    multiserver_profiles,
    sampled_regional_latency_components,
    zipf_region_probabilities,
)
from llm_offload_env.config import EnvConfig, NODE_TYPES
from llm_offload_env.serving import (
    _add_wait_workload,
    cost,
    feasible,
    quality,
    refresh_queue_load,
)
from llm_offload_env.taskgen import Node, generate_task


V12F_VERSION = "V12F-JOINT-LEGACY-GRAPH-QUEUE"
V12F_ACTIONS = V12E_ACTIONS
GLOBAL_FEATURE_DIM = 6
HOTSPOT_CURRICULUM_STRATA = tuple(
    (pattern, intensity)
    for pattern, intensity in LOAD_STRATA
    if pattern != "uniform"
)
REWARD_FORMULA = (
    "-1.00*latency/5 -0.20*cost +2.40*(quality-0.70) "
    "-0.02*edge_queue_variance -1.25*risk_local "
    "-0.10*clip(same_tier_queue_regret/1.0,2.5) "
    "-terminal[1.20*deadline_excess/5]"
)


@dataclass
class V12FConfig(EnvConfig):
    """Legacy-quality joint graph/queue experiment without q_req, SLA, or ECT."""

    load_pattern: str = "mixed"
    load_intensity: str = "mixed"
    task_region_zipf_exponent: float = 1.25
    causal_exogenous_only: bool = True
    max_task_resamples: int = 250
    training_phase: str = "all_strata"
    queue_regret_weight: float = 0.10
    queue_regret_normalizer: float = 1.0
    queue_regret_clip: float = 2.5
                                                                         
                                                                         
    reward_balance_weight: float = 0.02


class V12FJointLegacyGraphQueueEnv(gym.Env):
    """Independent V12F environment.

    It combines V12A's synchronized random node relabeling and legacy linear
    quality reward with the original R2 eight-action multi-server/hotspot queue
    mechanism.  The environment intentionally has no per-node quality target,
    quality-satisfaction metric, SLA metric, or ECT-derived reward.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: V12FConfig | None = None):
        super().__init__()
        self.config = config or V12FConfig()
        if self.config.variant not in {"full", "no_graph", "no_queue"}:
            raise ValueError("V12F supports full/no_graph/no_queue only")
        if not self.config.permute_node_order:
            raise ValueError("V12F requires synchronized random node relabeling")
        if not self.config.causal_exogenous_only:
            raise ValueError("V12F requires action-independent exogenous queues")
        if self.config.load_pattern not in {*LOAD_PATTERNS, "mixed"}:
            raise ValueError("unsupported load pattern")
        if self.config.load_intensity not in {*LOAD_INTENSITIES, "mixed"}:
            raise ValueError("unsupported load intensity")
        if self.config.training_phase not in {"hotspot_curriculum", "all_strata"}:
            raise ValueError("unsupported training phase")
        if self.config.queue_regret_weight < 0.0:
            raise ValueError("queue regret weight must be non-negative")
        if self.config.queue_regret_normalizer <= 0.0:
            raise ValueError("queue regret normalizer must be positive")
        if self.config.queue_regret_clip <= 0.0:
            raise ValueError("queue regret clip must be positive")
        if self.config.max_task_resamples < 1:
            raise ValueError("max_task_resamples must be positive")

        self.action_space = spaces.Discrete(len(V12F_ACTIONS))
        self.node_feature_dim = 12
        self.adjacency_dim = self.config.max_nodes * self.config.max_nodes
        self.profile_feature_dim = PROFILE_FEATURE_DIM_V12E
        self.profile_flat_dim = len(V12F_ACTIONS) * self.profile_feature_dim
        obs_dim = (
            self.config.max_nodes * self.node_feature_dim
            + self.adjacency_dim
            + len(V12F_ACTIONS)
            + self.profile_flat_dim
            + GLOBAL_FEATURE_DIM
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        seed = self.config.seed
        self.task_rng = random.Random(seed)
        self.queue_rng = random.Random(seed + 1_000_003)
        self.observation_rng = random.Random(seed + 2_000_003)
        self.task = None
        self.profiles = multiserver_profiles(self.config.profile_mode)
        self.ready_order: list[int] = []
        self.current_pos = 0
        self.completion: dict[int, float] = {}
        self.realized_quality: dict[int, float] = {}
        self.node_results: list[dict[str, Any]] = []
        self.episode_id = 0
        self.active_episode_index = 0
        self.total_reward = 0.0
        self.node_permutation = list(range(self.config.max_nodes))
        self.task_region = 0
        self.region_probabilities = zipf_region_probabilities(
            self.config.task_region_zipf_exponent
        )
        self.active_pattern = "uniform"
        self.active_intensity = "medium"
        self.hotspot_servers: tuple[str, ...] = ()
        self.queue_trace_seed = seed + 1_000_003
        self.queue_step = 0
        self.task_resample_count = 0
        self.total_rejected_tasks = 0
        self.terminal_deadline_penalty = 0.0

    @property
    def observation_layout(self) -> dict[str, tuple[int, int]]:
        node_end = self.config.max_nodes * self.node_feature_dim
        adjacency_end = node_end + self.adjacency_dim
        queue_end = adjacency_end + len(V12F_ACTIONS)
        profile_end = queue_end + self.profile_flat_dim
        return {
            "nodes": (0, node_end),
            "adjacency": (node_end, adjacency_end),
            "queues": (adjacency_end, queue_end),
            "profiles": (queue_end, profile_end),
            "globals": (profile_end, profile_end + GLOBAL_FEATURE_DIM),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        del options
        if seed is not None:
            self.config.seed = int(seed)
            self.task_rng.seed(seed)
            self.queue_rng.seed(seed + 1_000_003)
            self.observation_rng.seed(seed + 2_000_003)
            self.queue_trace_seed = seed + 1_000_003

        self.active_episode_index = self.episode_id
        self.profiles = multiserver_profiles(self.config.profile_mode)
        self._select_stratum()
        self.task_region = self.task_rng.choices(
            [0, 1], weights=self.region_probabilities, k=1
        )[0]
        self._select_hotspots()
        self.queue_step = 0
        self._initialize_exogenous_queues()

        accepted = None
        rejected = 0
        for attempt in range(self.config.max_task_resamples):
            candidate = generate_task(
                self.task_rng,
                task_id=self.episode_id * self.config.max_task_resamples + attempt,
                min_nodes=self.config.min_nodes,
                max_nodes=self.config.max_nodes,
                deadline_scale=self.config.deadline_scale,
                graph_mode=self.config.task_graph_mode,
            )
            if all(
                any(feasible(node, self.profiles[action]) for action in V12F_ACTIONS)
                for node in candidate.nodes
            ):
                accepted = candidate
                break
            rejected += 1
        if accepted is None:
            raise RuntimeError("V12F could not sample a task with legal actions")

        self.task = accepted
        self.task_resample_count = rejected
        self.total_rejected_tasks += rejected
        self.ready_order = list(range(len(self.task.nodes)))
        self.node_permutation = list(range(self.config.max_nodes))
        self.observation_rng.shuffle(self.node_permutation)
        self.current_pos = 0
        self.completion = {}
        self.realized_quality = {}
        self.node_results = []
        self.total_reward = 0.0
        self.terminal_deadline_penalty = 0.0
        self.episode_id += 1
        return self._obs(), {}

    def _select_stratum(self) -> None:
        if self.config.load_pattern == "mixed" and self.config.load_intensity == "mixed":
            strata = (
                HOTSPOT_CURRICULUM_STRATA
                if self.config.training_phase == "hotspot_curriculum"
                else LOAD_STRATA
            )
            index = (self.config.seed + self.episode_id) % len(strata)
            self.active_pattern, self.active_intensity = strata[index]
            return
        if self.config.load_pattern == "mixed":
            self.active_pattern = LOAD_PATTERNS[
                (self.config.seed + self.episode_id) % len(LOAD_PATTERNS)
            ]
        else:
            self.active_pattern = self.config.load_pattern
        if self.config.load_intensity == "mixed":
            self.active_intensity = LOAD_INTENSITIES[
                (self.config.seed + self.episode_id) % len(LOAD_INTENSITIES)
            ]
        else:
            self.active_intensity = self.config.load_intensity

    def _select_hotspots(self) -> None:
        local_region_edges = [
            action
            for action in V12E_EDGE_ACTIONS
            if SERVER_REGION[action] == self.task_region
        ]
        other_region_edges = [
            action
            for action in V12E_EDGE_ACTIONS
            if SERVER_REGION[action] != self.task_region
        ]
        index = self.config.seed + self.episode_id
        if self.active_pattern in {"single_hotspot", "burst_recovery"}:
            self.hotspot_servers = (local_region_edges[index % len(local_region_edges)],)
        elif self.active_pattern == "double_hotspot":
            self.hotspot_servers = (
                local_region_edges[index % len(local_region_edges)],
                other_region_edges[
                    (index // len(local_region_edges)) % len(other_region_edges)
                ],
            )
        else:
            self.hotspot_servers = ()

    def _load_weight(self, action: str, step: int) -> float:
        if action == "cloud":
            return 0.80
        if self.active_pattern == "uniform":
            return 1.00
        if self.active_pattern == "single_hotspot":
            return 3.00 if action in self.hotspot_servers else 0.55
        if self.active_pattern == "double_hotspot":
            return 2.40 if action in self.hotspot_servers else 0.55
        if self.active_pattern == "burst_recovery":
            if action in self.hotspot_servers:
                if step <= 2:
                    return 3.20
                if step <= 5:
                    return 1.35
                return 0.40
            return 0.65
        raise RuntimeError("unknown load pattern")

    def _initialize_exogenous_queues(self) -> None:
        intensity = INTENSITY_FACTOR[self.active_intensity]
        base_wait = intensity * (0.25 + 1.25 * max(self.config.arrival_rate, 0.0))
        for action in V12E_REMOTE_ACTIONS:
            target_wait = (
                base_wait
                * self._load_weight(action, 0)
                * self.queue_rng.uniform(0.97, 1.03)
            )
            _add_wait_workload(
                self.profiles[action],
                target_wait,
                self.queue_rng.uniform(0.28, 0.44),
            )
            refresh_queue_load(self.profiles[action])

    def _advance_exogenous_queues(self) -> None:
        self.queue_step += 1
        intensity = INTENSITY_FACTOR[self.active_intensity]
        base_increment = intensity * (0.10 + 0.22 * max(self.config.arrival_rate, 0.0))
        for action in V12E_REMOTE_ACTIONS:
            profile = self.profiles[action]
            profile.prefill_backlog *= 0.76
            profile.decode_backlog *= 0.82
            wait_increment = (
                base_increment
                * self._load_weight(action, self.queue_step)
                * self.queue_rng.uniform(0.90, 1.10)
            )
            _add_wait_workload(
                profile,
                wait_increment,
                self.queue_rng.uniform(0.28, 0.44),
            )
            refresh_queue_load(profile)

    def action_masks(self) -> np.ndarray:
        assert self.task is not None
        if self.current_pos >= len(self.task.nodes):
            return np.ones(len(V12F_ACTIONS), dtype=bool)
        node = self.task.nodes[self.ready_order[self.current_pos]]
        mask = np.asarray(
            [feasible(node, self.profiles[action]) for action in V12F_ACTIONS],
            dtype=bool,
        )
        if not mask.any():
            raise RuntimeError("V12F invariant violated: empty legal action set")
        return mask

    def _paired_action_rng(self, node: Node, action_name: str) -> random.Random:
        action_id = V12E_ACTION_TO_ID[action_name]
        mixed_seed = (
            int(self.config.seed) * 1_000_003
            + int(self.active_episode_index) * 100_019
            + int(node.idx) * 1009
            + action_id * 97
            + 17
        )
        return random.Random(mixed_seed)

    def _same_tier_queue_diagnostic(
        self,
        node: Node,
        action_name: str,
        queue_snapshot: dict[str, float],
    ) -> dict[str, Any]:
        eligible_pairs = [
            pair
            for pair in EQUIVALENT_PAIRS
            if all(feasible(node, self.profiles[action]) for action in pair)
        ]
        selected_pair = next(
            (pair for pair in eligible_pairs if action_name in pair),
            None,
        )
        if selected_pair is None:
            return {
                "same_tier_opportunity_count": len(eligible_pairs),
                "same_tier_selected_pair": "",
                "same_tier_queue_regret": 0.0,
                "same_tier_selected_low_queue": 0,
                "same_tier_selection_eligible": 0,
            }
        minimum = min(queue_snapshot[action] for action in selected_pair)
        regret = max(0.0, queue_snapshot[action_name] - minimum)
        return {
            "same_tier_opportunity_count": len(eligible_pairs),
            "same_tier_selected_pair": "|".join(selected_pair),
            "same_tier_queue_regret": regret,
            "same_tier_selected_low_queue": int(regret <= 1e-10),
            "same_tier_selection_eligible": 1,
        }

    def step(self, action: int):
        assert self.task is not None
        node = self.task.nodes[self.ready_order[self.current_pos]]
        mask = self.action_masks()
        invalid = action < 0 or action >= len(mask) or not bool(mask[action])
        if invalid:
            action = int(np.flatnonzero(mask)[0])
        action_name = V12E_ID_TO_ACTION[int(action)]
        profile = self.profiles[action_name]
        queue_snapshot = {
            name: float(candidate.queue_load)
            for name, candidate in self.profiles.items()
        }
        queue_diag = self._same_tier_queue_diagnostic(
            node, action_name, queue_snapshot
        )
        components = sampled_regional_latency_components(
            node,
            profile,
            action_name,
            self.task_region,
            self._paired_action_rng(node, action_name),
            self.config.network_scale,
        )
        step_latency = float(components["total"])
        base_step_quality = quality(node, profile)
        predecessor_quality = (
            float(np.mean([self.realized_quality[p] for p in node.predecessors]))
            if node.predecessors
            else 1.0
        )
        recovery_factor = max(0.0, 1.0 - profile.capability)
        propagation_penalty = (
            self.config.quality_propagation_strength
            * (1.0 - predecessor_quality)
            * node.downstream_risk
            * recovery_factor
        )
        step_quality = min(
            1.0, max(0.0, base_step_quality - propagation_penalty)
        )
        step_cost = cost(node, profile)

        start_time = max(
            [self.completion[p] for p in node.predecessors], default=0.0
        )
        finish_time = start_time + step_latency
        self.completion[node.idx] = finish_time
        self.realized_quality[node.idx] = step_quality
        self._advance_exogenous_queues()

        high_risk_local = int(
            action_name == "local" and node.uncertainty > 0.60
        )
        load_variance = self._load_variance()
        latency_norm = max(self.config.reward_latency_normalizer, 1e-9)
        reward_latency = -self.config.reward_latency_weight * (
            step_latency / latency_norm
        )
        reward_cost = -self.config.reward_cost_weight * step_cost
        reward_quality = self.config.reward_quality_weight * (
            step_quality - 0.70
        )
        reward_balance = -self.config.reward_balance_weight * load_variance
        reward_risk = -1.25 * high_risk_local
        reward_invalid = -self.config.invalid_action_penalty if invalid else 0.0
        normalized_queue_regret = min(
            float(queue_diag["same_tier_queue_regret"])
            / self.config.queue_regret_normalizer,
            self.config.queue_regret_clip,
        )
        reward_queue_regret = (
            -self.config.queue_regret_weight * normalized_queue_regret
        )
        will_terminate = self.current_pos + 1 >= len(self.task.nodes)
        reward_deadline = 0.0
        makespan = max(self.completion.values()) if self.completion else 0.0
        if will_terminate and makespan > self.task.deadline:
            reward_deadline = -self.config.reward_deadline_weight * (
                (makespan - self.task.deadline) / latency_norm
            )
            self.terminal_deadline_penalty = reward_deadline
        reward = sum(
            (
                reward_latency,
                reward_cost,
                reward_quality,
                reward_balance,
                reward_risk,
                reward_invalid,
                reward_queue_regret,
                reward_deadline,
            )
        )

        row: dict[str, Any] = {
            "step": self.current_pos,
            "node_id": node.idx,
            "node_kind": node.kind,
            "action": action_name,
            "feasible_actions": ",".join(
                V12F_ACTIONS[idx] for idx, allowed in enumerate(mask) if allowed
            ),
            "legal_action_count": int(mask.sum()),
            "latency": step_latency,
            "latency_queue": components["queue"],
            "latency_network": components["network"],
            "latency_prefill": components["prefill"],
            "latency_decode": components["decode"],
            "quality": step_quality,
            "base_quality": base_step_quality,
            "predecessor_quality": predecessor_quality,
            "propagation_penalty": propagation_penalty,
            "recovery_factor": recovery_factor,
            "cost": step_cost,
            "uncertainty": node.uncertainty,
            "criticality": node.criticality,
            "downstream_risk": node.downstream_risk,
            "start_time": start_time,
            "finish_time": finish_time,
            "high_risk_local": high_risk_local,
            "chosen_queue_wait": queue_snapshot[action_name],
            "reward_latency_term": reward_latency,
            "reward_cost_term": reward_cost,
            "reward_quality_term": reward_quality,
            "reward_balance_term": reward_balance,
            "reward_risk_term": reward_risk,
            "reward_invalid_term": reward_invalid,
            "reward_queue_regret_term": reward_queue_regret,
            "reward_deadline_term": reward_deadline,
            "reward_total_step": reward,
            "task_region": self.task_region,
            "load_pattern": self.active_pattern,
            "load_intensity": self.active_intensity,
            "hotspot_servers": ",".join(self.hotspot_servers),
            "queue_trace_seed": self.queue_trace_seed,
            **queue_diag,
        }
        for name, value in queue_snapshot.items():
            row[f"queue_{name}"] = value
        self.node_results.append(row)
        self.current_pos += 1
        self.total_reward += reward
        return self._obs(), float(reward), will_terminate, False, {}

    def episode_summary(self, policy: str, episode: int) -> dict[str, Any]:
        assert self.task is not None
        total_latency = max(self.completion.values()) if self.completion else 0.0
        avg_quality = float(np.mean([row["quality"] for row in self.node_results]))
        total_cost = float(sum(row["cost"] for row in self.node_results))
        high_risk_local = float(
            np.mean([row["high_risk_local"] for row in self.node_results])
        )
        offload_ratio = float(
            np.mean([row["action"] != "local" for row in self.node_results])
        )
        deadline_violation = int(total_latency > self.task.deadline)
        legacy_success = int(
            avg_quality >= 0.72
            and high_risk_local <= 0.05
            and not deadline_violation
        )
        queue_values = [row["chosen_queue_wait"] for row in self.node_results]
        selected_pair_rows = [
            row
            for row in self.node_results
            if row["same_tier_selection_eligible"] == 1
        ]
        return {
            "experiment_version": V12F_VERSION,
            "policy": policy,
            "episode": episode,
            "scenario": self.task.scenario,
            "nodes": len(self.task.nodes),
            "edges": sum(len(node.successors) for node in self.task.nodes),
            "max_indegree": max(
                (len(node.predecessors) for node in self.task.nodes), default=0
            ),
            "graph_depth": self._graph_depth(),
            "total_latency": total_latency,
            "deadline": self.task.deadline,
            "deadline_violation": deadline_violation,
            "avg_quality": avg_quality,
            "success": legacy_success,
            "total_cost": total_cost,
            "high_risk_local_ratio": high_risk_local,
            "offload_ratio": offload_ratio,
            "load_variance": self._load_variance(),
            "reward": self.total_reward,
            "mean_chosen_queue_wait": float(np.mean(queue_values)),
            "same_tier_opportunity_nodes": int(
                sum(row["same_tier_opportunity_count"] > 0 for row in self.node_results)
            ),
            "same_tier_selected_nodes": len(selected_pair_rows),
            "mean_same_tier_queue_regret": float(
                np.mean(
                    [row["same_tier_queue_regret"] for row in selected_pair_rows]
                )
            )
            if selected_pair_rows
            else 0.0,
            "same_tier_low_queue_selection_rate": float(
                np.mean(
                    [row["same_tier_selected_low_queue"] for row in selected_pair_rows]
                )
            )
            if selected_pair_rows
            else 0.0,
            "task_region": self.task_region,
            "load_pattern": self.active_pattern,
            "load_intensity": self.active_intensity,
            "hotspot_servers": ",".join(self.hotspot_servers),
            "queue_trace_seed": self.queue_trace_seed,
            "task_resample_count": self.task_resample_count,
            "reward_formula": REWARD_FORMULA,
            "terminal_deadline_penalty": self.terminal_deadline_penalty,
            "static_profile_signature": self.static_profile_signature(),
        }

    def action_records(self, policy: str, episode: int) -> list[dict[str, Any]]:
        return [
            {"policy": policy, "episode": episode, **row}
            for row in self.node_results
        ]

    def _obs(self) -> np.ndarray:
        assert self.task is not None
        current_node = (
            self.ready_order[self.current_pos]
            if self.current_pos < len(self.ready_order)
            else -1
        )
        node_rows: list[list[float]] = []
        for idx in range(self.config.max_nodes):
            if idx < len(self.task.nodes):
                node = self.task.nodes[idx]
                no_graph = self.config.variant == "no_graph"
                row = [
                    float(idx == current_node),
                    float(idx in self.completion),
                    min(node.kind_id() / max(len(NODE_TYPES) - 1, 1), 1.0),
                    min(node.input_tokens / 1200.0, 1.0),
                    min(node.output_tokens / 1300.0, 1.0),
                    node.uncertainty,
                    node.privacy / 3.0,
                    node.risk,
                    0.0 if no_graph else node.criticality,
                    0.0 if no_graph else node.downstream_risk,
                    0.0
                    if no_graph
                    else min(len(node.predecessors) / 4.0, 1.0),
                    self.realized_quality.get(idx, 0.0),
                ]
            else:
                row = [0.0] * self.node_feature_dim
            node_rows.append(row)
        adjacency = np.asarray(
            self._adjacency_features(), dtype=np.float32
        ).reshape(self.config.max_nodes, self.config.max_nodes)
        permutation = np.asarray(self.node_permutation, dtype=int)
        node_rows = [node_rows[idx] for idx in permutation]
        adjacency = adjacency[np.ix_(permutation, permutation)]
        rows = [value for row in node_rows for value in row]
        queues = [
            0.0
            if self.config.variant == "no_queue"
            else min(self.profiles[action].queue_load / 8.0, 1.0)
            for action in V12F_ACTIONS
        ]
        profile_features = [
            value
            for action in V12F_ACTIONS
            for value in _static_profile_features(
                self.profiles[action], action, self.task_region
            )
        ]
        progress = self.current_pos / max(len(self.task.nodes), 1)
        deadline_used = (
            (max(self.completion.values()) if self.completion else 0.0)
            / max(self.task.deadline, 1e-6)
        )
        globals_ = [
            progress,
            min(deadline_used, 1.0),
            min(self.config.arrival_rate / 2.0, 1.0),
            min(self.config.network_scale / 2.0, 1.0),
            len(self.task.nodes) / self.config.max_nodes,
            0.0
            if self.config.variant == "no_queue"
            else self._load_variance_norm(),
        ]
        return np.asarray(
            rows
            + adjacency.reshape(-1).tolist()
            + queues
            + profile_features
            + globals_,
            dtype=np.float32,
        )

    def _adjacency_features(self) -> list[float]:
        assert self.task is not None
        if self.config.variant == "no_graph":
            return [0.0] * self.adjacency_dim
        values = []
        n = len(self.task.nodes)
        for src in range(self.config.max_nodes):
            for dst in range(self.config.max_nodes):
                values.append(
                    1.0
                    if src < n
                    and dst < n
                    and dst in self.task.nodes[src].successors
                    else 0.0
                )
        return values

    def _load_variance(self) -> float:
        loads = [
            self.profiles[action].queue_load for action in V12E_EDGE_ACTIONS
        ]
        mean = sum(loads) / len(loads)
        return sum((value - mean) ** 2 for value in loads) / len(loads)

    def _load_variance_norm(self) -> float:
        return min(self._load_variance() / 8.0, 1.0)

    def _graph_depth(self) -> int:
        assert self.task is not None
        depth = [1] * len(self.task.nodes)
        for node in self.task.nodes:
            if node.predecessors:
                depth[node.idx] = 1 + max(depth[pred] for pred in node.predecessors)
        return max(depth, default=0)

    def static_profile_signature(self) -> str:
        data = {
            action: {
                "prefill_rate": self.profiles[action].prefill_rate,
                "decode_rate": self.profiles[action].decode_rate,
                "base_latency": self.profiles[action].base_latency,
                "quality_base": self.profiles[action].quality_base,
                "token_cost": self.profiles[action].token_cost,
                "privacy_allow": self.profiles[action].privacy_allow,
                "capability": self.profiles[action].capability,
                "region": SERVER_REGION.get(action),
            }
            for action in V12F_ACTIONS
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def close(self) -> None:
        return None
