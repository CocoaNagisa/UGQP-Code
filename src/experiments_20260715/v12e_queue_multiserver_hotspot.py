from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from experiments_20260715.v12c_queue_fairness import (
    GLOBAL_FEATURE_DIM,
    PROFILE_FEATURE_NAMES,
    quality_requirement,
)
from experiments_20260715.v12d_queue_quality_sla import (
    REWARD_FORMULA,
    V12DConfig,
    normalized_shortfall,
    shortfall_reward_term,
)
from llm_offload_env.config import NODE_TYPES
from llm_offload_env.serving import (
    _add_wait_workload,
    cost,
    default_profiles,
    feasible,
    quality,
    refresh_queue_load,
    serving_wait,
    update_queue,
)
from llm_offload_env.taskgen import Node, generate_task


V12E_ACTIONS = (
    "local",
    "edge_s_a",
    "edge_s_b",
    "edge_m_a",
    "edge_m_b",
    "edge_l_a",
    "edge_l_b",
    "cloud",
)
V12E_ACTION_TO_ID = {name: idx for idx, name in enumerate(V12E_ACTIONS)}
V12E_ID_TO_ACTION = {idx: name for name, idx in V12E_ACTION_TO_ID.items()}
V12E_REMOTE_ACTIONS = tuple(action for action in V12E_ACTIONS if action != "local")
V12E_EDGE_ACTIONS = tuple(action for action in V12E_ACTIONS if action.startswith("edge"))
EQUIVALENT_PAIRS = (
    ("edge_s_a", "edge_s_b"),
    ("edge_m_a", "edge_m_b"),
    ("edge_l_a", "edge_l_b"),
)
SERVER_REGION = {
    "edge_s_a": 0,
    "edge_s_b": 1,
    "edge_m_a": 0,
    "edge_m_b": 1,
    "edge_l_a": 0,
    "edge_l_b": 1,
}
LOAD_PATTERNS = ("uniform", "single_hotspot", "double_hotspot", "burst_recovery")
LOAD_INTENSITIES = ("low", "medium", "high")
LOAD_STRATA = tuple(
    (pattern, intensity)
    for pattern in LOAD_PATTERNS
    for intensity in LOAD_INTENSITIES
)
INTENSITY_FACTOR = {"low": 0.65, "medium": 1.00, "high": 1.45}
PROFILE_FEATURE_DIM_V12E = len(PROFILE_FEATURE_NAMES) + 2
QUEUE_COLUMNS_V12E = tuple(f"queue_{action}" for action in V12E_ACTIONS)


@dataclass
class V12EConfig(V12DConfig):
    load_pattern: str = "mixed"
    load_intensity: str = "mixed"
    task_region_zipf_exponent: float = 1.25


def zipf_region_probabilities(exponent: float) -> tuple[float, float]:
    if exponent <= 0:
        raise ValueError("Zipf exponent must be positive")
    weights = [1.0, 2.0 ** (-exponent)]
    total = sum(weights)
    return weights[0] / total, weights[1] / total


def _clone_profile(profile: Any, name: str) -> Any:
    cloned = profile.clone()
    cloned.name = name
    return cloned


def multiserver_profiles(profile_mode: str) -> dict[str, Any]:
    base = default_profiles(profile_mode)
    return {
        "local": _clone_profile(base["local"], "local"),
        "edge_s_a": _clone_profile(base["edge_s"], "edge_s_a"),
        "edge_s_b": _clone_profile(base["edge_s"], "edge_s_b"),
        "edge_m_a": _clone_profile(base["edge_m"], "edge_m_a"),
        "edge_m_b": _clone_profile(base["edge_m"], "edge_m_b"),
        "edge_l_a": _clone_profile(base["edge_l"], "edge_l_a"),
        "edge_l_b": _clone_profile(base["edge_l"], "edge_l_b"),
        "cloud": _clone_profile(base["cloud"], "cloud"),
    }


def _static_profile_features(
    profile: Any,
    action: str,
    task_region: int,
) -> list[float]:
    features = [
        min(profile.prefill_rate / 7000.0, 1.0),
        min(profile.decode_rate / 220.0, 1.0),
        min(profile.base_latency / 0.50, 1.0),
        min(profile.quality_base, 1.0),
        min(profile.token_cost, 1.0),
        min(profile.privacy_allow / 3.0, 1.0),
        min(profile.capability, 1.0),
    ]
    if action in SERVER_REGION:
        same_region = SERVER_REGION[action] == task_region
        features.extend([float(same_region), float(not same_region)])
    elif action == "local":
        features.extend([1.0, 0.0])
    else:
        features.extend([0.0, 0.0])
    return features


def sampled_regional_latency_components(
    node: Node,
    profile: Any,
    action: str,
    task_region: int,
    rng: random.Random,
    network_scale: float,
) -> dict[str, float]:
    network = 0.0
    if action.startswith("edge"):
        same_region = SERVER_REGION[action] == task_region
        if same_region:
            bandwidth = rng.uniform(40.0, 120.0) * network_scale
            fixed_rtt = 0.015
        else:
            bandwidth = rng.uniform(12.0, 70.0) * network_scale
            fixed_rtt = 0.080
        network = fixed_rtt + (
            node.input_tokens + 0.22 * node.output_tokens
        ) / max(bandwidth * 900.0, 1.0)
    elif action == "cloud":
        bandwidth = rng.uniform(12.0, 110.0) * network_scale
        network = 0.20 + (
            node.input_tokens + 0.28 * node.output_tokens
        ) / max(bandwidth * 750.0, 1.0)

    jitter = rng.uniform(0.94, 1.10)
    result = {
        "queue": serving_wait(profile) * jitter,
        "network": network * jitter,
        "prefill": (
            profile.base_latency + node.input_tokens / profile.prefill_rate
        )
        * jitter,
        "decode": (node.output_tokens / profile.decode_rate) * jitter,
        "jitter": jitter,
    }
    result["total"] = max(
        0.001,
        result["queue"]
        + result["network"]
        + result["prefill"]
        + result["decode"],
    )
    return result


def expected_regional_latency(
    node: Node,
    profile: Any,
    action: str,
    task_region: int,
    network_scale: float,
) -> float:
    network = 0.0
    if action.startswith("edge"):
        same_region = SERVER_REGION[action] == task_region
        bandwidth = (80.0 if same_region else 41.0) * network_scale
        fixed_rtt = 0.015 if same_region else 0.080
        network = fixed_rtt + (
            node.input_tokens + 0.22 * node.output_tokens
        ) / max(bandwidth * 900.0, 1.0)
    elif action == "cloud":
        bandwidth = 61.0 * network_scale
        network = 0.20 + (
            node.input_tokens + 0.28 * node.output_tokens
        ) / max(bandwidth * 750.0, 1.0)
    return max(
        0.001,
        serving_wait(profile)
        + network
        + profile.base_latency
        + node.input_tokens / profile.prefill_rate
        + node.output_tokens / profile.decode_rate,
    )


class V12EQueueMultiserverHotspotEnv(gym.Env):
    """Eight-action, paired multi-server queue environment for scheme three."""

    metadata = {"render_modes": []}

    def __init__(self, config: V12EConfig | None = None):
        super().__init__()
        self.config = config or V12EConfig()
        if self.config.variant not in {"full", "no_queue"}:
            raise ValueError("V12E supports only full/no_queue")
        if self.config.load_pattern not in {*LOAD_PATTERNS, "mixed"}:
            raise ValueError("unsupported load pattern")
        if self.config.load_intensity not in {*LOAD_INTENSITIES, "mixed"}:
            raise ValueError("unsupported load intensity")
        if self.config.max_task_resamples < 1:
            raise ValueError("max_task_resamples must be positive")
        zipf_region_probabilities(self.config.task_region_zipf_exponent)

        self.action_space = spaces.Discrete(len(V12E_ACTIONS))
        self.node_feature_dim = 13
        self.adjacency_dim = self.config.max_nodes * self.config.max_nodes
        self.profile_feature_dim = PROFILE_FEATURE_DIM_V12E
        self.profile_flat_dim = len(V12E_ACTIONS) * self.profile_feature_dim
        obs_dim = (
            self.config.max_nodes * self.node_feature_dim
            + self.adjacency_dim
            + len(V12E_ACTIONS)
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
        self.serving_rng = random.Random(seed + 3_000_003)
        self.task = None
        self.profiles = multiserver_profiles(self.config.profile_mode)
        self.ready_order: list[int] = []
        self.current_pos = 0
        self.completion: dict[int, float] = {}
        self.realized_quality: dict[int, float] = {}
        self.node_results: list[dict[str, Any]] = []
        self.episode_id = 0
        self.total_reward = 0.0
        self.node_q_req: dict[int, float] = {}
        self.node_q_tier: dict[int, str] = {}
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
        queue_end = adjacency_end + len(V12E_ACTIONS)
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
            self.task_rng.seed(seed)
            self.queue_rng.seed(seed + 1_000_003)
            self.observation_rng.seed(seed + 2_000_003)
            self.serving_rng.seed(seed + 3_000_003)
            self.queue_trace_seed = seed + 1_000_003

        self.profiles = multiserver_profiles(self.config.profile_mode)
        self._select_stratum()
        self.task_region = self.task_rng.choices(
            [0, 1],
            weights=self.region_probabilities,
            k=1,
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
                any(feasible(node, self.profiles[action]) for action in V12E_ACTIONS)
                for node in candidate.nodes
            ):
                accepted = candidate
                break
            rejected += 1
        if accepted is None:
            raise RuntimeError("V12E could not resample a task with legal actions")

        self.task = accepted
        self.task_resample_count = rejected
        self.total_rejected_tasks += rejected
        self.node_q_req = {}
        self.node_q_tier = {}
        for node in self.task.nodes:
            tier, requirement = quality_requirement(node, self.task.scenario)
            self.node_q_tier[node.idx] = tier
            self.node_q_req[node.idx] = requirement

        self.ready_order = list(range(len(self.task.nodes)))
        self.node_permutation = list(range(self.config.max_nodes))
        if self.config.permute_node_order:
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
            index = (self.config.seed + self.episode_id) % len(LOAD_STRATA)
            self.active_pattern, self.active_intensity = LOAD_STRATA[index]
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
            self.hotspot_servers = (
                local_region_edges[index % len(local_region_edges)],
            )
        elif self.active_pattern == "double_hotspot":
            self.hotspot_servers = (
                local_region_edges[index % len(local_region_edges)],
                other_region_edges[(index // len(local_region_edges)) % len(other_region_edges)],
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
        raise RuntimeError("unknown active load pattern")

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
            return np.ones(len(V12E_ACTIONS), dtype=bool)
        node = self.task.nodes[self.ready_order[self.current_pos]]
        mask = np.asarray(
            [feasible(node, self.profiles[action]) for action in V12E_ACTIONS],
            dtype=bool,
        )
        if not mask.any():
            raise RuntimeError("V12E invariant violated: empty legal action set")
        return mask

    def _candidate_stats(self, node: Node) -> dict[str, Any]:
        legal = [
            action for action in V12E_ACTIONS if feasible(node, self.profiles[action])
        ]
        requirement = self.node_q_req[node.idx]
        adequate = [
            action
            for action in legal
            if quality(node, self.profiles[action]) >= requirement
        ]
        qualifying_pairs = [
            pair for pair in EQUIVALENT_PAIRS if all(action in adequate for action in pair)
        ]
        mechanism_actions = sorted(
            {action for pair in qualifying_pairs for action in pair},
            key=V12E_ACTIONS.index,
        )
        latencies = {
            action: expected_regional_latency(
                node,
                self.profiles[action],
                action,
                self.task_region,
                self.config.network_scale,
            )
            for action in mechanism_actions
        }
        queues = {
            action: self.profiles[action].queue_load for action in mechanism_actions
        }
        return {
            "legal": legal,
            "adequate": adequate,
            "qualifying_pairs": qualifying_pairs,
            "mechanism_actions": mechanism_actions,
            "mechanism_eligible": int(bool(qualifying_pairs)),
            "min_expected_latency": min(latencies.values()) if latencies else float("nan"),
            "min_queue": min(queues.values()) if queues else float("nan"),
        }

    def step(self, action: int):
        assert self.task is not None
        node = self.task.nodes[self.ready_order[self.current_pos]]
        mask = self.action_masks()
        invalid = mask[action] == 0
        if invalid:
            action = int(np.flatnonzero(mask)[0])
        action_name = V12E_ID_TO_ACTION[int(action)]
        profile = self.profiles[action_name]
        queue_snapshot = {name: p.queue_load for name, p in self.profiles.items()}
        candidates = self._candidate_stats(node)
        expected_latency_value = expected_regional_latency(
            node,
            profile,
            action_name,
            self.task_region,
            self.config.network_scale,
        )
        components = sampled_regional_latency_components(
            node,
            profile,
            action_name,
            self.task_region,
            self.serving_rng,
            self.config.network_scale,
        )
        step_latency = components["total"]
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
        step_quality = min(1.0, max(0.0, base_step_quality - propagation_penalty))
        step_cost = cost(node, profile)

        makespan_before = max(self.completion.values()) if self.completion else 0.0
        start_time = max([self.completion[p] for p in node.predecessors], default=0.0)
        finish_time = start_time + step_latency
        self.completion[node.idx] = finish_time
        self.realized_quality[node.idx] = step_quality
        makespan_after = max(self.completion.values())
        critical_increment = max(0.0, makespan_after - makespan_before)
        downstream_proxy = node.criticality * step_latency

        if not self.config.causal_exogenous_only:
            update_queue(profile, node, step_latency, self.config.arrival_rate)
        self._advance_exogenous_queues()
        load_var = self._load_variance()
        high_risk_local = int(action_name == "local" and node.uncertainty > 0.60)
        latency_norm = max(self.config.sla_latency_normalizer, 1e-9)
        reward_latency = -self.config.sla_latency_weight * min(
            critical_increment / latency_norm,
            self.config.sla_latency_clip,
        )
        reward_downstream = -self.config.sla_downstream_weight * min(
            downstream_proxy / latency_norm,
            self.config.sla_downstream_clip,
        )
        shortfall = max(0.0, self.node_q_req[node.idx] - step_quality)
        reward_shortfall = shortfall_reward_term(
            step_quality,
            self.node_q_req[node.idx],
            self.config,
        )
        reward_cost = -self.config.sla_cost_weight * step_cost
        reward_balance = -self.config.sla_balance_weight * min(
            load_var,
            self.config.sla_load_variance_clip,
        )
        reward_risk = -self.config.sla_high_risk_local_weight * high_risk_local
        reward_invalid = -self.config.invalid_action_penalty if invalid else 0.0
        will_terminate = self.current_pos + 1 >= len(self.task.nodes)
        reward_deadline = 0.0
        if will_terminate and makespan_after > self.task.deadline:
            reward_deadline = -self.config.sla_deadline_weight * min(
                (makespan_after - self.task.deadline) / latency_norm,
                self.config.sla_deadline_clip,
            )
            self.terminal_deadline_penalty = reward_deadline
        reward = sum(
            (
                reward_latency,
                reward_downstream,
                reward_shortfall,
                reward_cost,
                reward_balance,
                reward_risk,
                reward_invalid,
                reward_deadline,
            )
        )

        min_expected = candidates["min_expected_latency"]
        min_queue = candidates["min_queue"]
        row = {
            "step": self.current_pos,
            "node_id": node.idx,
            "node_kind": node.kind,
            "action": action_name,
            "feasible_actions": ",".join(candidates["legal"]),
            "legal_action_count": len(candidates["legal"]),
            "quality_adequate_actions": ",".join(candidates["adequate"]),
            "adequate_action_count": len(candidates["adequate"]),
            "equivalent_adequate_pairs": ";".join("|".join(pair) for pair in candidates["qualifying_pairs"]),
            "mechanism_candidate_actions": ",".join(candidates["mechanism_actions"]),
            "mechanism_eligible": candidates["mechanism_eligible"],
            "chosen_equivalent_candidate": int(action_name in candidates["mechanism_actions"]),
            "q_req_tier": self.node_q_tier[node.idx],
            "q_req": self.node_q_req[node.idx],
            "quality_shortfall": shortfall,
            "normalized_quality_shortfall": normalized_shortfall(
                step_quality,
                self.node_q_req[node.idx],
                self.config,
            ),
            "quality_surplus": max(0.0, step_quality - self.node_q_req[node.idx]),
            "chosen_quality_adequate": int(step_quality >= self.node_q_req[node.idx]),
            "latency": step_latency,
            "latency_queue": components["queue"],
            "latency_network": components["network"],
            "latency_prefill": components["prefill"],
            "latency_decode": components["decode"],
            "latency_jitter": components["jitter"],
            "critical_path_increment": critical_increment,
            "downstream_latency_proxy": downstream_proxy,
            "expected_latency_before_action": expected_latency_value,
            "min_equivalent_expected_latency": min_expected,
            "expected_latency_regret": (
                expected_latency_value - min_expected if np.isfinite(min_expected) else float("nan")
            ),
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
            "makespan_before": makespan_before,
            "makespan_after": makespan_after,
            "high_risk_local": high_risk_local,
            "chosen_queue_wait": queue_snapshot[action_name],
            "min_equivalent_queue_wait": min_queue,
            "chosen_queue_regret": (
                queue_snapshot[action_name] - min_queue if np.isfinite(min_queue) else float("nan")
            ),
            "reward_latency_term": reward_latency,
            "reward_downstream_term": reward_downstream,
            "reward_shortfall_term": reward_shortfall,
            "reward_cost_term": reward_cost,
            "reward_balance_term": reward_balance,
            "reward_risk_term": reward_risk,
            "reward_invalid_term": reward_invalid,
            "reward_deadline_term": reward_deadline,
            "reward_total_step": reward,
            "task_region": self.task_region,
            "load_pattern": self.active_pattern,
            "load_intensity": self.active_intensity,
            "hotspot_servers": ",".join(self.hotspot_servers),
            "queue_trace_seed": self.queue_trace_seed,
            "causal_exogenous_only": int(self.config.causal_exogenous_only),
            "task_resample_count": self.task_resample_count,
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
        high_risk_local = float(np.mean([row["high_risk_local"] for row in self.node_results]))
        offload_ratio = float(np.mean([row["action"] != "local" for row in self.node_results]))
        deadline_violation = int(total_latency > self.task.deadline)
        legacy_success = int(avg_quality >= 0.72 and high_risk_local <= 0.05 and not deadline_violation)
        satisfaction = [int(row["quality"] >= row["q_req"]) for row in self.node_results]
        shortfalls = [row["quality_shortfall"] for row in self.node_results]
        satisfaction_rate = float(np.mean(satisfaction))
        mean_shortfall = float(np.mean(shortfalls))
        all_quality = int(all(satisfaction))
        sla_quality = int(
            satisfaction_rate >= self.config.sla_min_node_satisfaction_rate
            and mean_shortfall <= self.config.sla_max_mean_shortfall
        )
        sla_joint = int(sla_quality and high_risk_local <= 0.05 and not deadline_violation)
        queue_values = [row["chosen_queue_wait"] for row in self.node_results]
        observed_remote = [
            row[f"queue_{action}"]
            for row in self.node_results
            for action in V12E_REMOTE_ACTIONS
        ]
        eligible = [row["mechanism_eligible"] for row in self.node_results]
        summary = {
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
            "success": legacy_success,
            "total_cost": total_cost,
            "high_risk_local_ratio": high_risk_local,
            "offload_ratio": offload_ratio,
            "load_variance": self._load_variance(),
            "reward": self.total_reward,
            "mean_chosen_queue_wait": float(np.mean(queue_values)),
            "mean_observed_remote_queue": float(np.mean(observed_remote)),
            "max_observed_remote_queue": float(np.max(observed_remote)),
            "quality_satisfaction_rate": satisfaction_rate,
            "mean_quality_shortfall": mean_shortfall,
            "max_quality_shortfall": float(np.max(shortfalls)),
            "all_quality_requirements_met": all_quality,
            "q_req_joint_success": int(all_quality and high_risk_local <= 0.05 and not deadline_violation),
            "sla_quality_met": sla_quality,
            "sla_joint_success": sla_joint,
            "mechanism_eligible_nodes": int(sum(eligible)),
            "mechanism_eligible_node_ratio": float(np.mean(eligible)),
            "task_has_mechanism_choice": int(any(eligible)),
            "task_region": self.task_region,
            "region_probability_0": self.region_probabilities[0],
            "region_probability_1": self.region_probabilities[1],
            "zipf_exponent": self.config.task_region_zipf_exponent,
            "load_pattern": self.active_pattern,
            "load_intensity": self.active_intensity,
            "hotspot_servers": ",".join(self.hotspot_servers),
            "queue_trace_seed": self.queue_trace_seed,
            "causal_exogenous_only": int(self.config.causal_exogenous_only),
            "task_resample_count": self.task_resample_count,
            "total_rejected_tasks": self.total_rejected_tasks,
            "static_profile_signature": self.static_profile_signature(),
            "quality_reward_mode": "shortfall_only_saturated_at_q_req",
            "reward_formula": REWARD_FORMULA,
            "terminal_deadline_penalty": self.terminal_deadline_penalty,
            "total_queue_latency": float(sum(row["latency_queue"] for row in self.node_results)),
            "total_network_latency": float(sum(row["latency_network"] for row in self.node_results)),
            "total_prefill_latency": float(sum(row["latency_prefill"] for row in self.node_results)),
            "total_decode_latency": float(sum(row["latency_decode"] for row in self.node_results)),
            "total_critical_path_increment": float(sum(row["critical_path_increment"] for row in self.node_results)),
        }
        return summary

    def action_records(self, policy: str, episode: int) -> list[dict[str, Any]]:
        return [{"policy": policy, "episode": episode, **row} for row in self.node_results]

    def static_profile_signature(self) -> str:
        data = {
            action: {
                **{name: getattr(self.profiles[action], name) for name in PROFILE_FEATURE_NAMES},
                "region": SERVER_REGION.get(action, None),
            }
            for action in V12E_ACTIONS
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def _obs(self) -> np.ndarray:
        assert self.task is not None
        current_node = self.ready_order[self.current_pos] if self.current_pos < len(self.ready_order) else -1
        node_rows = []
        for idx in range(self.config.max_nodes):
            if idx < len(self.task.nodes):
                node = self.task.nodes[idx]
                row = [
                    float(idx == current_node),
                    float(idx in self.completion),
                    min(node.kind_id() / max(len(NODE_TYPES) - 1, 1), 1.0),
                    min(node.input_tokens / 1200.0, 1.0),
                    min(node.output_tokens / 1300.0, 1.0),
                    node.uncertainty,
                    node.privacy / 3.0,
                    node.risk,
                    node.criticality,
                    node.downstream_risk,
                    min(len(node.predecessors) / 4.0, 1.0),
                    self.realized_quality.get(idx, 0.0),
                    self.node_q_req[idx],
                ]
            else:
                row = [0.0] * self.node_feature_dim
            node_rows.append(row)
        adjacency = np.asarray(self._adjacency_features(), dtype=np.float32).reshape(
            self.config.max_nodes, self.config.max_nodes
        )
        if self.config.permute_node_order:
            permutation = np.asarray(self.node_permutation, dtype=int)
            node_rows = [node_rows[idx] for idx in permutation]
            adjacency = adjacency[np.ix_(permutation, permutation)]
        rows = [value for row in node_rows for value in row]
        queues = [
            0.0 if self.config.variant == "no_queue" else min(self.profiles[action].queue_load / 8.0, 1.0)
            for action in V12E_ACTIONS
        ]
        profile_features = [
            value
            for action in V12E_ACTIONS
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
            0.0 if self.config.variant == "no_queue" else self._load_variance_norm(),
        ]
        return np.asarray(
            rows + adjacency.reshape(-1).tolist() + queues + profile_features + globals_,
            dtype=np.float32,
        )

    def _adjacency_features(self) -> list[float]:
        assert self.task is not None
        values = []
        n = len(self.task.nodes)
        for src in range(self.config.max_nodes):
            for dst in range(self.config.max_nodes):
                values.append(
                    1.0
                    if src < n and dst < n and dst in self.task.nodes[src].successors
                    else 0.0
                )
        return values

    def _load_variance(self) -> float:
        loads = [self.profiles[action].queue_load for action in V12E_EDGE_ACTIONS]
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

    def close(self) -> None:
        return None
