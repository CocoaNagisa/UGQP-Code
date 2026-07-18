from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from llm_offload_env.config import ACTIONS, ID_TO_ACTION, NODE_TYPES, EnvConfig
from llm_offload_env.env import LLMAgentOffloadEnv
from llm_offload_env.serving import (
    _add_wait_workload,
    cost,
    default_profiles,
    expected_latency,
    feasible,
    latency,
    quality,
    refresh_queue_load,
    update_queue,
)
from llm_offload_env.taskgen import Node, generate_task


PROFILE_FEATURE_NAMES = (
    "prefill_rate",
    "decode_rate",
    "base_latency",
    "quality_base",
    "token_cost",
    "privacy_allow",
    "capability",
)
PROFILE_FEATURE_DIM = len(PROFILE_FEATURE_NAMES)
GLOBAL_FEATURE_DIM = 6
REMOTE_ACTIONS = tuple(action for action in ACTIONS if action != "local")

                                                                        
                                                                             
                                                                             
                                   
QUEUE_RANK_LEVELS = (1.45, 1.15, 0.85, 0.55)
QUEUE_RANK_PERMUTATIONS = tuple(itertools.permutations(QUEUE_RANK_LEVELS))

Q_REQ_BY_KIND = {
    "intent": ("low", 0.62),
    "extract": ("low", 0.62),
    "retrieve": ("medium", 0.68),
    "tool_call": ("medium", 0.68),
    "summarize": ("medium", 0.68),
    "plan": ("high", 0.74),
    "generate": ("high", 0.74),
    "verify": ("high", 0.74),
}
SCENARIO_Q_REQ_OFFSET = {
    "emergency": 0.02,
    "traffic": 0.00,
    "municipal_service": -0.01,
    "facility_inspection": 0.00,
}


@dataclass
class V12CConfig(EnvConfig):
    """Configuration additions used only by the isolated V12C experiment."""

    queue_assignment_mode: str = "balanced"
    causal_exogenous_only: bool = True
    max_task_resamples: int = 250


def quality_requirement(node: Node, scenario: str) -> tuple[str, float]:
    tier, base = Q_REQ_BY_KIND[node.kind]
    value = base + SCENARIO_Q_REQ_OFFSET[scenario]
    return tier, min(0.80, max(0.58, value))


def _profile_features(profile: Any) -> list[float]:
    """Return bounded static profile descriptors; queue state is excluded."""

    return [
        min(profile.prefill_rate / 7000.0, 1.0),
        min(profile.decode_rate / 220.0, 1.0),
        min(profile.base_latency / 0.50, 1.0),
        min(profile.quality_base, 1.0),
        min(profile.token_cost, 1.0),
        min(profile.privacy_allow / 3.0, 1.0),
        min(profile.capability, 1.0),
    ]


class V12CQueueFairnessEnv(LLMAgentOffloadEnv):
    """Queue-fair V12C environment without changing the v10 core defaults.

    Structural differences between ``full`` and ``no_queue`` are restricted to
    the live per-action queue vector and its derived global load variance. Both
    variants observe the same static server descriptors and task requirements.
    In causal mode, the exogenous queue process is action-independent, allowing
    a paired trace audit even when the two policies choose different actions.
    """

    node_feature_dim = 13

    def __init__(self, config: V12CConfig | None = None):
        cfg = config or V12CConfig()
        if cfg.variant not in {"full", "no_queue"}:
            raise ValueError("V12C queue fairness supports only full/no_queue")
        if cfg.queue_assignment_mode not in {"balanced", "random"}:
            raise ValueError("queue_assignment_mode must be balanced or random")
        if len(REMOTE_ACTIONS) != len(QUEUE_RANK_LEVELS):
            raise ValueError("V12C was specified for four remote actions")
        if cfg.max_task_resamples < 1:
            raise ValueError("max_task_resamples must be positive")

        super().__init__(cfg)
        self.config: V12CConfig = cfg
        self.node_feature_dim = 13
        self.profile_feature_dim = PROFILE_FEATURE_DIM
        self.profile_flat_dim = len(ACTIONS) * self.profile_feature_dim
        self.adjacency_dim = cfg.max_nodes * cfg.max_nodes
        obs_dim = (
            cfg.max_nodes * self.node_feature_dim
            + self.adjacency_dim
            + len(ACTIONS)
            + self.profile_flat_dim
            + GLOBAL_FEATURE_DIM
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

                                                                                
                                                                             
                                                 
        self.task_rng = random.Random(cfg.seed)
        self.serving_rng = random.Random(cfg.seed + 3_000_003)
        self.assignment_rng = random.Random(cfg.seed + 4_000_003)
        self.node_q_req: dict[int, float] = {}
        self.node_q_tier: dict[int, str] = {}
        self.queue_rank_assignment: dict[str, float] = {}
        self.queue_permutation_index = -1
        self.task_resample_count = 0
        self.total_rejected_tasks = 0
        self.queue_trace_seed = cfg.seed + 1_000_003

    @property
    def observation_layout(self) -> dict[str, tuple[int, int]]:
        node_end = self.config.max_nodes * self.node_feature_dim
        adjacency_end = node_end + self.adjacency_dim
        queue_end = adjacency_end + len(ACTIONS)
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
            self.rng.seed(seed)
            self.task_rng.seed(seed)
            self.queue_rng.seed(seed + 1_000_003)
            self.observation_rng.seed(seed + 2_000_003)
            self.serving_rng.seed(seed + 3_000_003)
            self.assignment_rng.seed(seed + 4_000_003)
            self.queue_trace_seed = seed + 1_000_003

        self.profiles = default_profiles(self.config.profile_mode)
        self._assign_queue_ranks()
        self.queue_burst_remaining = self._initialize_exogenous_queues()

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
            if self._task_has_no_empty_action_set(candidate):
                accepted = candidate
                break
            rejected += 1
        if accepted is None:
            raise RuntimeError(
                "Unable to sample a task with non-empty legal actions after "
                f"{self.config.max_task_resamples} attempts"
            )

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
        self.episode_id += 1
        return self._obs(), {}

    def _task_has_no_empty_action_set(self, task: Any) -> bool:
        return all(
            any(feasible(node, self.profiles[action]) for action in ACTIONS)
            for node in task.nodes
        )

    def _assign_queue_ranks(self) -> None:
        if self.config.queue_assignment_mode == "balanced":
            index = (self.config.seed + self.episode_id) % len(
                QUEUE_RANK_PERMUTATIONS
            )
            levels = QUEUE_RANK_PERMUTATIONS[index]
            self.queue_permutation_index = index
        else:
            levels = list(QUEUE_RANK_LEVELS)
            self.assignment_rng.shuffle(levels)
            self.queue_permutation_index = -1
        self.queue_rank_assignment = dict(zip(REMOTE_ACTIONS, levels))

    def _initialize_exogenous_queues(self) -> dict[str, int]:
        burst_remaining = {name: 0 for name in self.profiles}
        base_wait = self.config.queue_scale * (
            0.25 + 1.25 * max(self.config.arrival_rate, 0.0)
        )
        for name in REMOTE_ACTIONS:
            profile = self.profiles[name]
            target_wait = (
                base_wait
                * self.queue_rank_assignment[name]
                * self.queue_rng.uniform(0.97, 1.03)
            )
            _add_wait_workload(
                profile,
                target_wait,
                self.queue_rng.uniform(0.28, 0.44),
            )
            refresh_queue_load(profile)
        return burst_remaining

    def _advance_exogenous_queues(self) -> None:
        if self.config.queue_mode == "legacy":
            for name in REMOTE_ACTIONS:
                profile = self.profiles[name]
                profile.prefill_backlog *= 0.90
                profile.decode_backlog *= 0.92
                refresh_queue_load(profile)
            return
        if self.config.queue_mode != "bursty":
            raise ValueError(f"Unsupported queue_mode: {self.config.queue_mode}")

        base_increment = self.config.queue_scale * (
            0.10 + 0.22 * max(self.config.arrival_rate, 0.0)
        )
        for name in REMOTE_ACTIONS:
            profile = self.profiles[name]
            profile.prefill_backlog *= 0.76
            profile.decode_backlog *= 0.82

            remaining = int(self.queue_burst_remaining.get(name, 0))
            if (
                remaining <= 0
                and self.queue_rng.random() < self.config.queue_burst_probability
            ):
                remaining = self.queue_rng.randint(
                    self.config.queue_burst_min_steps,
                    self.config.queue_burst_max_steps,
                )
            multiplier = (
                self.config.queue_burst_multiplier if remaining > 0 else 1.0
            )
            wait_increment = (
                base_increment
                * self.queue_rank_assignment[name]
                * self.queue_rng.uniform(0.90, 1.10)
                * multiplier
            )
            _add_wait_workload(
                profile,
                wait_increment,
                self.queue_rng.uniform(0.28, 0.44),
            )
            refresh_queue_load(profile)
            self.queue_burst_remaining[name] = max(0, remaining - 1)

    def action_masks(self) -> np.ndarray:
        assert self.task is not None
        if self.current_pos >= len(self.task.nodes):
            return np.ones(len(ACTIONS), dtype=bool)
        node = self.task.nodes[self.ready_order[self.current_pos]]
        mask = np.asarray(
            [feasible(node, self.profiles[action]) for action in ACTIONS],
            dtype=bool,
        )
        if not mask.any():
                                                                               
                                                                               
            raise RuntimeError("V12C invariant violated: empty legal action set")
        return mask

    def _predefined_candidate_stats(self, node: Node) -> dict[str, Any]:
        legal = [
            action for action in ACTIONS if feasible(node, self.profiles[action])
        ]
        requirement = self.node_q_req[node.idx]
        adequate = [
            action
            for action in legal
            if quality(node, self.profiles[action]) >= requirement
        ]
        candidate_latencies = {
            action: expected_latency(
                node,
                self.profiles[action],
                self.config.network_scale,
            )
            for action in adequate
        }
        candidate_queues = {
            action: self.profiles[action].queue_load for action in adequate
        }
        return {
            "legal": legal,
            "adequate": adequate,
            "mechanism_eligible": int(len(adequate) >= 2),
            "min_adequate_expected_latency": (
                min(candidate_latencies.values())
                if candidate_latencies
                else float("nan")
            ),
            "min_adequate_queue_wait": (
                min(candidate_queues.values()) if candidate_queues else float("nan")
            ),
        }

    def step(self, action: int):
        assert self.task is not None
        node = self.task.nodes[self.ready_order[self.current_pos]]
        mask = self.action_masks()
        invalid = mask[action] == 0
        if invalid:
            action = int(np.flatnonzero(mask)[0])

        action_name = ID_TO_ACTION[int(action)]
        profile = self.profiles[action_name]
        queue_snapshot = {name: p.queue_load for name, p in self.profiles.items()}
        candidates = self._predefined_candidate_stats(node)
        chosen_expected_latency = expected_latency(
            node,
            profile,
            self.config.network_scale,
        )

        step_latency = latency(
            node,
            profile,
            self.serving_rng,
            self.config.network_scale,
        )
        base_step_quality = quality(node, profile)
        predecessor_quality = (
            float(
                np.mean(
                    [
                        self.realized_quality[pred]
                        for pred in node.predecessors
                        if pred in self.realized_quality
                    ]
                )
            )
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
            1.0,
            max(0.0, base_step_quality - propagation_penalty),
        )
        step_cost = cost(node, profile)

        start_time = max(
            [self.completion[pred] for pred in node.predecessors],
            default=0.0,
        )
        finish_time = start_time + step_latency
        self.completion[node.idx] = finish_time
        self.realized_quality[node.idx] = step_quality

        if not self.config.causal_exogenous_only:
            update_queue(
                profile,
                node,
                step_latency,
                self.config.arrival_rate,
            )
        self._advance_exogenous_queues()

        high_risk_local = int(
            action_name == "local" and node.uncertainty > 0.60
        )
        load_var = self._load_variance()
        latency_norm = max(self.config.reward_latency_normalizer, 1e-6)
                                                                               
                                                                                
        reward = (
            -self.config.reward_latency_weight * (step_latency / latency_norm)
            -self.config.reward_cost_weight * step_cost
            + self.config.reward_quality_weight * (step_quality - 0.70)
            - self.config.reward_balance_weight * load_var
            - 1.25 * high_risk_local
        )
        if invalid:
            reward -= self.config.invalid_action_penalty

        min_expected = candidates["min_adequate_expected_latency"]
        min_queue = candidates["min_adequate_queue_wait"]
        self.node_results.append(
            {
                "step": self.current_pos,
                "node_id": node.idx,
                "node_kind": node.kind,
                "action": action_name,
                "feasible_actions": ",".join(candidates["legal"]),
                "legal_action_count": len(candidates["legal"]),
                "quality_adequate_actions": ",".join(candidates["adequate"]),
                "adequate_action_count": len(candidates["adequate"]),
                "mechanism_eligible": candidates["mechanism_eligible"],
                "q_req_tier": self.node_q_tier[node.idx],
                "q_req": self.node_q_req[node.idx],
                "chosen_quality_adequate": int(
                    step_quality >= self.node_q_req[node.idx]
                ),
                "latency": step_latency,
                "expected_latency_before_action": chosen_expected_latency,
                "min_adequate_expected_latency": min_expected,
                "expected_latency_regret": (
                    chosen_expected_latency - min_expected
                    if np.isfinite(min_expected)
                    else float("nan")
                ),
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
                "min_adequate_queue_wait": min_queue,
                "chosen_queue_regret": (
                    queue_snapshot[action_name] - min_queue
                    if np.isfinite(min_queue)
                    else float("nan")
                ),
                "queue_local": queue_snapshot["local"],
                "queue_edge_s": queue_snapshot["edge_s"],
                "queue_edge_m": queue_snapshot["edge_m"],
                "queue_edge_l": queue_snapshot["edge_l"],
                "queue_cloud": queue_snapshot["cloud"],
                "queue_assignment": self.queue_assignment_json(),
                "queue_permutation_index": self.queue_permutation_index,
                "queue_trace_seed": self.queue_trace_seed,
                "causal_exogenous_only": int(
                    self.config.causal_exogenous_only
                ),
                "task_resample_count": self.task_resample_count,
            }
        )

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

    def episode_summary(self, policy: str, episode: int) -> dict[str, Any]:
        summary = super().episode_summary(policy, episode)
        satisfactions = [
            int(row["quality"] >= row["q_req"]) for row in self.node_results
        ]
        shortfalls = [
            max(0.0, row["q_req"] - row["quality"])
            for row in self.node_results
        ]
        all_quality_requirements_met = int(all(satisfactions))
        q_req_joint_success = int(
            all_quality_requirements_met
            and summary["high_risk_local_ratio"] <= 0.05
            and not summary["deadline_violation"]
        )
        eligible = [row["mechanism_eligible"] for row in self.node_results]
        summary.update(
            {
                "quality_satisfaction_rate": float(np.mean(satisfactions)),
                "mean_quality_shortfall": float(np.mean(shortfalls)),
                "max_quality_shortfall": float(np.max(shortfalls)),
                "all_quality_requirements_met": all_quality_requirements_met,
                "q_req_joint_success": q_req_joint_success,
                "mechanism_eligible_nodes": int(sum(eligible)),
                "mechanism_eligible_node_ratio": float(np.mean(eligible)),
                "task_has_mechanism_choice": int(any(eligible)),
                "task_resample_count": self.task_resample_count,
                "total_rejected_tasks": self.total_rejected_tasks,
                "queue_assignment": self.queue_assignment_json(),
                "queue_permutation_index": self.queue_permutation_index,
                "queue_trace_seed": self.queue_trace_seed,
                "queue_assignment_mode": self.config.queue_assignment_mode,
                "causal_exogenous_only": int(
                    self.config.causal_exogenous_only
                ),
                "quality_reward_mode": "legacy_linear_q_minus_0.70",
                "static_profile_signature": self.static_profile_signature(),
            }
        )
        return summary

    def queue_assignment_json(self) -> str:
        return json.dumps(
            self.queue_rank_assignment,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def static_profile_signature(self) -> str:
        values = {
            action: {
                name: getattr(self.profiles[action], name)
                for name in PROFILE_FEATURE_NAMES
            }
            for action in ACTIONS
        }
        return json.dumps(
            values,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _obs(self) -> np.ndarray:
        assert self.task is not None
        node_rows: list[list[float]] = []
        current_node_id = (
            self.ready_order[self.current_pos]
            if self.current_pos < len(self.ready_order)
            else -1
        )
        for idx in range(self.config.max_nodes):
            if idx < len(self.task.nodes):
                node = self.task.nodes[idx]
                row = [
                    1.0 if idx == current_node_id else 0.0,
                    1.0 if idx in self.completion else 0.0,
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

        adjacency_matrix = np.asarray(
            self._adjacency_features(),
            dtype=np.float32,
        ).reshape(self.config.max_nodes, self.config.max_nodes)
        if self.config.permute_node_order:
            permutation = np.asarray(self.node_permutation, dtype=int)
            node_rows = [node_rows[idx] for idx in permutation]
            adjacency_matrix = adjacency_matrix[np.ix_(permutation, permutation)]

        rows = [value for row in node_rows for value in row]
        adjacency = adjacency_matrix.reshape(-1).tolist()
        queues = [
            0.0
            if self.config.variant == "no_queue"
            else min(self.profiles[action].queue_load / 8.0, 1.0)
            for action in ACTIONS
        ]
        static_profiles = [
            value
            for action in ACTIONS
            for value in _profile_features(self.profiles[action])
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
            (
                0.0
                if self.config.variant == "no_queue"
                else self._load_variance_norm()
            ),
        ]
        return np.asarray(
            rows + adjacency + queues + static_profiles + globals_,
            dtype=np.float32,
        )

