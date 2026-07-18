from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments_20260715.v12c_queue_fairness import (
    V12CConfig,
    V12CQueueFairnessEnv,
)
from llm_offload_env.config import ID_TO_ACTION
from llm_offload_env.serving import (
    cost,
    expected_latency,
    quality,
    serving_wait,
    update_queue,
)
from llm_offload_env.taskgen import Node


@dataclass
class V12DConfig(V12CConfig):
    """SLA-aligned reward parameters for the isolated V12D experiment."""

    sla_latency_weight: float = 1.00
    sla_downstream_weight: float = 0.20
    sla_shortfall_weight: float = 0.80
    sla_cost_weight: float = 0.20
    sla_balance_weight: float = 0.08
    sla_high_risk_local_weight: float = 1.00
    sla_deadline_weight: float = 1.00
    sla_latency_normalizer: float = 5.0
    sla_shortfall_normalizer: float = 0.10
    sla_latency_clip: float = 3.0
    sla_downstream_clip: float = 2.0
    sla_shortfall_clip: float = 2.0
    sla_deadline_clip: float = 2.0
    sla_load_variance_clip: float = 4.0
    sla_min_node_satisfaction_rate: float = 0.80
    sla_max_mean_shortfall: float = 0.03


REWARD_FORMULA = (
    "r_t=-w_L*clip(delta_makespan/L0)-w_DS*clip(criticality*latency/L0)"
    "-w_Q*clip(max(0,q_req-q)/Q0)-w_C*cost-w_B*clip(load_variance)"
    "-w_R*high_risk_local; terminal adds "
    "-w_D*clip(max(0,makespan-deadline)/L0)"
)


def normalized_shortfall(
    realized_quality: float,
    requirement: float,
    config: V12DConfig,
) -> float:
    shortfall = max(0.0, requirement - realized_quality)
    return min(
        shortfall / max(config.sla_shortfall_normalizer, 1e-9),
        config.sla_shortfall_clip,
    )


def shortfall_reward_term(
    realized_quality: float,
    requirement: float,
    config: V12DConfig,
) -> float:
                                                                           
    return -config.sla_shortfall_weight * normalized_shortfall(
        realized_quality,
        requirement,
        config,
    )


def sampled_latency_components(
    node: Node,
    profile: Any,
    rng: Any,
    network_scale: float,
) -> dict[str, float]:
    """Sample the same latency model as v10 while retaining its components."""

    network = 0.0
    if profile.name.startswith("edge"):
        bandwidth = rng.uniform(8.0, 95.0) * network_scale
        network = (node.input_tokens + 0.22 * node.output_tokens) / max(
            bandwidth * 900.0,
            1.0,
        )
    elif profile.name == "cloud":
        bandwidth = rng.uniform(12.0, 110.0) * network_scale
        network = 0.20 + (
            node.input_tokens + 0.28 * node.output_tokens
        ) / max(bandwidth * 750.0, 1.0)

    queue = serving_wait(profile)
    prefill = profile.base_latency + node.input_tokens / profile.prefill_rate
    decode = node.output_tokens / profile.decode_rate
    jitter = rng.uniform(0.94, 1.10)
    realized = {
        "queue": queue * jitter,
        "network": network * jitter,
        "prefill": prefill * jitter,
        "decode": decode * jitter,
        "jitter": jitter,
    }
    realized["total"] = (
        realized["queue"]
        + realized["network"]
        + realized["prefill"]
        + realized["decode"]
    )
    realized["total"] = max(0.001, realized["total"])
    return realized


class V12DQueueQualitySLAEnv(V12CQueueFairnessEnv):
    """V12C fairness plus a quality-SLA-constrained latency objective."""

    def __init__(self, config: V12DConfig | None = None):
        cfg = config or V12DConfig()
        self._validate_sla_config(cfg)
        super().__init__(cfg)
        self.config: V12DConfig = cfg
        self.terminal_deadline_penalty = 0.0

    @staticmethod
    def _validate_sla_config(config: V12DConfig) -> None:
        nonnegative = [
            config.sla_latency_weight,
            config.sla_downstream_weight,
            config.sla_shortfall_weight,
            config.sla_cost_weight,
            config.sla_balance_weight,
            config.sla_high_risk_local_weight,
            config.sla_deadline_weight,
        ]
        positive = [
            config.sla_latency_normalizer,
            config.sla_shortfall_normalizer,
            config.sla_latency_clip,
            config.sla_downstream_clip,
            config.sla_shortfall_clip,
            config.sla_deadline_clip,
            config.sla_load_variance_clip,
        ]
        if any(value < 0 for value in nonnegative):
            raise ValueError("V12D reward weights must be nonnegative")
        if any(value <= 0 for value in positive):
            raise ValueError("V12D normalizers and clips must be positive")
        if not 0 < config.sla_min_node_satisfaction_rate <= 1:
            raise ValueError("invalid SLA satisfaction-rate threshold")
        if config.sla_max_mean_shortfall < 0:
            raise ValueError("invalid SLA mean-shortfall threshold")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self.terminal_deadline_penalty = 0.0
        return observation, info

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
        components = sampled_latency_components(
            node,
            profile,
            self.serving_rng,
            self.config.network_scale,
        )
        step_latency = components["total"]

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

        makespan_before = max(self.completion.values()) if self.completion else 0.0
        start_time = max(
            [self.completion[pred] for pred in node.predecessors],
            default=0.0,
        )
        finish_time = start_time + step_latency
        self.completion[node.idx] = finish_time
        self.realized_quality[node.idx] = step_quality
        makespan_after = max(self.completion.values())
        critical_path_increment = max(0.0, makespan_after - makespan_before)
        downstream_latency_proxy = node.criticality * step_latency

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
        latency_normalizer = max(self.config.sla_latency_normalizer, 1e-9)
        latency_norm = min(
            critical_path_increment / latency_normalizer,
            self.config.sla_latency_clip,
        )
        downstream_norm = min(
            downstream_latency_proxy / latency_normalizer,
            self.config.sla_downstream_clip,
        )
        quality_shortfall = max(
            0.0,
            self.node_q_req[node.idx] - step_quality,
        )
        shortfall_norm = normalized_shortfall(
            step_quality,
            self.node_q_req[node.idx],
            self.config,
        )
        clipped_load_var = min(
            load_var,
            self.config.sla_load_variance_clip,
        )

        reward_latency_term = -self.config.sla_latency_weight * latency_norm
        reward_downstream_term = (
            -self.config.sla_downstream_weight * downstream_norm
        )
        reward_shortfall_term = shortfall_reward_term(
            step_quality,
            self.node_q_req[node.idx],
            self.config,
        )
        reward_cost_term = -self.config.sla_cost_weight * step_cost
        reward_balance_term = -self.config.sla_balance_weight * clipped_load_var
        reward_risk_term = (
            -self.config.sla_high_risk_local_weight * high_risk_local
        )
        reward_invalid_term = (
            -self.config.invalid_action_penalty if invalid else 0.0
        )

        will_terminate = self.current_pos + 1 >= len(self.task.nodes)
        reward_deadline_term = 0.0
        if will_terminate and makespan_after > self.task.deadline:
            normalized_lateness = min(
                (makespan_after - self.task.deadline) / latency_normalizer,
                self.config.sla_deadline_clip,
            )
            reward_deadline_term = (
                -self.config.sla_deadline_weight * normalized_lateness
            )
            self.terminal_deadline_penalty = reward_deadline_term

        reward = sum(
            [
                reward_latency_term,
                reward_downstream_term,
                reward_shortfall_term,
                reward_cost_term,
                reward_balance_term,
                reward_risk_term,
                reward_invalid_term,
                reward_deadline_term,
            ]
        )

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
                "quality_shortfall": quality_shortfall,
                "normalized_quality_shortfall": shortfall_norm,
                "quality_surplus": max(
                    0.0,
                    step_quality - self.node_q_req[node.idx],
                ),
                "chosen_quality_adequate": int(
                    step_quality >= self.node_q_req[node.idx]
                ),
                "latency": step_latency,
                "latency_queue": components["queue"],
                "latency_network": components["network"],
                "latency_prefill": components["prefill"],
                "latency_decode": components["decode"],
                "latency_jitter": components["jitter"],
                "critical_path_increment": critical_path_increment,
                "downstream_latency_proxy": downstream_latency_proxy,
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
                "makespan_before": makespan_before,
                "makespan_after": makespan_after,
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
                "reward_latency_term": reward_latency_term,
                "reward_downstream_term": reward_downstream_term,
                "reward_shortfall_term": reward_shortfall_term,
                "reward_cost_term": reward_cost_term,
                "reward_balance_term": reward_balance_term,
                "reward_risk_term": reward_risk_term,
                "reward_invalid_term": reward_invalid_term,
                "reward_deadline_term": reward_deadline_term,
                "reward_total_step": reward,
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
        self.total_reward += reward
        return self._obs(), float(reward), will_terminate, False, {}

    def episode_summary(self, policy: str, episode: int) -> dict[str, Any]:
        summary = super().episode_summary(policy, episode)
        satisfaction_rate = float(summary["quality_satisfaction_rate"])
        mean_shortfall = float(summary["mean_quality_shortfall"])
        sla_quality_met = int(
            satisfaction_rate >= self.config.sla_min_node_satisfaction_rate
            and mean_shortfall <= self.config.sla_max_mean_shortfall
        )
        sla_joint_success = int(
            sla_quality_met
            and summary["high_risk_local_ratio"] <= 0.05
            and not summary["deadline_violation"]
        )
        reward_keys = [
            "reward_latency_term",
            "reward_downstream_term",
            "reward_shortfall_term",
            "reward_cost_term",
            "reward_balance_term",
            "reward_risk_term",
            "reward_invalid_term",
            "reward_deadline_term",
        ]
        summary.update(
            {
                "sla_quality_met": sla_quality_met,
                "sla_joint_success": sla_joint_success,
                "terminal_deadline_penalty": self.terminal_deadline_penalty,
                "total_queue_latency": float(
                    sum(row["latency_queue"] for row in self.node_results)
                ),
                "total_network_latency": float(
                    sum(row["latency_network"] for row in self.node_results)
                ),
                "total_prefill_latency": float(
                    sum(row["latency_prefill"] for row in self.node_results)
                ),
                "total_decode_latency": float(
                    sum(row["latency_decode"] for row in self.node_results)
                ),
                "total_critical_path_increment": float(
                    sum(
                        row["critical_path_increment"]
                        for row in self.node_results
                    )
                ),
                "quality_reward_mode": "shortfall_only_saturated_at_q_req",
                "reward_formula": REWARD_FORMULA,
            }
        )
        for key in reward_keys:
            summary[f"sum_{key}"] = float(
                sum(row[key] for row in self.node_results)
            )
            summary[f"mean_abs_{key}"] = float(
                np.mean([abs(row[key]) for row in self.node_results])
            )
        return summary

