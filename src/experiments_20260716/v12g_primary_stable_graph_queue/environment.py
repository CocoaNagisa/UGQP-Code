from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments_20260715.v12e_queue_multiserver_hotspot import (
    INTENSITY_FACTOR,
    LOAD_INTENSITIES,
    LOAD_PATTERNS,
    LOAD_STRATA,
    SERVER_REGION,
    V12E_ACTION_TO_ID,
    V12E_EDGE_ACTIONS,
    V12E_ID_TO_ACTION,
    V12E_REMOTE_ACTIONS,
    _static_profile_features,
    sampled_regional_latency_components,
)
from experiments_20260716.v12f_joint_legacy_graph_queue.environment import (
    V12F_ACTIONS,
    V12FConfig,
    V12FJointLegacyGraphQueueEnv,
)
from llm_offload_env.serving import _add_wait_workload, cost, quality, refresh_queue_load
from llm_offload_env.taskgen import add_edge, compute_graph_features


V12G_VERSION = "V12G-PRIMARY-STABLE-GRAPH-QUEUE"
REWARD_FORMULA = (
    "-1.00*(1+1.00*(1-graph_exposure))*latency/5 "
    "-0.20*(1+1.00*(1-graph_exposure))*cost "
    "+3.40*(min(quality,0.72)-0.70)*(1+4.00*graph_exposure) "
    "-1.25*risk_local "
    "+1.20*clip((pair_mean_queue-chosen_queue)/0.25,-2.0,2.0) "
    "-terminal[2.00*deadline_excess/5] "
    "-terminal[3.00*max(0,0.72-episode_quality)/0.10]"
)


@dataclass
class V12GConfig(V12FConfig):
    """Deeper revision after V12F failed the frozen primary-effect gate."""

    queue_amplification: float = 1.00
    queue_observation_normalizer: float = 4.0
    structural_quality_multiplier: float = 4.00
    structural_efficiency_multiplier: float = 1.00
    step_quality_cap: float = 0.72
    terminal_quality_floor: float = 0.72
    terminal_quality_floor_weight: float = 3.00
    terminal_quality_floor_normalizer: float = 0.10
    reward_latency_weight: float = 1.00
    reward_cost_weight: float = 0.20
    reward_quality_weight: float = 3.40
    reward_deadline_weight: float = 2.00
    reward_balance_weight: float = 0.00
    queue_regret_weight: float = 1.20
    queue_regret_normalizer: float = 0.25
    queue_regret_clip: float = 2.00


class V12GPrimaryStableGraphQueueEnv(V12FJointLegacyGraphQueueEnv):
    """V12G with dependency-rich core DAGs and pure same-path queue twins."""

    config: V12GConfig

    def __init__(self, config: V12GConfig | None = None):
        config = config or V12GConfig(permute_node_order=True)
        if config.task_graph_mode != "dependency_stress":
            raise ValueError("V12G requires the 12-node dependency-rich core DAG")
        if config.min_nodes != 12 or config.max_nodes != 12:
            raise ValueError("V12G dependency-rich core requires exactly 12 nodes")
        if config.queue_amplification <= 0.0:
            raise ValueError("queue amplification must be positive")
        if config.queue_observation_normalizer <= 0.0:
            raise ValueError("queue observation normalizer must be positive")
        super().__init__(config)
        self.graph_pair_id = 0
        self.graph_pair_member = "A"
        self.terminal_quality_floor_penalty = 0.0

    def reset(self, *args, **kwargs):
                                                                           
                                                                             
                                                                  
        kwargs.pop("seed", None)
        episode_index = self.episode_id
        self.graph_pair_id = episode_index // 2
        self.graph_pair_member = "A" if episode_index % 2 == 0 else "B"
        task_seed = (
            int(self.config.seed) * 1_000_003
            + self.graph_pair_id * 100_019
            + 31
        )
        queue_seed = (
            int(self.config.seed) * 1_000_033
            + self.graph_pair_id * 100_043
            + 37
        )
        self.task_rng.seed(task_seed)
        self.queue_rng.seed(queue_seed)
        self.observation_rng.seed(
            int(self.config.seed) * 1_000_087 + episode_index * 1009 + 41
        )
        result = super().reset(*args, **kwargs)
        if self.graph_pair_member == "B":
            self._rewire_pair_b()
                                                                           
                                                                               
        self.active_episode_index = self.graph_pair_id
        self.queue_trace_seed = queue_seed
        self.terminal_quality_floor_penalty = 0.0
        return self._obs(), result[1]

    def _select_stratum(self) -> None:
        pair_index = self.episode_id // 2
        if self.config.load_pattern == "mixed" and self.config.load_intensity == "mixed":
            strata = (
                tuple(
                    item for item in LOAD_STRATA if item[0] != "uniform"
                )
                if self.config.training_phase == "hotspot_curriculum"
                else LOAD_STRATA
            )
            self.active_pattern, self.active_intensity = strata[
                (self.config.seed + pair_index) % len(strata)
            ]
            return
        self.active_pattern = (
            LOAD_PATTERNS[(self.config.seed + pair_index) % len(LOAD_PATTERNS)]
            if self.config.load_pattern == "mixed"
            else self.config.load_pattern
        )
        self.active_intensity = (
            LOAD_INTENSITIES[
                (self.config.seed + pair_index) % len(LOAD_INTENSITIES)
            ]
            if self.config.load_intensity == "mixed"
            else self.config.load_intensity
        )

    def _rewire_pair_b(self) -> None:
        assert self.task is not None
        nodes = self.task.nodes
        for node in nodes:
            node.predecessors = []
            node.successors = []
        layers = ((0, 1), (2, 3, 4), (5, 6, 7, 8), (9, 10), (11,))
        for dst in layers[1]:
            add_edge(nodes, 0, dst)
            add_edge(nodes, 1, dst)
        for offset, dst in enumerate(layers[2]):
            add_edge(nodes, layers[1][offset % 3], dst)
            add_edge(nodes, layers[1][(offset + 1) % 3], dst)
        for offset, dst in enumerate(layers[3]):
            for shift in range(3):
                add_edge(nodes, layers[2][(offset + shift) % 4], dst)
        add_edge(nodes, 9, 11)
        add_edge(nodes, 10, 11)
        compute_graph_features(nodes)

    def _select_hotspots(self) -> None:
                                                                              
                                                                              
                                                          
        edges = list(V12E_EDGE_ACTIONS)
        if self.active_pattern in {"single_hotspot", "burst_recovery"}:
            self.hotspot_servers = (self.queue_rng.choice(edges),)
        elif self.active_pattern == "double_hotspot":
            tiers = (("edge_s_a", "edge_s_b"), ("edge_m_a", "edge_m_b"), ("edge_l_a", "edge_l_b"))
            selected_tiers = self.queue_rng.sample(list(tiers), k=2)
            self.hotspot_servers = tuple(
                self.queue_rng.choice(list(pair)) for pair in selected_tiers
            )
        else:
            self.hotspot_servers = ()

    def _paired_action_rng(self, node, action_name: str) -> random.Random:
        tier_id = {
            "local": 0,
            "edge_s_a": 1,
            "edge_s_b": 1,
            "edge_m_a": 2,
            "edge_m_b": 2,
            "edge_l_a": 3,
            "edge_l_b": 3,
            "cloud": 4,
        }[action_name]
        mixed_seed = (
            int(self.config.seed) * 1_000_003
            + int(self.graph_pair_id) * 100_019
            + int(node.idx) * 1009
            + tier_id * 97
            + 17
        )
        return random.Random(mixed_seed)

    def _initialize_exogenous_queues(self) -> None:
        intensity = INTENSITY_FACTOR[self.active_intensity]
        base_wait = (
            self.config.queue_amplification
            * intensity
            * (0.25 + 1.25 * max(self.config.arrival_rate, 0.0))
        )
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
        base_increment = (
            self.config.queue_amplification
            * intensity
            * (0.10 + 0.22 * max(self.config.arrival_rate, 0.0))
        )
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

    def _obs(self) -> np.ndarray:
        observation = super()._obs()
        layout = self.observation_layout
        queue_start, queue_end = layout["queues"]
        if self.config.variant == "no_queue":
            observation[queue_start:queue_end] = 0.0
        else:
            observation[queue_start:queue_end] = np.asarray(
                [
                    min(
                        self.profiles[action].queue_load
                        / self.config.queue_observation_normalizer,
                        1.0,
                    )
                    for action in V12F_ACTIONS
                ],
                dtype=np.float32,
            )

                                                                         
                                                                            
                                   
        profile_start, profile_end = layout["profiles"]
        profile_features = [
            value
            for action in V12F_ACTIONS
            for value in _static_profile_features(
                self.profiles[action],
                action,
                SERVER_REGION[action]
                if action in SERVER_REGION
                else self.task_region,
            )
        ]
        observation[profile_start:profile_end] = np.asarray(
            profile_features, dtype=np.float32
        )
        return observation

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

                                                                              
                                                                               
                                                                                 
        latency_region = (
            SERVER_REGION[action_name]
            if action_name in SERVER_REGION
            else self.task_region
        )
        components = sampled_regional_latency_components(
            node,
            profile,
            action_name,
            latency_region,
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
        graph_exposure = 0.5 * (
            node.criticality + node.downstream_risk
        )
                                                                      
                                                                              
                                                                            
                                                                              
                                                                         
        efficiency_multiplier = (
            1.0
            + self.config.structural_efficiency_multiplier
            * (1.0 - graph_exposure)
        )
        reward_latency = -self.config.reward_latency_weight * (
            step_latency / latency_norm
        ) * efficiency_multiplier
        reward_cost = (
            -self.config.reward_cost_weight
            * step_cost
            * efficiency_multiplier
        )
        structural_multiplier = (
            1.0
            + self.config.structural_quality_multiplier * graph_exposure
        )
        reward_quality = (
            self.config.reward_quality_weight
            * (min(step_quality, self.config.step_quality_cap) - 0.70)
            * structural_multiplier
        )
        reward_balance = -self.config.reward_balance_weight * load_variance
        reward_risk = -1.25 * high_risk_local
        reward_invalid = -self.config.invalid_action_penalty if invalid else 0.0
        pair_text = str(queue_diag["same_tier_selected_pair"])
        centered_queue_advantage = 0.0
        if pair_text:
            pair = pair_text.split("|")
            pair_mean = float(np.mean([queue_snapshot[name] for name in pair]))
            centered_queue_advantage = pair_mean - queue_snapshot[action_name]
        normalized_queue_credit = float(
            np.clip(
                centered_queue_advantage / self.config.queue_regret_normalizer,
                -self.config.queue_regret_clip,
                self.config.queue_regret_clip,
            )
        )
        reward_queue_regret = (
            self.config.queue_regret_weight * normalized_queue_credit
        )

        will_terminate = self.current_pos + 1 >= len(self.task.nodes)
        reward_deadline = 0.0
        reward_quality_floor = 0.0
        makespan = max(self.completion.values()) if self.completion else 0.0
        if will_terminate:
            if makespan > self.task.deadline:
                reward_deadline = -self.config.reward_deadline_weight * (
                    (makespan - self.task.deadline) / latency_norm
                )
            episode_qualities = [
                float(row["quality"]) for row in self.node_results
            ] + [step_quality]
            episode_quality = float(np.mean(episode_qualities))
            reward_quality_floor = (
                -self.config.terminal_quality_floor_weight
                * max(0.0, self.config.terminal_quality_floor - episode_quality)
                / self.config.terminal_quality_floor_normalizer
            )
            self.terminal_deadline_penalty = reward_deadline
            self.terminal_quality_floor_penalty = reward_quality_floor

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
                reward_quality_floor,
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
            "graph_exposure": graph_exposure,
            "structural_efficiency_multiplier": efficiency_multiplier,
            "reward_balance_term": reward_balance,
            "reward_risk_term": reward_risk,
            "reward_invalid_term": reward_invalid,
            "reward_queue_regret_term": reward_queue_regret,
            "centered_same_tier_queue_advantage": centered_queue_advantage,
            "normalized_same_tier_queue_credit": normalized_queue_credit,
            "reward_deadline_term": reward_deadline,
            "reward_terminal_quality_floor_term": reward_quality_floor,
            "reward_total_step": reward,
            "task_region": self.task_region,
            "load_pattern": self.active_pattern,
            "load_intensity": self.active_intensity,
            "hotspot_servers": ",".join(self.hotspot_servers),
            "queue_trace_seed": self.queue_trace_seed,
            "network_path_mode": "same_path_within_tier",
            "graph_pair_id": self.graph_pair_id,
            "graph_pair_member": self.graph_pair_member,
            **queue_diag,
        }
        for name, value in queue_snapshot.items():
            row[f"queue_{name}"] = value
        self.node_results.append(row)
        self.current_pos += 1
        self.total_reward += reward
        return self._obs(), float(reward), will_terminate, False, {}

    def episode_summary(self, policy: str, episode: int) -> dict[str, Any]:
        summary = super().episode_summary(policy, episode)
        summary.update(
            {
                "experiment_version": V12G_VERSION,
                "reward_formula": REWARD_FORMULA,
                "network_path_mode": "same_path_within_tier",
                "queue_amplification": self.config.queue_amplification,
                "queue_observation_normalizer": (
                    self.config.queue_observation_normalizer
                ),
                "graph_pair_id": self.graph_pair_id,
                "graph_pair_member": self.graph_pair_member,
                "terminal_quality_floor_penalty": (
                    self.terminal_quality_floor_penalty
                ),
            }
        )
        return summary
