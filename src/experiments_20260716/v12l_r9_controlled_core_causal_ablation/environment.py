from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from experiments_20260715.v12e_queue_multiserver_hotspot import (
    V12E_ACTION_TO_ID,
)
from experiments_20260716.v12f_joint_legacy_graph_queue.environment import (
    V12FJointLegacyGraphQueueEnv,
)
from experiments_20260716.v12g_primary_stable_graph_queue.environment import (
    V12GConfig,
    V12GPrimaryStableGraphQueueEnv,
)
from llm_offload_env.serving import feasible, refresh_queue_load
from llm_offload_env.taskgen import Node

from .graph_pair import (
    MAX_CORRELATION,
    MAX_TOP3_OVERLAP_RATIO,
    MIN_MEAN_ABSOLUTE_DIFFERENCE,
    apply_role_swap_member,
    build_role_swap_pair,
)


V12H_VERSION = "V12H-R9-CONTROLLED-CORE-CAUSAL-ABLATION"
TIER_ACTIONS = ("local", "edge_s", "edge_m", "edge_l", "cloud")
TIER_TO_BOTTOM = {
    "local": ("local",),
    "edge_s": ("edge_s_a", "edge_s_b"),
    "edge_m": ("edge_m_a", "edge_m_b"),
    "edge_l": ("edge_l_a", "edge_l_b"),
    "cloud": ("cloud",),
}
HIGH_CAPABILITY_TIERS = frozenset({"edge_l"})

                                                                             
                                                                          
REWARD_FORMULA = (
    "-1.00*latency/5 "
    "-0.15*cost "
    "+8.00*(min(quality,0.90)-0.70) "
    "-1.25*risk_local "
    "-terminal[3.00*deadline_excess/5] "
    "-terminal[10.00*max(0,0.72-episode_quality)/0.10]"
)

ROLE_SWAP_GATE = {
    "max_exposure_correlation": MAX_CORRELATION,
    "min_exposure_mad": MIN_MEAN_ABSOLUTE_DIFFERENCE,
    "max_top3_overlap_fraction": MAX_TOP3_OVERLAP_RATIO,
}

@dataclass
class V12HConfig(V12GConfig):
    """Configuration for the isolated V12H experiment.

    ``train_aux_queue_credit_weight`` is an optional training-only diagnostic.
    Formal evaluation and the default training protocol keep it at zero.
    """

    queue_regret_weight: float = 0.0
    train_aux_queue_credit_weight: float = 0.0
    train_aux_graph_alignment_weight: float = 0.0
    high_capability_tier_budget: int = 2
    merge_softmin_temperature: float = 0.08
    graph_gate_max_correlation: float = 0.50
    graph_gate_min_mad: float = 0.15
    graph_gate_max_top3_overlap: float = 1.0 / 3.0
    reward_latency_weight: float = 1.00
    reward_cost_weight: float = 0.15
    reward_quality_weight: float = 8.00
    structural_quality_multiplier: float = 0.00
    terminal_quality_floor_weight: float = 10.00
    reward_deadline_weight: float = 3.00
    controlled_core_only: bool = True
    centered_queue_contrast: float = 0.80
    match_medium_large_queue_trajectories: bool = True

class V12HRoleSwapHierarchicalQueueEnv(V12GPrimaryStableGraphQueueEnv):
    """True graph-input tier controller with an isolated queue router.

    Full observes the synchronized node/adjacency representation while
    no-graph masks only those graph channels and is trained independently.
    Both arms have the same hard high-quality quota.  No-queue reuses the
    frozen Full controller and changes only the within-tier replica router.
    """

    config: V12HConfig

    def __init__(self, config: V12HConfig | None = None):
        config = config or V12HConfig(
            permute_node_order=True,
            min_nodes=12,
            max_nodes=12,
            task_graph_mode="dependency_stress",
        )
        if config.high_capability_tier_budget < 1:
            raise ValueError("high-capability tier budget must be positive")
        if config.merge_softmin_temperature <= 0.0:
            raise ValueError("merge soft-min temperature must be positive")
        if not 0.0 <= config.centered_queue_contrast < 1.0:
            raise ValueError("centered queue contrast must be in [0, 1)")
        if not math.isclose(
            config.queue_regret_weight,
            config.train_aux_queue_credit_weight,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "queue credit weight must equal the declared training auxiliary"
            )
        self._routing_bottom_action = False
        self._high_capacity_remaining = config.high_capability_tier_budget
        self._role_mapping_b = tuple(range(12))
        self._role_swap_pair = None
        self._promotion_targets: set[int] = set()
        self._graph_gate: dict[str, Any] = {}
        self._graph_path_multiplicity: dict[int, float] = {}
        super().__init__(config)
        self.action_space = spaces.Discrete(len(TIER_ACTIONS))

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        assert self.task is not None
        self._high_capacity_remaining = self.config.high_capability_tier_budget
        self._prepare_role_swap_pair()
        self._select_promotion_targets()

                                                                          
                                                                           
        permutation_rng = random.Random(
            int(self.config.seed) * 1_000_151
            + int(self.graph_pair_id) * 100_153
            + 53
        )
        self.node_permutation = list(range(self.config.max_nodes))
        permutation_rng.shuffle(self.node_permutation)
        return self._obs(), {}

    def _configure_controlled_graph_profiles(self) -> None:
        """Match edge_m/edge_l execution fields and vary only model quality."""

        for suffix in ("a", "b"):
            medium = self.profiles[f"edge_m_{suffix}"]
            scarce = self.profiles[f"edge_l_{suffix}"]
            medium.prefill_rate *= 1.20
            medium.decode_rate *= 1.20
            medium.base_latency *= 0.85
            medium.quality_base = 0.94
            medium.capability = 0.86
            scarce.prefill_rate = medium.prefill_rate
            scarce.decode_rate = medium.decode_rate
            scarce.base_latency = medium.base_latency
            scarce.token_cost = medium.token_cost
            scarce.privacy_allow = medium.privacy_allow
            scarce.eta_pre = medium.eta_pre
            scarce.eta_dec = medium.eta_dec
            scarce.quality_base = 1.14
            scarce.capability = 0.99

    def _initialize_exogenous_queues(self) -> None:
        self._configure_controlled_graph_profiles()
        super()._initialize_exogenous_queues()
        self._apply_controlled_queue_pairs()

    def _advance_exogenous_queues(self) -> None:
        super()._advance_exogenous_queues()
        self._apply_controlled_queue_pairs()

    def _apply_controlled_queue_pairs(self) -> None:
        """Preserve pair means while exposing live within-tier imbalance."""

        contrast = self.config.centered_queue_contrast
        for left_name, right_name in (
            ("edge_s_a", "edge_s_b"),
            ("edge_m_a", "edge_m_b"),
            ("edge_l_a", "edge_l_b"),
        ):
            left = self.profiles[left_name]
            right = self.profiles[right_name]
            if left.queue_load >= right.queue_load:
                high, low = left, right
            else:
                high, low = right, left
            for field in ("prefill_backlog", "decode_backlog"):
                mean = 0.5 * (
                    float(getattr(high, field)) + float(getattr(low, field))
                )
                setattr(high, field, mean * (1.0 + contrast))
                setattr(low, field, mean * (1.0 - contrast))
            refresh_queue_load(high)
            refresh_queue_load(low)

        if self.config.match_medium_large_queue_trajectories:
            for suffix in ("a", "b"):
                medium = self.profiles[f"edge_m_{suffix}"]
                large = self.profiles[f"edge_l_{suffix}"]
                for field in ("prefill_backlog", "decode_backlog"):
                    mean = 0.5 * (
                        float(getattr(medium, field))
                        + float(getattr(large, field))
                    )
                    setattr(medium, field, mean)
                    setattr(large, field, mean)
                refresh_queue_load(medium)
                refresh_queue_load(large)

    def _prepare_role_swap_pair(self) -> None:
        assert self.task is not None
        nodes = self.task.nodes
        pair = build_role_swap_pair(nodes, strict=True)
        audit = pair.audit
        if not (
            audit.correlation <= self.config.graph_gate_max_correlation
            and audit.mean_absolute_difference >= self.config.graph_gate_min_mad
            and audit.top3_overlap_ratio
            <= self.config.graph_gate_max_top3_overlap
        ):
            raise RuntimeError("role-swap graph failed the frozen pretraining gate")
        self._role_swap_pair = pair
        self._role_mapping_b = pair.role_permutation
        self._graph_gate = {
            "exposure_correlation": audit.correlation,
            "exposure_mad": audit.mean_absolute_difference,
            "top3_overlap_count": audit.top3_overlap_count,
            "top3_overlap_fraction": audit.top3_overlap_ratio,
            "top3_a": list(audit.top3_a),
            "top3_b": list(audit.top3_b),
            "edge_count_equal": audit.edge_count_equal,
            "depth_equal": audit.depth_equal,
            "degree_multiset_equal": audit.degree_multiset_equal,
            "pass": True,
            "thresholds": {
                "max_exposure_correlation": self.config.graph_gate_max_correlation,
                "min_exposure_mad": self.config.graph_gate_min_mad,
                "max_top3_overlap_fraction": (
                    self.config.graph_gate_max_top3_overlap
                ),
            },
        }
        exposure = apply_role_swap_member(
            nodes, pair, self.graph_pair_member
        )
        selected = pair.member(self.graph_pair_member)
        self._graph_path_multiplicity = {
            idx: float(value)
            for idx, value in enumerate(selected.raw_exposure)
        }
        self.ready_order = list(range(len(nodes)))

    def _select_promotion_targets(self) -> None:
        """Define the hidden training target; never force the controller action."""

        assert self.task is not None
        eligible = [
            node
            for node in self.task.nodes
            if all(
                feasible(node, self.profiles[action])
                for action in TIER_TO_BOTTOM["edge_l"]
            )
        ]
        ranked = sorted(
            eligible,
            key=lambda node: (
                self._graph_path_multiplicity.get(node.idx, 0.0),
                -node.idx,
            ),
            reverse=True,
        )
        self._promotion_targets = {
            node.idx
            for node in ranked[: self.config.high_capability_tier_budget]
        }

    def _bottom_action_masks(self) -> np.ndarray:
        return V12FJointLegacyGraphQueueEnv.action_masks(self)

    def action_masks(self) -> np.ndarray:
        if self._routing_bottom_action:
            return self._bottom_action_masks()
        assert self.task is not None
        if self.current_pos >= len(self.task.nodes):
            return np.ones(len(TIER_ACTIONS), dtype=bool)
        bottom = self._bottom_action_masks()
        mask = []
        for tier in TIER_ACTIONS:
            allowed = any(bottom[V12E_ACTION_TO_ID[name]] for name in TIER_TO_BOTTOM[tier])
            if tier == "edge_l":
                allowed = allowed and self._high_capacity_remaining > 0
            if tier == "cloud":
                allowed = False
            mask.append(allowed)
        result = np.asarray(mask, dtype=bool)
        if self.config.controlled_core_only:
            original_result = result.copy()
            medium_allowed = bool(result[TIER_ACTIONS.index("edge_m")])
            large_allowed = bool(result[TIER_ACTIONS.index("edge_l")])
            result[:] = False
            if medium_allowed or large_allowed:
                result[TIER_ACTIONS.index("edge_m")] = medium_allowed
                result[TIER_ACTIONS.index("edge_l")] = large_allowed
            else:
                for fallback in ("local", "edge_s"):
                    fallback_id = TIER_ACTIONS.index(fallback)
                    if original_result[fallback_id]:
                        result[fallback_id] = True
                        break
        current_node = self.task.nodes[self.ready_order[self.current_pos]]
        current_high_feasible = all(
            feasible(current_node, self.profiles[name])
            for name in TIER_TO_BOTTOM["edge_l"]
        )
        remaining_high_feasible = sum(
            all(
                feasible(self.task.nodes[self.ready_order[position]], self.profiles[name])
                for name in TIER_TO_BOTTOM["edge_l"]
            )
            for position in range(self.current_pos, len(self.task.nodes))
        )
                                                                              
                                                                             
                                                                               
                                 
        if (
            current_high_feasible
            and self._high_capacity_remaining > 0
            and remaining_high_feasible <= self._high_capacity_remaining
        ):
            result[:] = False
            result[TIER_ACTIONS.index("edge_l")] = True
        if not result.any():
            raise RuntimeError("V12H invariant violated: empty tier action set")
        return result

    def _balanced_bit(self, tier_id: int, flip_pair_member: bool) -> int:
        value = (
            int(self.config.seed) * 1_000_183
            + int(self.graph_pair_id) * 100_189
            + int(self.current_pos) * 1013
            + int(tier_id) * 101
            + 59
        ) & 1
        if flip_pair_member and self.graph_pair_member == "B":
            value ^= 1
        return int(value)

    def _route_tier(
        self, tier_id: int, node: Node
    ) -> tuple[str, dict[str, Any]]:
        controller_tier = TIER_ACTIONS[tier_id]
        promoted = bool(controller_tier == "edge_l")
        effective_tier = controller_tier
        effective_tier_id = TIER_ACTIONS.index(effective_tier)
        bottom_mask = self._bottom_action_masks()
        candidates = [
            name
            for name in TIER_TO_BOTTOM[effective_tier]
            if bottom_mask[V12E_ACTION_TO_ID[name]]
        ]
        if not candidates:
            raise RuntimeError(
                f"tier {effective_tier} has no feasible bottom action"
            )
        queues = {name: float(self.profiles[name].queue_load) for name in candidates}
        blind_bit = -1
        if len(candidates) == 1:
            selected = candidates[0]
            router_mode = "singleton"
        elif self.config.variant in {"full", "no_graph"}:
            tie_bit = self._balanced_bit(
                effective_tier_id, flip_pair_member=True
            )
            selected = min(
                candidates,
                key=lambda name: (
                    queues[name],
                    int(name.endswith("_b")) != tie_bit,
                ),
            )
            router_mode = "live_queue_argmin"
        else:
            blind_bit = self._balanced_bit(
                effective_tier_id, flip_pair_member=True
            )
            selected = candidates[blind_bit % len(candidates)]
            router_mode = "paired_queue_blind_hash"
        minimum = min(queues.values())
        return selected, {
            "tier_action_id": tier_id,
            "tier_action": controller_tier,
            "effective_tier": effective_tier,
            "graph_quality_upgrade": int(promoted),
            "graph_promotion_target": int(node.idx in self._promotion_targets),
            "routed_action": selected,
            "router_mode": router_mode,
            "router_pair": "|".join(candidates),
            "router_selected_queue": queues[selected],
            "router_pair_min_queue": minimum,
            "router_queue_regret": max(0.0, queues[selected] - minimum),
            "blind_hash_bit": blind_bit,
            "role_position": self.current_pos,
            "physical_node_id": node.idx,
        }

    def _risk_weighted_predecessor(self, node: Node) -> tuple[float, float]:
        if not node.predecessors:
            return 1.0, 1.0
        qualities = np.asarray(
            [self.realized_quality[pred] for pred in node.predecessors],
            dtype=float,
        )
        weights = np.asarray(
            [
                0.25
                + self.task.nodes[pred].risk
                + self.task.nodes[pred].downstream_risk
                for pred in node.predecessors
            ],
            dtype=float,
        )
        weights /= max(float(weights.sum()), 1e-12)
        weighted_mean = float(np.sum(weights * qualities))
        return float(np.mean(qualities)), float(np.clip(weighted_mean, 0.0, 1.0))

    def _graph_alignment_target(self, node: Node) -> bool:
        return node.idx in self._promotion_targets

    def _obs(self) -> np.ndarray:
        observation = super()._obs()
        layout = self.observation_layout
                                                                           
                                                                            
                                                                           
        queue_start, queue_end = layout["queues"]
        observation[queue_start:queue_end] = 0.0
        globals_start, globals_end = layout["globals"]
        progress = observation[globals_start]
                                                                            
                                                                               
        observation[globals_start + 1] = progress
        observation[globals_start + 2] = (
            self._high_capacity_remaining
            / max(self.config.high_capability_tier_budget, 1)
        )
        observation[globals_end - 1] = 0.0
        return observation

    def step(self, action: int):
        assert self.task is not None
        node = self.task.nodes[self.ready_order[self.current_pos]]
        tier_mask = self.action_masks()
        tier_invalid = (
            action < 0 or action >= len(tier_mask) or not bool(tier_mask[action])
        )
        if tier_invalid:
            action = int(np.flatnonzero(tier_mask)[0])
        tier_id = int(action)
        bottom_action, routing = self._route_tier(tier_id, node)
        bottom_id = V12E_ACTION_TO_ID[bottom_action]

        raw_predecessor_mean, predecessor_weighted = self._risk_weighted_predecessor(node)
        saved_predecessors = {
            pred: self.realized_quality[pred] for pred in node.predecessors
        }
        for pred in node.predecessors:
            self.realized_quality[pred] = predecessor_weighted
        self._routing_bottom_action = True
        try:
            _, reward, terminated, truncated, info = super().step(bottom_id)
        finally:
            self._routing_bottom_action = False
            for pred, value in saved_predecessors.items():
                self.realized_quality[pred] = value

        if routing["effective_tier"] in HIGH_CAPABILITY_TIERS:
            self._high_capacity_remaining -= 1
        row = self.node_results[-1]
        row.update(
            {
                **routing,
                "tier_invalid_action": int(tier_invalid),
                "feasible_tiers": ",".join(
                    TIER_ACTIONS[idx]
                    for idx, allowed in enumerate(tier_mask)
                    if allowed
                ),
                "legal_tier_count": int(tier_mask.sum()),
                "high_capacity_budget_remaining": self._high_capacity_remaining,
                "predecessor_quality_mean_raw": raw_predecessor_mean,
                "predecessor_quality_weighted": predecessor_weighted,
                "merge_propagation_mode": "risk_weighted_arithmetic_mean",
                "controlled_core_only": int(self.config.controlled_core_only),
                "centered_queue_contrast": self.config.centered_queue_contrast,
                "matched_medium_large_queue_trajectories": int(
                    self.config.match_medium_large_queue_trajectories
                ),
                "structural_weighted_exposure_raw": self._graph_path_multiplicity.get(
                    node.idx, 0.0
                ),
            }
        )
        target_high = self._graph_alignment_target(node)
        chosen_high = routing["effective_tier"] in HIGH_CAPABILITY_TIERS
        graph_alignment_match = chosen_high == target_high
        if target_high and chosen_high:
            graph_alignment_signal = 1.0
        elif target_high or chosen_high:
            graph_alignment_signal = -1.0
        else:
            graph_alignment_signal = 0.0
        graph_alignment_aux = (
            self.config.train_aux_graph_alignment_weight
            * graph_alignment_signal
        )
        if graph_alignment_aux:
            reward += graph_alignment_aux
            self.total_reward += graph_alignment_aux
            row["reward_total_step"] = float(
                row.get("reward_total_step", 0.0) + graph_alignment_aux
            )
        row.update(
            {
                "graph_alignment_target_high": int(target_high),
                "graph_alignment_chosen_high": int(chosen_high),
                "graph_alignment_match": int(graph_alignment_match),
                "graph_alignment_signal": graph_alignment_signal,
                "reward_graph_alignment_aux_term": graph_alignment_aux,
            }
        )
        return self._obs(), reward, terminated, truncated, info

    def _graph_depth(self) -> int:
        assert self.task is not None
        depth: dict[int, int] = {}
        for node_id in self.ready_order:
            predecessors = self.task.nodes[node_id].predecessors
            depth[node_id] = 1 + max(
                (depth[pred] for pred in predecessors), default=0
            )
        return max(depth.values(), default=0)

    def episode_summary(self, policy: str, episode: int) -> dict[str, Any]:
        summary = super().episode_summary(policy, episode)
        tier_counts = Counter(row["tier_action"] for row in self.node_results)
        effective_tier_counts = Counter(
            row["effective_tier"] for row in self.node_results
        )
        route_regrets = [row["router_queue_regret"] for row in self.node_results]
        summary.update(
            {
                "experiment_version": V12H_VERSION,
                "reward_formula": REWARD_FORMULA,
                "controller_action_space": "|".join(TIER_ACTIONS),
                "controller_queue_visibility": "none",
                "elapsed_queue_proxy_visibility": "none",
                "router_mode": (
                    "paired_queue_blind_hash"
                    if self.config.variant == "no_queue"
                    else "live_queue_argmin"
                ),
                "tier_action_counts": json.dumps(
                    dict(sorted(tier_counts.items())), sort_keys=True
                ),
                "effective_tier_counts": json.dumps(
                    dict(sorted(effective_tier_counts.items())), sort_keys=True
                ),
                "graph_upgrade_mode": (
                    "learned_with_graph_channels"
                    if self.config.variant != "no_graph"
                    else "learned_with_graph_channels_masked"
                ),
                "graph_promotion_targets": "|".join(
                    map(str, sorted(self._promotion_targets))
                ),
                "mean_router_queue_regret": float(np.mean(route_regrets)),
                "high_capacity_tier_budget": (
                    self.config.high_capability_tier_budget
                ),
                "high_capacity_tier_uses": (
                    self.config.high_capability_tier_budget
                    - self._high_capacity_remaining
                ),
                "merge_propagation_mode": "risk_weighted_arithmetic_mean",
                "graph_exposure_correlation_ab": self._graph_gate[
                    "exposure_correlation"
                ],
                "graph_exposure_mad_ab": self._graph_gate["exposure_mad"],
                "graph_top3_overlap_fraction_ab": self._graph_gate[
                    "top3_overlap_fraction"
                ],
                "graph_role_swap_gate_pass": int(self._graph_gate["pass"]),
                "role_mapping_b": "|".join(map(str, self._role_mapping_b)),
                "train_aux_queue_credit_weight": (
                    self.config.train_aux_queue_credit_weight
                ),
                "train_aux_graph_alignment_weight": (
                    self.config.train_aux_graph_alignment_weight
                ),
            }
        )
        return summary


__all__ = [
    "HIGH_CAPABILITY_TIERS",
    "REWARD_FORMULA",
    "ROLE_SWAP_GATE",
    "TIER_ACTIONS",
    "TIER_TO_BOTTOM",
    "V12H_VERSION",
    "V12HConfig",
    "V12HRoleSwapHierarchicalQueueEnv",
]
