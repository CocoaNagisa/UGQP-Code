from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments_20260715.v12e_queue_multiserver_hotspot import (
    V12E_ACTION_TO_ID,
)
from experiments_20260716.v12f_joint_legacy_graph_queue.environment import (
    V12F_ACTIONS,
    V12FJointLegacyGraphQueueEnv,
)
from experiments_20260716.v12l_r9_controlled_core_causal_ablation.environment import (
    V12HConfig,
    V12HRoleSwapHierarchicalQueueEnv,
)
from llm_offload_env.serving import feasible


ABLATION_VARIANTS = ("ppo_mlp", "no_uncertainty_risk_mask")


class CompleteCoreAblationEnv(V12HRoleSwapHierarchicalQueueEnv):
    """R9 physics with two isolated policy-input/encoder interventions.

    ``ppo_mlp`` keeps the complete observation and legal-action mask.  Its
    intervention is applied in the trainer by omitting the graph extractor.

    ``no_uncertainty_risk_mask`` zeros only uncertainty, node-risk and
    downstream-risk observation channels.  Its legal-action mask is computed
    from a temporary node with uncertainty/risk set to zero, while the actual
    simulator, quality, reward and logged outcomes retain the original node.
    This matches the corrected historical definition without changing R9.
    """

    def __init__(self, config: V12HConfig, ablation_variant: str):
        if ablation_variant not in ABLATION_VARIANTS:
            raise ValueError(f"unsupported extension variant: {ablation_variant}")
        if config.variant != "full":
            raise ValueError("extension variants must inherit the R9 Full router")
        self.ablation_variant = ablation_variant
        super().__init__(config)

    def _full_bottom_mask(self) -> np.ndarray:
        return V12FJointLegacyGraphQueueEnv.action_masks(self)

    def _risk_blind_bottom_mask(self) -> np.ndarray:
        assert self.task is not None
        if self.current_pos >= len(self.task.nodes):
            return np.ones(len(V12F_ACTIONS), dtype=bool)
        node = self.task.nodes[self.ready_order[self.current_pos]]
        risk_blind = replace(node, uncertainty=0.0, risk=0.0)
        result = np.asarray(
            [feasible(risk_blind, self.profiles[action]) for action in V12F_ACTIONS],
            dtype=bool,
        )
        if not result.any():
            raise RuntimeError("risk-blind mask unexpectedly has no legal action")
        return result

    def _bottom_action_masks(self) -> np.ndarray:
        if self.ablation_variant == "no_uncertainty_risk_mask":
            return self._risk_blind_bottom_mask()
        return self._full_bottom_mask()

    def _obs(self) -> np.ndarray:
        observation = super()._obs()
        if self.ablation_variant != "no_uncertainty_risk_mask":
            return observation
        node_start, node_end = self.observation_layout["nodes"]
        nodes = observation[node_start:node_end].reshape(
            self.config.max_nodes, self.node_feature_dim
        )
                                                                        
        nodes[:, (5, 7, 9)] = 0.0
        return observation

    def step(self, action: int):
        if self.current_pos < len(self.task.nodes):
            full_mask = self._full_bottom_mask()
            ablated_mask = self._risk_blind_bottom_mask()
        else:
            full_mask = ablated_mask = np.ones(len(V12F_ACTIONS), dtype=bool)
        result = super().step(action)
        row = self.node_results[-1]
        row.update(
            {
                "extension_variant": self.ablation_variant,
                "uncertainty_observation_visible": int(
                    self.ablation_variant != "no_uncertainty_risk_mask"
                ),
                "risk_observation_visible": int(
                    self.ablation_variant != "no_uncertainty_risk_mask"
                ),
                "risk_mask_enabled": int(
                    self.ablation_variant != "no_uncertainty_risk_mask"
                ),
                "full_bottom_mask": "|".join(
                    name for name, ok in zip(V12F_ACTIONS, full_mask) if ok
                ),
                "intervention_bottom_mask": "|".join(
                    name for name, ok in zip(V12F_ACTIONS, ablated_mask) if ok
                ),
                "risk_mask_changed": int(not np.array_equal(full_mask, ablated_mask)),
            }
        )
        return result

    def episode_summary(self, policy: str, episode: int):
        summary = super().episode_summary(policy, episode)
        changed = [int(row.get("risk_mask_changed", 0)) for row in self.node_results]
        summary.update(
            {
                "extension_variant": self.ablation_variant,
                "encoder": (
                    "flat_mlp"
                    if self.ablation_variant == "ppo_mlp"
                    else "directional_residual_graph_encoder"
                ),
                "uncertainty_observation_visible": int(
                    self.ablation_variant != "no_uncertainty_risk_mask"
                ),
                "risk_observation_visible": int(
                    self.ablation_variant != "no_uncertainty_risk_mask"
                ),
                "risk_mask_enabled": int(
                    self.ablation_variant != "no_uncertainty_risk_mask"
                ),
                "risk_mask_changed_node_count": int(sum(changed)),
                "risk_mask_changed_node_ratio": float(np.mean(changed)) if changed else 0.0,
            }
        )
        return summary

