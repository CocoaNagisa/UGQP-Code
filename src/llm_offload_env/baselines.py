from __future__ import annotations

import random

from .config import ACTIONS, ACTION_TO_ID
from .env import LLMAgentOffloadEnv
from .serving import cost, expected_latency, latency, quality, refresh_queue_load


BASELINES = ["local_only", "edge_only", "random_feasible", "goa_dag", "goa_style", "queue_heuristic", "no_uncertainty_heuristic", "no_queue_heuristic", "no_graph_heuristic"]


def select_action(env: LLMAgentOffloadEnv, policy: str, rng: random.Random) -> int:
                                                         
    assert env.task is not None
                                    
    node = env.task.nodes[env.ready_order[env.current_pos]]
                                       
    mask = env.action_masks()
                                               
    feasible_ids = [idx for idx, ok in enumerate(mask) if ok]
    if policy == "local_only":
                                                
        return ACTION_TO_ID["local"] if ACTION_TO_ID["local"] in feasible_ids else feasible_ids[0]
    if policy == "edge_only":
                                       
        edge_ids = [ACTION_TO_ID[a] for a in ["edge_s", "edge_m", "edge_l"] if ACTION_TO_ID[a] in feasible_ids]
        return min(edge_ids or feasible_ids, key=lambda idx: env.profiles[ACTIONS[idx]].queue_load)
    if policy == "random_feasible":
                                             
        return rng.choice(feasible_ids)
    if policy == "goa_dag":
        return select_goa_dag_action(env, feasible_ids, uncertainty_threshold=0.60)
    if policy == "goa_style":
                                                
        if node.uncertainty < 0.42 and ACTION_TO_ID["local"] in feasible_ids:
            return ACTION_TO_ID["local"]
                            
        candidates = [idx for idx in feasible_ids if ACTIONS[idx] != "local"]
        return min(candidates or feasible_ids, key=lambda idx: env.profiles[ACTIONS[idx]].queue_load + env.profiles[ACTIONS[idx]].base_latency)
                           
    return score_policy(env, feasible_ids, policy, rng)


def select_goa_dag_action(
    env: LLMAgentOffloadEnv,
    feasible_ids: list[int],
    uncertainty_threshold: float = 0.60,
) -> int:
    """Adapt token-level GOA to sequential location selection on a DAG.

    High-uncertainty nodes must use a feasible remote location. Low-uncertainty
    nodes remain local unless the minimum uncertainty-weighted remote delay gap
    is negative. Live queue state is included through ``expected_latency`` and
    therefore the gaps are recomputed after every scheduling decision.
    """
    assert env.task is not None
    node = env.task.nodes[env.ready_order[env.current_pos]]
    local_id = ACTION_TO_ID["local"]
    remote_ids = [idx for idx in feasible_ids if idx != local_id]

    if not remote_ids:
        return local_id if local_id in feasible_ids else feasible_ids[0]

    local_latency = expected_latency(
        node,
        env.profiles["local"],
        env.config.network_scale,
    )

    def weighted_delay_gap(action_id: int) -> float:
        remote_latency = expected_latency(
            node,
            env.profiles[ACTIONS[action_id]],
            env.config.network_scale,
        )
        return node.token_uncertainty * (remote_latency - local_latency)

    best_remote = min(remote_ids, key=weighted_delay_gap)
    if node.token_uncertainty > uncertainty_threshold:
        return best_remote
    if local_id not in feasible_ids or weighted_delay_gap(best_remote) < 0.0:
        return best_remote
    return local_id


def score_policy(env: LLMAgentOffloadEnv, feasible_ids: list[int], policy: str, rng: random.Random) -> int:
                                              
    assert env.task is not None
    node = env.task.nodes[env.ready_order[env.current_pos]]
    best_id = feasible_ids[0]
    best_score = float("inf")
    for idx in feasible_ids:
        action = ACTIONS[idx]
        profile = env.profiles[action]
                                            
        score_profile = profile
        if policy == "no_queue_heuristic":
            score_profile = profile.clone()
            score_profile.prefill_backlog = 0.0
            score_profile.decode_backlog = 0.0
            refresh_queue_load(score_profile)
        l = latency(node, score_profile, rng, env.config.network_scale)
        q = quality(node, profile)
        c = cost(node, profile)
                                                  
        uncertainty = 0.0 if policy == "no_uncertainty_heuristic" else node.uncertainty
        criticality = 0.0 if policy == "no_graph_heuristic" else node.criticality
        queue = 0.0 if policy == "no_queue_heuristic" else profile.queue_load
                                       
        local_risk = uncertainty * criticality if action == "local" else 0.0
                                 
        score = 1.0 * l + 0.22 * c + 0.35 * queue + 1.35 * local_risk + 0.50 * uncertainty * (1 - q) + 0.20 * criticality * l - 0.75 * q
        if score < best_score:
            best_score = score
            best_id = idx
    return best_id
