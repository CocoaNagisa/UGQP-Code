from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_io import ensure_dir, write_csv, write_json              
from experiments_20260715.v12e_queue_multiserver_hotspot import (              
    LOAD_STRATA,
    PROFILE_FEATURE_DIM_V12E,
)
from experiments_20260716.v12f_joint_legacy_graph_queue.environment import (              
    GLOBAL_FEATURE_DIM,
    V12F_ACTIONS,
)

from .aggregate import (              
    HISTORICAL_EFFECTS,
    MODULE_THRESHOLDS,
    PRACTICAL_THRESHOLDS,
    UTILITY_BOUNDS,
    UTILITY_WEIGHTS,
    aggregate_root,
)
from .environment import (              
    REWARD_FORMULA,
    ROLE_SWAP_GATE,
    TIER_ACTIONS,
    V12H_VERSION,
    V12HConfig,
    V12HRoleSwapHierarchicalQueueEnv,
)
from .gnn import DirectionalResidualJKExtractor              


VARIANTS = ("full", "no_graph", "no_queue")
TRAINED_CONTROLLERS = ("full", "no_graph")
PRESETS = {
    "smoke": ([2081], 64, 2, 1),
    "pilot": ([2081], 40_000, 20, 4),
    "formal": ([2082, 2083, 2084], 100_000, 50, 4),
}
FORMAL_ROOT_NAME = "paper_v12h_r9_controlled_core_causal_ablation_20260716"
GRAPH_TRAIN_AUX_WEIGHT = 2.0


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return seeds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_root(root: Path, resume: bool) -> Path:
    resolved = root.resolve()
    lowered = str(resolved).casefold()
    if "retired_experiments" in lowered or "v13" in lowered:
        raise ValueError("V12H output cannot use a retired experiment path")
    if resolved.exists() and any(resolved.iterdir()) and not resume:
        raise FileExistsError(f"non-empty output root: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _build_config(
    seed: int,
    variant: str,
    pattern: str,
    intensity: str,
    phase: str,
    *,
    train_aux_queue_credit_weight: float = 0.0,
    train_aux_graph_alignment_weight: float = 0.0,
) -> V12HConfig:
    return V12HConfig(
        seed=seed,
        variant=variant,
        arrival_rate=0.7,
        network_scale=1.0,
        deadline_scale=0.70,
        profile_mode="tradeoff_v2",
        min_nodes=12,
        max_nodes=12,
        task_graph_mode="dependency_stress",
        quality_propagation_strength=10.00,
        permute_node_order=True,
        causal_exogenous_only=True,
        max_task_resamples=400,
        load_pattern=pattern,
        load_intensity=intensity,
        task_region_zipf_exponent=1.25,
        training_phase=phase,
        queue_amplification=1.00,
        queue_observation_normalizer=4.0,
        structural_quality_multiplier=0.00,
        structural_efficiency_multiplier=0.00,
        step_quality_cap=0.90,
        terminal_quality_floor=0.72,
        terminal_quality_floor_weight=10.00,
        terminal_quality_floor_normalizer=0.10,
        reward_latency_weight=1.00,
        reward_latency_normalizer=5.0,
        reward_cost_weight=0.15,
        reward_quality_weight=8.00,
        reward_deadline_weight=3.00,
        reward_balance_weight=0.00,
        queue_regret_weight=train_aux_queue_credit_weight,
        train_aux_queue_credit_weight=train_aux_queue_credit_weight,
        train_aux_graph_alignment_weight=train_aux_graph_alignment_weight,
        queue_regret_normalizer=0.25,
        queue_regret_clip=2.0,
        high_capability_tier_budget=2,
        merge_softmin_temperature=0.08,
        controlled_core_only=True,
        centered_queue_contrast=0.80,
        match_medium_large_queue_trajectories=True,
    )


def _policy_kwargs(env: V12HRoleSwapHierarchicalQueueEnv) -> dict[str, Any]:
    return {
        "features_extractor_class": DirectionalResidualJKExtractor,
        "features_extractor_kwargs": {
            "max_nodes": env.config.max_nodes,
            "node_feature_dim": env.node_feature_dim,
                                                                               
                                                                                 
            "action_count": len(V12F_ACTIONS),
            "global_feature_dim": (
                GLOBAL_FEATURE_DIM
                + len(V12F_ACTIONS) * PROFILE_FEATURE_DIM_V12E
            ),
            "hidden_dim": 96,
            "message_steps": 4,
            "features_dim": 160,
        },
        "net_arch": {"pi": [160, 96], "vf": [160, 96]},
    }


def _vec_env(
    seed: int,
    variant: str,
    n_envs: int,
    *,
    auxiliary_weight: float,
) -> DummyVecEnv:
    def factory(index: int):
        def make_env() -> V12HRoleSwapHierarchicalQueueEnv:
            return V12HRoleSwapHierarchicalQueueEnv(
                _build_config(
                    seed + 1009 * index,
                    variant,
                    "mixed",
                    "mixed",
                    "all_strata",
                    train_aux_queue_credit_weight=auxiliary_weight,
                    train_aux_graph_alignment_weight=GRAPH_TRAIN_AUX_WEIGHT,
                )
            )

        return make_env

    return DummyVecEnv([factory(index) for index in range(n_envs)])


def _preflight(seed: int) -> dict[str, Any]:
    env = V12HRoleSwapHierarchicalQueueEnv(
        _build_config(seed, "full", "uniform", "medium", "all_strata")
    )
    env.reset()
    permutation_a = list(env.node_permutation)
    edges_a = [tuple(node.successors) for node in env.task.nodes]
    gate_a = dict(env._graph_gate)
    env.reset()
    edges_b = [tuple(node.successors) for node in env.task.nodes]
    result = {
        "seed": seed,
        "gate": gate_a,
        "same_node_permutation": permutation_a == env.node_permutation,
        "edges_changed": edges_a != edges_b,
        "ready_order": list(env.ready_order),
    }
    env.close()
    if not (
        result["gate"]["pass"]
        and result["same_node_permutation"]
        and result["edges_changed"]
        and result["ready_order"] == list(range(12))
    ):
        raise RuntimeError("V12H preflight failed")
    return result


def train(
    output: Path,
    timesteps: int,
    seed: int,
    variant: str,
    n_envs: int,
    *,
    auxiliary_weight: float,
) -> Path:
    if variant not in TRAINED_CONTROLLERS:
        raise ValueError("only Full and no_graph train independent controllers")
    ensure_dir(output)
    env = _vec_env(
        seed, variant, n_envs, auxiliary_weight=auxiliary_weight
    )
    rollout_steps = 32 if timesteps <= 1024 else 256
    batch_size = min(128, rollout_steps * n_envs)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        n_steps=rollout_steps,
        batch_size=batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=2e-4,
        ent_coef=0.02,
        clip_range=0.15,
        target_kl=0.02,
        max_grad_norm=0.5,
        tensorboard_log=str(output / "tb"),
        policy_kwargs=_policy_kwargs(env.envs[0]),
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model_path = output / f"ppo_tier_{variant}.zip"
    model.save(model_path)
    write_json(
        output / "train_config.json",
        {
            "experiment_version": V12H_VERSION,
            "seed": seed,
            "variant": variant,
            "timesteps": timesteps,
            "n_envs": n_envs,
            "controller_action_space": list(TIER_ACTIONS),
            "controller_queue_visibility": "none",
            "train_aux_queue_credit_weight": auxiliary_weight,
            "train_aux_graph_alignment_weight": GRAPH_TRAIN_AUX_WEIGHT,
            "formal_reward_formula": REWARD_FORMULA,
            "config": asdict(
                _build_config(
                    seed,
                    variant,
                    "mixed",
                    "mixed",
                    "all_strata",
                    train_aux_queue_credit_weight=auxiliary_weight,
                    train_aux_graph_alignment_weight=GRAPH_TRAIN_AUX_WEIGHT,
                )
            ),
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
        },
    )
    env.close()
    return model_path


def evaluate(
    model_path: Path,
    output: Path,
    episodes: int,
    variant: str,
    pattern: str,
    intensity: str,
    evaluation_seed: int,
) -> None:
    if episodes % 2:
        raise ValueError("paired role-swap evaluation requires an even episode count")
    ensure_dir(output)
    model = MaskablePPO.load(model_path)
    env = V12HRoleSwapHierarchicalQueueEnv(
        _build_config(evaluation_seed, variant, pattern, intensity, "all_strata")
    )
    episode_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=env.action_masks(),
            )
            observation, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        episode_rows.append(env.episode_summary(variant, episode))
        action_rows.extend(env.action_records(variant, episode))
    env.close()
    write_csv(output / "episode_records.csv", episode_rows)
    write_csv(output / "action_records.csv", action_rows)
    write_json(
        output / "infer_config.json",
        {
            "experiment_version": V12H_VERSION,
            "variant": variant,
            "episodes": episodes,
            "pattern": pattern,
            "intensity": intensity,
            "evaluation_seed": evaluation_seed,
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "formal_queue_credit_weight": 0.0,
        },
    )


def _protocol(
    root: Path,
    name: str,
    seeds: list[int],
    timesteps: int,
    episodes: int,
    n_envs: int,
    auxiliary_weight: float,
) -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    sources = [
        package / name
        for name in (
            "environment.py",
            "graph_pair.py",
            "gnn.py",
            "aggregate.py",
            "run.py",
            "test_invariants.py",
        )
    ]
    protocol: dict[str, Any] = {
        "experiment_version": V12H_VERSION,
        "protocol": name,
        "root": str(root.resolve()),
        "seeds": seeds,
        "variants": list(VARIANTS),
        "trained_controllers": list(TRAINED_CONTROLLERS),
        "no_graph_controller_source": "independently_trained_no_graph",
        "no_queue_controller_source": "full",
        "timesteps_per_trained_seed_variant": timesteps,
        "episodes_per_seed_stratum_variant": episodes,
        "n_envs": n_envs,
        "evaluation_strata": [
            f"{pattern}__{intensity}" for pattern, intensity in LOAD_STRATA
        ],
        "controller_action_space": list(TIER_ACTIONS),
        "controller_queue_visibility": {
            variant: "none" for variant in VARIANTS
        },
        "controller_graph_visibility": {
            "full": "node_graph_features_and_adjacency",
            "no_graph": "none",
            "no_queue": "node_graph_features_and_adjacency",
        },
        "router_rules": {
            "full": "strict argmin of live queue within selected twin tier",
            "no_graph": "strict argmin of live queue within selected twin tier",
            "no_queue": "paired balanced hash independent of live queue",
        },
        "controlled_queue_intervention": (
            "Full and no_queue share the exact Full checkpoint and queue-free "
            "controller observation; only the within-tier replica router differs"
        ),
        "controlled_graph_intervention": (
            "Full and no_graph are independently trained with identical PPO, "
            "tasks, random relabeling, reward, edge_m/edge_l-only masks and a "
            "hard two-use edge_l quota; only graph-derived node fields and "
            "adjacency are masked"
        ),
        "role_swap_graph_gate": ROLE_SWAP_GATE,
        "graph_pair_controls": (
            "A/B share node attributes, deadline, queues, physical execution "
            "order and one random label permutation; edge set and audited "
            "weighted descendant/path-multiplicity exposure change"
        ),
        "structural_scarcity": (
            "exactly two learned edge_l choices per DAG; edge_l is matched to "
            "edge_m in speed, cost, privacy and path, differing only in "
            "quality/capability; cloud is disabled in the controlled core test"
        ),
        "controlled_graph_profiles": {
            "edge_m_quality_base": 0.94,
            "edge_l_quality_base": 1.14,
            "matched_service_rate_multiplier": 1.20,
            "matched_base_latency_multiplier": 0.85,
            "matched_fields": [
                "prefill_rate",
                "decode_rate",
                "base_latency",
                "token_cost",
                "privacy_allow",
                "network_path_class",
            ],
        },
        "controlled_core_action_tiers": ["edge_m", "edge_l"],
        "controlled_core_constraint_fallback": (
            "nodes infeasible on both controlled tiers receive the same "
            "deterministic local-or-edge_s singleton action in every variant"
        ),
        "centered_queue_contrast": 0.80,
        "queue_pair_mean_preserved": True,
        "medium_large_queue_trajectories_matched": True,
        "training_only_graph_alignment": {
            "weight": GRAPH_TRAIN_AUX_WEIGHT,
            "applied_identically_to": list(TRAINED_CONTROLLERS),
            "formal_evaluation_weight": 0.0,
            "reported_reward_includes_term": False,
        },
        "merge_quality_propagation": "risk-weighted arithmetic mean",
        "formal_reward_formula": REWARD_FORMULA,
        "train_aux_queue_credit_weight": auxiliary_weight,
        "train_aux_graph_alignment_weight": GRAPH_TRAIN_AUX_WEIGHT,
        "formal_queue_credit_weight": 0.0,
        "composite_utility": {
            "weights": UTILITY_WEIGHTS,
            "bounds": {key: list(value) for key, value in UTILITY_BOUNDS.items()},
            "Reward_is_not_a_component": True,
        },
        "practical_thresholds": PRACTICAL_THRESHOLDS,
        "module_thresholds": MODULE_THRESHOLDS,
        "historical_effect_references": HISTORICAL_EFFECTS,
        "statistical_unit": "training seed",
        "warning": "smoke and pilot cannot update manuscript claims",
        "source_sha256": {
            path.name: _sha256(path) for path in sources if path.exists()
        },
        "provenance": (
            "isolated successor that imports only the preserved V12G stable "
            "environment lineage plus its V12F/V12E dependencies"
        ),
    }
    canonical = json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode(
        "utf-8"
    )
    protocol["protocol_sha256"] = hashlib.sha256(canonical).hexdigest().upper()
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--seeds", type=_parse_seeds)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--episodes-per-stratum", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--train-aux-queue-credit", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    defaults = PRESETS[args.protocol]
    seeds = args.seeds or list(defaults[0])
    timesteps = args.timesteps if args.timesteps is not None else defaults[1]
    episodes = (
        args.episodes_per_stratum
        if args.episodes_per_stratum is not None
        else defaults[2]
    )
    n_envs = args.n_envs if args.n_envs is not None else defaults[3]
    if timesteps <= 0 or n_envs <= 0 or episodes <= 0 or episodes % 2:
        raise ValueError("timesteps/n_envs must be positive; episodes must be positive/even")
    if args.train_aux_queue_credit < 0.0:
        raise ValueError("training auxiliary queue credit must be non-negative")
    if args.protocol == "formal" and args.train_aux_queue_credit != 0.0:
        raise ValueError("formal protocol forbids training queue credit")

    chosen = args.root or Path("outputs") / (
        FORMAL_ROOT_NAME
        if args.protocol == "formal"
        else f"_{args.protocol}_v12h_role_swap_hierarchical_queue_20260716"
    )
    root = _validate_root(chosen, args.resume)
    protocol = _protocol(
        root,
        args.protocol,
        seeds,
        timesteps,
        episodes,
        n_envs,
        args.train_aux_queue_credit,
    )
    protocol_path = root / "experiment_protocol.json"
    if args.resume:
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        immutable = (
            "experiment_version",
            "protocol",
            "seeds",
            "variants",
            "trained_controllers",
            "no_graph_controller_source",
            "no_queue_controller_source",
            "timesteps_per_trained_seed_variant",
            "episodes_per_seed_stratum_variant",
            "n_envs",
            "evaluation_strata",
            "role_swap_graph_gate",
            "formal_reward_formula",
            "train_aux_queue_credit_weight",
            "train_aux_graph_alignment_weight",
            "source_sha256",
        )
        mismatch = [key for key in immutable if existing.get(key) != protocol.get(key)]
        if mismatch:
            raise ValueError("resume protocol mismatch: " + ", ".join(mismatch))
    else:
        preflight = [_preflight(seed) for seed in seeds]
        protocol["preflight"] = preflight
        write_json(protocol_path, protocol)
    if args.initialize_only:
        print(f"INITIALIZED {protocol_path}", flush=True)
        return

    run_seeds = seeds
    if args.worker_seed is not None:
        if args.worker_seed not in seeds:
            raise ValueError("worker seed not in frozen protocol")
        run_seeds = [args.worker_seed]
    for seed in run_seeds:
        models: dict[str, Path] = {}
        for variant in TRAINED_CONTROLLERS:
            model_dir = root / f"seed_{seed}" / "models" / variant
            model_path = model_dir / f"ppo_tier_{variant}.zip"
            if not (args.resume and model_path.exists()):
                print(
                    f"TRAIN {V12H_VERSION} seed={seed} controller={variant} "
                    f"steps={timesteps}",
                    flush=True,
                )
                model_path = train(
                    model_dir,
                    timesteps,
                    seed,
                    variant,
                    n_envs,
                    auxiliary_weight=args.train_aux_queue_credit,
                )
            models[variant] = model_path
        models["no_queue"] = models["full"]
        for shared_variant, reason in (
            ("no_queue", "controlled within-tier queue-router intervention"),
        ):
            shared_dir = root / f"seed_{seed}" / "models" / shared_variant
            ensure_dir(shared_dir)
            write_json(
                shared_dir / "shared_controller.json",
                {
                    "experiment_version": V12H_VERSION,
                    "source_variant": "full",
                    "source_model": str(models["full"].resolve()),
                    "source_sha256": _sha256(models["full"]),
                    "reason": reason,
                },
            )
        for pattern, intensity in LOAD_STRATA:
            stratum = f"{pattern}__{intensity}"
            for variant in VARIANTS:
                output = root / f"seed_{seed}" / "eval" / stratum / variant
                if args.resume and (output / "episode_records.csv").exists():
                    continue
                print(
                    f"EVAL {V12H_VERSION} seed={seed} variant={variant} "
                    f"stratum={stratum}",
                    flush=True,
                )
                evaluate(
                    models[variant],
                    output,
                    episodes,
                    variant,
                    pattern,
                    intensity,
                    50_000 + seed,
                )
    if not args.skip_aggregate:
        print(json.dumps(aggregate_root(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
