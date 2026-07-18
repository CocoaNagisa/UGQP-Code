from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from common_io import ensure_dir, write_csv, write_json
from experiments_20260715.v12e_queue_multiserver_hotspot import LOAD_STRATA
from experiments_20260716.v12l_r9_controlled_core_causal_ablation.environment import (
    REWARD_FORMULA,
    TIER_ACTIONS,
)
from experiments_20260716.v12l_r9_controlled_core_causal_ablation.run import (
    GRAPH_TRAIN_AUX_WEIGHT,
    _build_config,
    _policy_kwargs,
)

from .aggregate import aggregate_root
from .environment import ABLATION_VARIANTS, CompleteCoreAblationEnv


PRESETS = {
    "smoke": ([2081], 256, 2, 1),
    "formal": ([2082, 2083, 2084], 100_000, 50, 4),
}
FULL_REFERENCE_ROOT = Path(
    "outputs/paper_v12h_r9_controlled_core_causal_ablation_20260716"
)
FORMAL_ROOT = Path("outputs/paper_complete_core_ablation_extension_20260717")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _parse_seeds(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return result


def _config(seed: int, variant: str, pattern: str, intensity: str, phase: str, *, training: bool):
                                                                             
    return _build_config(
        seed,
        "full",
        pattern,
        intensity,
        phase,
        train_aux_graph_alignment_weight=(GRAPH_TRAIN_AUX_WEIGHT if training else 0.0),
    )


def _vec_env(seed: int, variant: str, n_envs: int) -> DummyVecEnv:
    def factory(index: int):
        def make_env():
            return CompleteCoreAblationEnv(
                _config(seed + 1009 * index, variant, "mixed", "mixed", "all_strata", training=True),
                variant,
            )
        return make_env
    return DummyVecEnv([factory(i) for i in range(n_envs)])


def _policy(variant: str, env: CompleteCoreAblationEnv) -> dict[str, Any]:
    if variant == "ppo_mlp":
        return {"net_arch": {"pi": [160, 96], "vf": [160, 96]}}
    return _policy_kwargs(env)


def train(output: Path, timesteps: int, seed: int, variant: str, n_envs: int) -> Path:
    ensure_dir(output)
    env = _vec_env(seed, variant, n_envs)
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
        policy_kwargs=_policy(variant, env.envs[0]),
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model_path = output / f"ppo_{variant}.zip"
    model.save(model_path)
    write_json(
        output / "train_config.json",
        {
            "seed": seed,
            "variant": variant,
            "base_environment_variant": "full",
            "timesteps": timesteps,
            "n_envs": n_envs,
            "encoder": "flat_mlp" if variant == "ppo_mlp" else "directional_residual_graph_encoder",
            "policy_heads": {"pi": [160, 96], "vf": [160, 96]},
            "train_aux_graph_alignment_weight": GRAPH_TRAIN_AUX_WEIGHT,
            "formal_reward_formula": REWARD_FORMULA,
            "config": asdict(_config(seed, variant, "mixed", "mixed", "all_strata", training=True)),
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
        },
    )
    env.close()
    return model_path


def evaluate(model_path: Path, output: Path, episodes: int, variant: str, pattern: str, intensity: str, evaluation_seed: int):
    if episodes % 2:
        raise ValueError("paired role-swap evaluation requires even episodes")
    ensure_dir(output)
    model = MaskablePPO.load(model_path)
    env = CompleteCoreAblationEnv(
        _config(evaluation_seed, variant, pattern, intensity, "all_strata", training=False), variant
    )
    episodes_out, actions_out = [], []
    for episode in range(episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
            observation, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        episodes_out.append(env.episode_summary(variant, episode))
        actions_out.extend(env.action_records(variant, episode))
    env.close()
    write_csv(output / "episode_records.csv", episodes_out)
    write_csv(output / "action_records.csv", actions_out)
    write_json(
        output / "infer_config.json",
        {
            "variant": variant,
            "base_environment_variant": "full",
            "episodes": episodes,
            "pattern": pattern,
            "intensity": intensity,
            "evaluation_seed": evaluation_seed,
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
        },
    )


def _protocol(root: Path, protocol: str, seeds: list[int], timesteps: int, episodes: int, n_envs: int) -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    sources = [package / name for name in ("environment.py", "run.py", "aggregate.py", "test_invariants.py")]
    reference = (Path.cwd() / FULL_REFERENCE_ROOT).resolve()
    if not (reference / "experiment_protocol.json").exists():
        raise FileNotFoundError(f"R9 Full reference not found: {reference}")
    result = {
        "protocol": protocol,
        "root": str(root.resolve()),
        "seeds": seeds,
        "variants": list(ABLATION_VARIANTS),
        "timesteps_per_seed_variant": timesteps,
        "episodes_per_seed_stratum_variant": episodes,
        "n_envs": n_envs,
        "evaluation_strata": [f"{p}__{i}" for p, i in LOAD_STRATA],
        "full_reference_root": str(reference),
        "full_reference_protocol_sha256": _sha256(reference / "experiment_protocol.json"),
        "interventions": {
            "ppo_mlp": "same complete observation and action mask; flat MLP replaces graph message passing extractor",
            "no_uncertainty_risk_mask": "zero uncertainty/node-risk/downstream-risk observations and compute mask with uncertainty/risk zero; simulator outcomes unchanged",
        },
        "base_environment_variant": "full",
        "controller_action_space": list(TIER_ACTIONS),
        "formal_reward_formula": REWARD_FORMULA,
        "train_aux_graph_alignment_weight": GRAPH_TRAIN_AUX_WEIGHT,
        "formal_evaluation_graph_alignment_weight": 0.0,
        "source_sha256": {p.name: _sha256(p) for p in sources if p.exists()},
        "forbidden_lineage": "retired experiment lineage is not imported or referenced",
    }
    canonical = json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    result["protocol_sha256"] = hashlib.sha256(canonical).hexdigest().upper()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--seeds", type=_parse_seeds)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--episodes-per-stratum", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()
    defaults = PRESETS[args.protocol]
    seeds = args.seeds or list(defaults[0])
    timesteps = args.timesteps if args.timesteps is not None else defaults[1]
    episodes = args.episodes_per_stratum if args.episodes_per_stratum is not None else defaults[2]
    n_envs = args.n_envs if args.n_envs is not None else defaults[3]
    root = (args.root or (FORMAL_ROOT if args.protocol == "formal" else Path("outputs/_smoke_complete_core_ablation_extension_20260717"))).resolve()
    lowered = str(root).casefold()
    if "v13" in lowered or "retired_experiments" in lowered:
        raise ValueError("forbidden output lineage")
    root.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(root, args.protocol, seeds, timesteps, episodes, n_envs)
    protocol_path = root / "experiment_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        immutable = ("protocol", "seeds", "variants", "timesteps_per_seed_variant", "episodes_per_seed_stratum_variant", "n_envs", "evaluation_strata", "full_reference_root", "source_sha256")
        mismatch = [key for key in immutable if existing.get(key) != protocol.get(key)]
        if mismatch:
            raise ValueError("resume protocol mismatch: " + ", ".join(mismatch))
    else:
        write_json(protocol_path, protocol)
    if args.initialize_only:
        print(f"INITIALIZED {protocol_path}", flush=True)
        return
    run_seeds = [args.worker_seed] if args.worker_seed is not None else seeds
    if any(seed not in seeds for seed in run_seeds):
        raise ValueError("worker seed outside frozen protocol")
    for seed in run_seeds:
        for variant in ABLATION_VARIANTS:
            model_dir = root / f"seed_{seed}" / "models" / variant
            model_path = model_dir / f"ppo_{variant}.zip"
            if not (args.resume and model_path.exists()):
                print(f"TRAIN seed={seed} variant={variant} steps={timesteps}", flush=True)
                model_path = train(model_dir, timesteps, seed, variant, n_envs)
            for pattern, intensity in LOAD_STRATA:
                stratum = f"{pattern}__{intensity}"
                output = root / f"seed_{seed}" / "eval" / stratum / variant
                if args.resume and (output / "episode_records.csv").exists():
                    continue
                print(f"EVAL seed={seed} variant={variant} stratum={stratum}", flush=True)
                evaluate(model_path, output, episodes, variant, pattern, intensity, 50_000 + seed)
    if not args.skip_aggregate:
        print(json.dumps(aggregate_root(root), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

