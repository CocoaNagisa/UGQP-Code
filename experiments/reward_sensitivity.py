"""Train and evaluate coordinated ±20% reward-coefficient profiles."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv


ROOT = Path(__file__).resolve().parent
CORE = ROOT.parent / "src"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from llm_offload_env.config import ACTIONS, make_config
from llm_offload_env.env import LLMAgentOffloadEnv
from llm_offload_env.gnn_policy import DAGMessagePassingExtractor


PAPER_MODEL_ROOT = ROOT.parent.parent / "data" / "main_comparison"
SEEDS = [2026, 2027, 2028]
TRAINING_STEPS = 60_000
EVALUATION_EPISODES = 400
EVALUATION_SEED = 20_000
PROFILES = {
    "efficiency_tilt": {
        "reward_latency_weight": 1.20,
        "reward_cost_weight": 0.24,
        "reward_quality_weight": 1.92,
        "reward_deadline_weight": 0.96,
        "reward_balance_weight": 0.216,
    },
    "default": {
        "reward_latency_weight": 1.00,
        "reward_cost_weight": 0.20,
        "reward_quality_weight": 2.40,
        "reward_deadline_weight": 1.20,
        "reward_balance_weight": 0.180,
    },
    "reliability_tilt": {
        "reward_latency_weight": 0.80,
        "reward_cost_weight": 0.16,
        "reward_quality_weight": 2.88,
        "reward_deadline_weight": 1.44,
        "reward_balance_weight": 0.144,
    },
}


def _sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries as a UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    """Return the arithmetic mean of a numeric field."""
    return statistics.fmean(float(row[key]) for row in rows)


def _paper_model(seed: int) -> Path:
    """Return the existing default-profile model for one seed."""
    return PAPER_MODEL_ROOT / f"seed_{seed}" / "ugqp_gnn" / "model" / "ppo_full.zip"


def _environment_factory(seed: int, weights: dict[str, float]):
    """Return one exact-config environment factory."""

    def factory() -> LLMAgentOffloadEnv:
        return LLMAgentOffloadEnv(
            make_config(
                seed=seed,
                variant="full",
                arrival_rate=0.7,
                network_scale=1.0,
                deadline_scale=1.55,
                queue_mode="bursty",
                queue_scale=1.5,
                profile_mode="tradeoff_v2",
                min_nodes=5,
                max_nodes=12,
                task_graph_mode="standard",
                quality_propagation_strength=1.50,
                **weights,
            )
        )

    return factory


def _train(output: Path, profile: str, seed: int) -> tuple[Path, float]:
    """Train or reuse one exact reward-profile model."""
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "ppo_full.zip"
    config_path = output / "train_config.json"
    expected = {"profile": profile, "seed": seed, "timesteps": TRAINING_STEPS, **PROFILES[profile]}
    if model_path.exists() and config_path.exists():
        actual = json.loads(config_path.read_text(encoding="utf-8"))
        if all(actual.get(key) == value for key, value in expected.items()):
            return model_path, 0.0
        raise RuntimeError(f"Existing reward-profile model has mismatched config: {config_path}")

    env = DummyVecEnv(
        [_environment_factory(seed + 1009 * index, PROFILES[profile]) for index in range(4)]
    )
    raw_env = env.envs[0]
    policy_kwargs = {
        "features_extractor_class": DAGMessagePassingExtractor,
        "features_extractor_kwargs": {
            "max_nodes": raw_env.config.max_nodes,
            "node_feature_dim": raw_env.node_feature_dim,
            "action_count": len(ACTIONS),
            "global_feature_dim": 6,
            "hidden_dim": 64,
            "message_steps": 2,
            "features_dim": 128,
        },
        "net_arch": {"pi": [128, 64], "vf": [128, 64]},
    }
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        n_steps=128,
        batch_size=64,
        gamma=0.96,
        learning_rate=3e-4,
        ent_coef=0.05,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=TRAINING_STEPS, progress_bar=False)
    elapsed = time.perf_counter() - started
    model.save(model_path)
    config_path.write_text(
        json.dumps(
            {
                **expected,
                "environment_seeds": [seed + 1009 * index for index in range(4)],
                "n_envs": 4,
                "min_nodes": 5,
                "max_nodes": 12,
                "arrival_rate": 0.7,
                "network_scale": 1.0,
                "deadline_scale": 1.55,
                "queue_mode": "bursty",
                "queue_scale": 1.5,
                "profile_mode": "tradeoff_v2",
                "quality_propagation_strength": 1.50,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    env.close()
    return model_path, elapsed


def _evaluate(model_path: Path, profile: str, seed: int) -> list[dict[str, object]]:
    """Evaluate one frozen policy on the common task stream."""
    model = MaskablePPO.load(model_path, device="cpu")
    rows: list[dict[str, object]] = []
    for episode in range(EVALUATION_EPISODES):
        env = _environment_factory(EVALUATION_SEED + episode, PROFILES[profile])()
        observation, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
            observation, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        rows.append(
            {
                "reward_profile": profile,
                "training_seed": seed,
                **env.episode_summary(f"UGQP_{profile}_seed{seed}", episode),
            }
        )
    return rows


def main() -> None:
    """Run coordinated reward sensitivity and write machine-readable evidence."""
    output = ROOT.parent / "outputs" / "sensitivity_evidence" / "reward"
    output.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    for profile in PROFILES:
        for seed in SEEDS:
            if profile == "default":
                model_path = _paper_model(seed)
                training_seconds = 0.0
                origin = "baseline-profile policy"
            else:
                model_path, training_seconds = _train(output / profile / f"seed_{seed}", profile, seed)
                origin = "profile-specific retrained policy"
            rows = _evaluate(model_path, profile, seed)
            episode_rows.extend(rows)
            model_rows.append(
                {
                    "reward_profile": profile,
                    "training_seed": seed,
                    "origin": origin,
                    "model_path": model_path.name,
                    "model_sha256": _sha256(model_path),
                    "new_training_wall_seconds": training_seconds,
                    **PROFILES[profile],
                }
            )
            print(
                f"profile={profile} seed={seed} quality={_mean(rows, 'avg_quality'):.4f} "
                f"DSR={1.0 - _mean(rows, 'deadline_violation'):.4f} "
                f"cost={_mean(rows, 'total_cost'):.4f}"
            )

    aggregate_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for profile in PROFILES:
        for seed in SEEDS:
            subset = [
                row
                for row in episode_rows
                if row["reward_profile"] == profile and int(row["training_seed"]) == seed
            ]
            seed_rows.append(
                {
                    "reward_profile": profile,
                    "training_seed": seed,
                    "episodes": len(subset),
                    "mean_total_latency": _mean(subset, "total_latency"),
                    "deadline_satisfaction_rate": 1.0 - _mean(subset, "deadline_violation"),
                    "mean_avg_quality": _mean(subset, "avg_quality"),
                    "success_rate": _mean(subset, "success"),
                    "mean_total_cost": _mean(subset, "total_cost"),
                    "mean_high_risk_local_ratio": _mean(subset, "high_risk_local_ratio"),
                    "mean_load_variance": _mean(subset, "load_variance"),
                }
            )
        pooled = [row for row in episode_rows if row["reward_profile"] == profile]
        aggregate_rows.append(
            {
                "reward_profile": profile,
                "training_seeds": len(SEEDS),
                "episodes": len(pooled),
                "mean_total_latency": _mean(pooled, "total_latency"),
                "deadline_satisfaction_rate": 1.0 - _mean(pooled, "deadline_violation"),
                "mean_avg_quality": _mean(pooled, "avg_quality"),
                "success_rate": _mean(pooled, "success"),
                "mean_total_cost": _mean(pooled, "total_cost"),
                "mean_high_risk_local_ratio": _mean(pooled, "high_risk_local_ratio"),
                "mean_load_variance": _mean(pooled, "load_variance"),
                **PROFILES[profile],
            }
        )

    _write_csv(output / "reward_episode_records.csv", episode_rows)
    _write_csv(output / "reward_seed_summary.csv", seed_rows)
    _write_csv(output / "reward_aggregate.csv", aggregate_rows)
    _write_csv(output / "reward_models.csv", model_rows)
    manifest = {
        "purpose": "Coordinated reward-coefficient sensitivity",
        "profiles": PROFILES,
        "training_seeds": SEEDS,
        "training_steps_per_new_model": TRAINING_STEPS,
        "new_models": 6,
        "reused_default_models": 3,
        "evaluation_episodes_per_seed": EVALUATION_EPISODES,
        "evaluation_seed": EVALUATION_SEED,
        "scope": (
            "All five reward coefficients are varied by ±20% in two coordinated relative-tradeoff endpoints. "
            "This is a bounded profile sensitivity, not an exhaustive one-at-a-time optimization."
        ),
    }
    (output / "reward_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
