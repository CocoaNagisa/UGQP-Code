from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from common_io import write_csv, write_json
from eval_baselines import evaluate_baselines
from infer_ppo import evaluate_model
from pretrain_dtodrl_gat import pretrain
from train_ppo import train


TRAIN_SEEDS = [2026, 2027, 2028]
RULE_POLICIES = [
    "local_only",
    "edge_only",
    "random_feasible",
    "goa_dag",
    "queue_heuristic",
]
METRICS = [
    "mean_total_latency",
    "mean_deadline_violation",
    "mean_avg_quality",
    "mean_success",
    "mean_total_cost",
    "mean_high_risk_local_ratio",
    "mean_offload_ratio",
    "mean_reward",
    "mean_mean_chosen_queue_wait",
]


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}, found {len(rows)}")
    return rows[0]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(root: Path) -> None:
    seed_level = []
    for seed in TRAIN_SEEDS:
        for method, folder in [
            ("UGQP-GNN", "ugqp_gnn"),
            ("DTODRL-adapted", "dtodrl_adapted"),
        ]:
            row = _read_single_row(root / f"seed_{seed}" / "eval" / folder / "summary.csv")
            seed_level.append(
                {
                    "method": method,
                    "training_seed": seed,
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            )
    write_csv(root / "analysis" / "learned_method_seed_summary.csv", seed_level)

    learned_summary = []
    for method in ["UGQP-GNN", "DTODRL-adapted"]:
        subset = [row for row in seed_level if row["method"] == method]
        result: dict[str, object] = {
            "method": method,
            "training_repeats": len(subset),
            "episodes_per_repeat": 400,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in subset]
            result[f"{metric}_across_seed_mean"] = statistics.mean(values)
            result[f"{metric}_across_seed_sd"] = statistics.stdev(values)
        learned_summary.append(result)
    write_csv(root / "analysis" / "learned_method_aggregate.csv", learned_summary)

    paired_seed_effects = []
    for seed in TRAIN_SEEDS:
        ugqp = next(
            row for row in seed_level if row["method"] == "UGQP-GNN" and row["training_seed"] == seed
        )
        dtodrl = next(
            row
            for row in seed_level
            if row["method"] == "DTODRL-adapted" and row["training_seed"] == seed
        )
        paired_seed_effects.append(
            {
                "training_seed": seed,
                "success_pp_ugqp_minus_dtodrl": 100.0
                * (float(ugqp["mean_success"]) - float(dtodrl["mean_success"])),
                "quality_ugqp_minus_dtodrl": float(ugqp["mean_avg_quality"])
                - float(dtodrl["mean_avg_quality"]),
                "latency_seconds_ugqp_minus_dtodrl": float(ugqp["mean_total_latency"])
                - float(dtodrl["mean_total_latency"]),
                "deadline_pp_ugqp_minus_dtodrl": 100.0
                * (
                    float(ugqp["mean_deadline_violation"])
                    - float(dtodrl["mean_deadline_violation"])
                ),
            }
        )
    write_csv(root / "analysis" / "paired_seed_effects.csv", paired_seed_effects)

    rules = _read_rows(root / "baselines" / "core" / "baseline_summary.csv")
    write_csv(root / "analysis" / "rule_baseline_summary.csv", rules)


def run(
    root: Path,
    source_model_root: Path,
    timesteps: int,
    episodes: int,
    resume: bool,
    deadline_scale: float,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "pretraining" / "dtodrl_gat_pretrained.pt"
    if not (resume and checkpoint.exists()):
        print("PRETRAIN DTODRL GAT", flush=True)
        checkpoint = pretrain(root / "pretraining")

    baseline_summary = root / "baselines" / "core" / "baseline_summary.csv"
    if not (resume and baseline_summary.exists()):
        print("EVAL common rule baselines", flush=True)
        evaluate_baselines(
            root / "baselines" / "core",
            episodes=episodes,
            seed=20000,
            policies=RULE_POLICIES,
            arrival_rate=0.7,
            network_scale=1.0,
            deadline_scale=deadline_scale,
            queue_mode="bursty",
            queue_scale=1.5,
            profile_mode="tradeoff_v2",
        )

    for seed in TRAIN_SEEDS:
        source_model = source_model_root / f"seed_{seed}" / "models" / "full" / "ppo_full.zip"
        if not source_model.exists():
            raise FileNotFoundError(f"Missing frozen UGQP model: {source_model}")

        ugqp_output = root / f"seed_{seed}" / "eval" / "ugqp_gnn"
        if not (resume and (ugqp_output / "summary.csv").exists()):
            print(f"EVAL frozen UGQP seed={seed}", flush=True)
            evaluate_model(
                source_model,
                ugqp_output,
                episodes=episodes,
                seed=20000,
                variant="full",
                policy_name="UGQP-GNN",
                arrival_rate=0.7,
                network_scale=1.0,
                deadline_scale=deadline_scale,
                queue_mode="bursty",
                queue_scale=1.5,
                profile_mode="tradeoff_v2",
            )

        model_dir = root / f"seed_{seed}" / "models" / "dtodrl_adapted"
        model = model_dir / "ppo_full.zip"
        if not (resume and model.exists()):
            print(f"TRAIN DTODRL-adapted seed={seed} steps={timesteps}", flush=True)
            model = train(
                model_dir,
                timesteps,
                seed=seed,
                variant="full",
                arrival_rate=0.7,
                network_scale=1.0,
                deadline_scale=deadline_scale,
                encoder="dtodrl",
                queue_mode="bursty",
                queue_scale=1.5,
                ent_coef=0.05,
                profile_mode="tradeoff_v2",
                n_envs=4,
                min_nodes=5,
                max_nodes=12,
                task_graph_mode="standard",
                quality_propagation_strength=1.50,
                dtodrl_checkpoint=checkpoint,
            )

        dtodrl_output = root / f"seed_{seed}" / "eval" / "dtodrl_adapted"
        if not (resume and (dtodrl_output / "summary.csv").exists()):
            print(f"EVAL DTODRL-adapted seed={seed}", flush=True)
            evaluate_model(
                model,
                dtodrl_output,
                episodes=episodes,
                seed=20000,
                variant="full",
                policy_name="DTODRL-adapted",
                arrival_rate=0.7,
                network_scale=1.0,
                deadline_scale=deadline_scale,
                queue_mode="bursty",
                queue_scale=1.5,
                profile_mode="tradeoff_v2",
            )

    aggregate(root)
    write_json(
        root / "experiment_protocol.json",
        {
            "train_seeds": TRAIN_SEEDS,
            "evaluation_seed": 20000,
            "episodes_per_method_per_seed": episodes,
            "timesteps_per_learned_method_seed": timesteps,
            "deadline_scale": deadline_scale,
            "common_action_set": ["local", "edge_s", "edge_m", "edge_l", "cloud"],
            "common_feasibility_mask": True,
            "common_environment_reward_for_learned_methods": True,
            "dtodrl_adaptation": {
                "task_encoder": "two-layer four-head GAT",
                "pretraining": "feature reconstruction plus graph-structure reconstruction",
                "pretrained_encoder_frozen_during_ppo": True,
                "ppo_budget_matches_ugqp": True,
                "original_joint_ready-node/location_action_reduced_to_location_only": True,
                "reason": "the shared environment fixes the current ready node in topological order",
            },
            "goa_dag_adaptation": {
                "token_uncertainty": "alpha=1-(p1-p2), profile-derived first-token proxy",
                "uncertainty_threshold": 0.60,
                "delay_gap_recomputed_after_each_node": True,
                "high_uncertainty_remote_constraint": True,
                "profile_limitation": "no real decoder logits are available",
            },
            "source_ugqp_model_root": str(source_model_root),
            "goa_primary_source": "https://arxiv.org/abs/2602.07958",
            "dtodrl_primary_source": "https://doi.org/10.1109/JIOT.2024.3374969",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/paper_v9_external_baselines_20260713"),
    )
    parser.add_argument(
        "--source-model-root",
        type=Path,
        default=Path("outputs/paper_v5_bursty_20260710"),
    )
    parser.add_argument("--timesteps", type=int, default=60000)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--deadline-scale", type=float, default=1.25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        args.root,
        args.source_model_root,
        args.timesteps,
        args.episodes,
        args.resume,
        args.deadline_scale,
    )
    print(f"saved external-baseline results: {args.root}")


if __name__ == "__main__":
    main()
