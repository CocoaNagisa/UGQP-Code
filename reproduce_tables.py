"""Recompute the reported learned-policy summaries from released records."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METHOD_DIRS = {"ugqp_gnn": "UGQP-GNN", "dtodrl_frozen": "DTODRL-frozen", "dtodrl_joint": "DTODRL-joint", "ppo_mlp": "PPO-MLP"}
METRICS = {
    "mean_total_latency": "total_latency",
    "mean_deadline_violation": "deadline_violation",
    "mean_avg_quality": "avg_quality",
    "mean_success": "success",
    "mean_total_cost": "total_cost",
    "mean_high_risk_local_ratio": "high_risk_local_ratio",
    "mean_offload_ratio": "offload_ratio",
    "mean_reward": "reward",
    "mean_mean_chosen_queue_wait": "mean_chosen_queue_wait",
}


def main_summary(data_root: Path) -> pd.DataFrame:
    rows = []
    main_root = data_root / "main_comparison"
    for seed_dir in sorted(main_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        for folder, method in METHOD_DIRS.items():
            path = seed_dir / folder / "eval" / "episode_records.csv"
            frame = pd.read_csv(path)
            row = {"method": method, "training_seed": seed}
            row.update({name: float(frame[column].mean()) for name, column in METRICS.items()})
            rows.append(row)
    per_seed = pd.DataFrame(rows)
    aggregate = per_seed.groupby("method", as_index=False)[list(METRICS)].mean()
    expected = pd.read_csv(main_root / "analysis" / "method_aggregate.csv")
    for metric in METRICS:
        expected_column = f"{metric}_mean"
        left = aggregate.set_index("method")[metric]
        right = expected.set_index("method")[expected_column]
        for method in right.index:
            if not np.isclose(left[method], right[method], atol=1e-9):
                raise AssertionError(f"{method}: {metric}")
    return aggregate


def queue_control_summary(data_root: Path) -> pd.DataFrame:
    path = data_root / "queue_observation_control" / "analysis" / "aggregate.csv"
    result = pd.read_csv(path)
    if len(result) != 1 or int(result.iloc[0]["training_repeats"]) != 5:
        raise AssertionError("queue-observation control is incomplete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    learned = main_summary(args.data)
    control = queue_control_summary(args.data)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        learned.to_csv(args.output / "main_summary_reproduced.csv", index=False)
        control.to_csv(args.output / "queue_control_checked.csv", index=False)
    print("Main comparison summaries verified.")
    print("Queue-observation control verified.")


if __name__ == "__main__":
    main()
