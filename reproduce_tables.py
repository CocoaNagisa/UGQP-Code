from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_MAP = {
    "success": "joint_success_rate",
    "quality": "mean_quality",
    "latency": "mean_latency_s",
    "deadline": "deadline_violation_rate",
    "cost": "mean_cost",
    "risk": "high_risk_local_execution_rate",
}


def utility(row: pd.Series, definition: pd.DataFrame) -> float:
    total = 0.0
    for item in definition.itertuples(index=False):
        value = float(row[METRIC_MAP[item.metric]])
        scaled = (value - item.min) / (item.max - item.min)
        if not bool(item.higher_is_better):
            scaled = (item.max - value) / (item.max - item.min)
        total += item.default_weight * float(np.clip(scaled, 0.0, 1.0))
    return 100.0 * total


def main_table(data: Path) -> pd.DataFrame:
    records = pd.read_csv(data / "main_comparison/episode_records.csv")
    published = pd.read_csv(data / "main_comparison/table1.csv")
    definition = pd.read_csv(data / "main_comparison/utility_definition.csv")
    learned = {"UGQP-GNN", "DTODRL-adapted"}
    rows = []
    for method in published["method"]:
        subset = records[records["method"] == method]
        if method in learned:
            seed_means = subset.groupby("training_seed", dropna=True).agg(
                mean_quality=("avg_quality", "mean"),
                joint_success_rate=("success", "mean"),
                mean_latency_s=("total_latency", "mean"),
                deadline_violation_rate=("deadline_violation", "mean"),
                mean_cost=("total_cost", "mean"),
                high_risk_local_execution_rate=("high_risk_local_ratio", "mean"),
                mean_reward=("reward", "mean"),
            )
            row = {column: float(seed_means[column].mean()) for column in seed_means.columns}
        else:
            row = {
                "mean_quality": float(subset["avg_quality"].mean()),
                "joint_success_rate": float(subset["success"].mean()),
                "mean_latency_s": float(subset["total_latency"].mean()),
                "deadline_violation_rate": float(subset["deadline_violation"].mean()),
                "mean_cost": float(subset["total_cost"].mean()),
                "high_risk_local_execution_rate": float(subset["high_risk_local_ratio"].mean()),
                "mean_reward": float(subset["reward"].mean()),
            }
        row["method"] = method
        row["composite_utility"] = utility(pd.Series(row), definition)
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_table(data: Path) -> pd.DataFrame:
    records = pd.read_csv(data / "ablation/episode_records.csv")
    definition = pd.read_csv(data / "main_comparison/utility_definition.csv")
    base_metrics = [
        "avg_quality",
        "success",
        "reward",
        "total_latency",
        "deadline_violation",
        "total_cost",
        "mean_chosen_queue_wait",
        "high_risk_local_ratio",
    ]
    groups = ["seed", "variant", "load_pattern", "load_intensity"]
    stratum = records.groupby(groups)[base_metrics].mean()
    stratum["p95_total_latency"] = records.groupby(groups)["total_latency"].quantile(0.95)
    per_seed = stratum.groupby(["seed", "variant"]).mean().reset_index()
    utilities = []
    for row in per_seed.itertuples(index=False):
        values = pd.Series(
            {
                "joint_success_rate": row.success,
                "mean_quality": row.avg_quality,
                "mean_latency_s": row.total_latency,
                "deadline_violation_rate": row.deadline_violation,
                "mean_cost": row.total_cost,
                "high_risk_local_execution_rate": row.high_risk_local_ratio,
            }
        )
        utilities.append(utility(values, definition))
    per_seed["composite_utility"] = utilities
    rows = []
    for variant, subset in per_seed.groupby("variant", sort=False):
        row = {"variant": variant, "training_seeds": len(subset)}
        for metric in [*base_metrics, "p95_total_latency", "composite_utility"]:
            row[f"mean_{metric}"] = float(subset[metric].mean())
            row[f"sd_{metric}"] = float(subset[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def assert_close(actual: pd.DataFrame, expected: pd.DataFrame, key: str, columns: list[str]) -> None:
    left = actual.set_index(key)
    right = expected.set_index(key)
    for label in right.index:
        for column in columns:
            if not np.isclose(left.loc[label, column], right.loc[label, column], rtol=1e-9, atol=1e-9):
                raise AssertionError(f"{key}={label}, column={column}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("../data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    table1 = main_table(args.data)
    expected1 = pd.read_csv(args.data / "main_comparison/table1.csv")
    assert_close(
        table1,
        expected1,
        "method",
        [
            "mean_quality",
            "joint_success_rate",
            "mean_latency_s",
            "deadline_violation_rate",
            "mean_cost",
            "high_risk_local_execution_rate",
            "mean_reward",
            "composite_utility",
        ],
    )

    table2 = ablation_table(args.data)
    expected2 = pd.read_csv(args.data / "ablation/table2.csv")
    assert_close(
        table2,
        expected2,
        "variant",
        [
            "mean_avg_quality",
            "mean_success",
            "mean_reward",
            "mean_composite_utility",
            "mean_total_latency",
            "mean_p95_total_latency",
            "mean_deadline_violation",
            "mean_total_cost",
            "mean_mean_chosen_queue_wait",
            "mean_high_risk_local_ratio",
        ],
    )
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        table1.to_csv(args.output / "table1_reproduced.csv", index=False)
        table2.to_csv(args.output / "table2_reproduced.csv", index=False)
    print("Table 1 verified from episode-level source data.")
    print("Table 2 verified from episode-level source data.")


if __name__ == "__main__":
    main()

