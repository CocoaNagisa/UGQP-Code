"""Summarize real DAG traces and compare simulator serving profiles."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


SIMULATION_PROFILES = {
    "local": {"prefill": 1800.0, "decode": 55.0, "base": 0.040},
    "edge_s": {"prefill": 4800.0, "decode": 160.0, "base": 0.080},
    "edge_m": {"prefill": 4000.0, "decode": 115.0, "base": 0.120},
    "edge_l": {"prefill": 3000.0, "decode": 75.0, "base": 0.180},
    "cloud_compute_only": {"prefill": 6500.0, "decode": 180.0, "base": 0.350},
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load non-empty JSON lines from a trace file."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values: Iterable[float], probability: float) -> float:
    """Return an inclusive linearly interpolated percentile."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile from an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_profile(records: List[Dict[str, Any]], profile: Dict[str, float]) -> Dict[str, float]:
    """Compare one serving profile with measured node latencies."""
    actual = [float(row["latency_s"]) for row in records]
    predicted = [
        profile["base"]
        + float(row["prompt_tokens"]) / profile["prefill"]
        + float(row["output_tokens"]) / profile["decode"]
        for row in records
    ]
    absolute_errors = [abs(prediction - observation) for prediction, observation in zip(predicted, actual)]
    percentage_errors = [error / observation for error, observation in zip(absolute_errors, actual) if observation > 0]
    return {
        "mae_s": statistics.mean(absolute_errors),
        "mape_percent": statistics.mean(percentage_errors) * 100.0,
        "predicted_mean_s": statistics.mean(predicted),
        "measured_mean_s": statistics.mean(actual),
    }


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a reproducible summary from real-serving experiment outputs."""
    records = load_jsonl(args.trace)
    microbenchmark = json.loads(args.microbenchmark.read_text(encoding="utf-8"))
    workflows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scenarios: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        workflows[row["workflow_id"]].append(row)
        scenarios[row["scenario"]].append(row)

    scenario_summary: Dict[str, Dict[str, float]] = {}
    for scenario, rows in scenarios.items():
        workflow_latencies = [
            sum(float(item["latency_s"]) for item in workflow_rows)
            for workflow_rows in workflows.values()
            if workflow_rows[0]["scenario"] == scenario
        ]
        scenario_summary[scenario] = {
            "nodes": len(rows),
            "mean_workflow_latency_s": statistics.mean(workflow_latencies),
            "mean_node_latency_s": statistics.mean(float(row["latency_s"]) for row in rows),
            "json_valid_percent": statistics.mean(bool(row["json_valid"]) for row in rows) * 100.0,
        }

    gpu_rows = [row["gpu"] for row in records if row.get("gpu")]
    power_values = [float(row["power_w"]) for row in gpu_rows if row.get("power_w") is not None]
    fitted_profile = {
        "prefill": float(microbenchmark["prefill_fit"]["prefill_tokens_per_s"]),
        "decode": float(microbenchmark["decode_fit"]["decode_tokens_per_s"]),
        "base": float(microbenchmark["prefill_fit"]["base_latency_s"]),
    }
    profiles = dict(SIMULATION_PROFILES)
    profiles["independently_fitted_laptop"] = fitted_profile

    return {
        "trace": str(args.trace.resolve()),
        "microbenchmark": str(args.microbenchmark.resolve()),
        "nodes": len(records),
        "workflows": len(workflows),
        "prompt_tokens": {
            "min": min(int(row["prompt_tokens"]) for row in records),
            "max": max(int(row["prompt_tokens"]) for row in records),
        },
        "output_tokens": {
            "min": min(int(row["output_tokens"]) for row in records),
            "max": max(int(row["output_tokens"]) for row in records),
        },
        "json_valid_percent": statistics.mean(bool(row["json_valid"]) for row in records) * 100.0,
        "required_keys_valid_percent": statistics.mean(bool(row["required_keys_valid"]) for row in records) * 100.0,
        "ttft_mean_s": statistics.mean(float(row["ttft_s"]) for row in records),
        "ttft_p95_s": percentile((float(row["ttft_s"]) for row in records), 0.95),
        "node_latency_mean_s": statistics.mean(float(row["latency_s"]) for row in records),
        "node_latency_p95_s": percentile((float(row["latency_s"]) for row in records), 0.95),
        "decode_tokens_per_s_mean": statistics.mean(
            float(row["decode_tokens_per_s"]) for row in records if row["decode_tokens_per_s"] is not None
        ),
        "gpu_snapshot": {
            "memory_used_mib_mean": statistics.mean(float(row["used_mib"]) for row in gpu_rows),
            "power_w_mean": statistics.mean(power_values) if power_values else None,
        },
        "by_scenario": scenario_summary,
        "profile_comparison": {
            name: summarize_profile(records, profile) for name, profile in profiles.items()
        },
    }


def main() -> None:
    """Parse command-line arguments and write the experiment summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("data/prototype_validation/real_dag_calibration.jsonl"))
    parser.add_argument(
        "--microbenchmark",
        type=Path,
        default=Path("data/prototype_validation/vllm_profile_microbenchmark.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/prototype_validation/real_dag_calibration_summary.json"))
    args = parser.parse_args()
    summary = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
