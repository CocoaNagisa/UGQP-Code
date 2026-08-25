"""Benchmark scheduler-side CPU overhead for the frozen UGQP policy."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np
import torch
from sb3_contrib import MaskablePPO


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent.parent / "data" / "scheduler_overhead"
ARCHIVE_CODE = ROOT.parent / "src"
MODEL = ROOT.parent.parent / "data" / "main_comparison" / "seed_2026" / "ugqp_gnn" / "model" / "ppo_full.zip"
WARMUP_CALLS = 250
MEASURED_CALLS = 2000
BLOCKS = 5
BATCH_CALLS = 300
NODE_COUNTS = (5, 8, 12)
BATCH_SIZES = (1, 16, 64, 256)
EVALUATION_SEED = 20000


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file.

    :param path: Input file
    :type path: Path
    :return: SHA-256 digest
    :rtype: str
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    """Return a linear-interpolated percentile.

    :param values: Sample values
    :type values: list[float]
    :param probability: Quantile in the closed unit interval
    :type probability: float
    :return: Requested percentile
    :rtype: float
    """
    return float(np.quantile(np.asarray(values, dtype=float), probability, method="linear"))


def time_calls(function, calls: int) -> list[float]:
    """Measure repeated callable latency in milliseconds.

    :param function: Zero-argument callable
    :param calls: Number of measured invocations
    :type calls: int
    :return: One wall-clock latency per invocation
    :rtype: list[float]
    """
    values: list[float] = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(calls):
            start = time.perf_counter_ns()
            function()
            values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    finally:
        if gc_enabled:
            gc.enable()
    return values


def summarize(component: str, node_count: int, values: list[float]) -> dict[str, object]:
    """Summarize one latency sample.

    :param component: Benchmark component label
    :type component: str
    :param node_count: Number of real DAG nodes before padding
    :type node_count: int
    :param values: Per-call latency values in milliseconds
    :type values: list[float]
    :return: Summary row
    :rtype: dict[str, object]
    """
    return {
        "component": component,
        "real_nodes": node_count,
        "padded_nodes": 12,
        "calls": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "std_ms": statistics.stdev(values),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of dictionaries as UTF-8 CSV.

    :param path: Destination file
    :type path: Path
    :param rows: Data rows
    :type rows: list[dict[str, object]]
    """
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the scheduler overhead benchmark and write machine-readable evidence."""
    import sys

    sys.path.insert(0, str(ARCHIVE_CODE))
    from llm_offload_env.config import make_config
    from llm_offload_env.env import LLMAgentOffloadEnv
    from llm_offload_env.gnn_policy import DAGMessagePassingExtractor

    if not MODEL.exists():
        raise FileNotFoundError(MODEL)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(EVALUATION_SEED)
    np.random.seed(EVALUATION_SEED)
    model = MaskablePPO.load(MODEL, device="cpu")
    parameter_count = sum(parameter.numel() for parameter in model.policy.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.policy.parameters() if parameter.requires_grad
    )

    latency_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    reference_observation = None
    reference_mask = None

    for node_count in NODE_COUNTS:
        env = LLMAgentOffloadEnv(
            make_config(
                seed=EVALUATION_SEED,
                min_nodes=node_count,
                max_nodes=12,
                arrival_rate=0.7,
                queue_mode="bursty",
                queue_scale=1.5,
                profile_mode="tradeoff_v2",
                deadline_scale=1.55,
            )
        )
        observation, _ = env.reset()
        mask = env.action_masks()
        reference_observation = observation
        reference_mask = mask

        functions = {
            "observation_construction": env._obs,
            "feasibility_mask": env.action_masks,
            "policy_predict": lambda: model.predict(
                observation,
                deterministic=True,
                action_masks=mask,
            ),
            "complete_scheduler_decision": lambda: model.predict(
                env._obs(),
                deterministic=True,
                action_masks=env.action_masks(),
            ),
        }
        for function in functions.values():
            for _ in range(WARMUP_CALLS):
                function()
        for component, function in functions.items():
            combined: list[float] = []
            for block in range(BLOCKS):
                values = time_calls(function, MEASURED_CALLS // BLOCKS)
                combined.extend(values)
                block_rows.append(
                    {
                        "component": component,
                        "real_nodes": node_count,
                        "block": block + 1,
                        "calls": len(values),
                        "mean_ms": statistics.fmean(values),
                        "median_ms": statistics.median(values),
                        "p95_ms": percentile(values, 0.95),
                    }
                )
            latency_rows.append(summarize(component, node_count, combined))

    if reference_observation is None or reference_mask is None:
        raise RuntimeError("No reference observation was produced")
    for batch_size in BATCH_SIZES:
        observations = np.repeat(reference_observation[None, :], batch_size, axis=0)
        masks = np.repeat(reference_mask[None, :], batch_size, axis=0)
        for _ in range(WARMUP_CALLS):
            model.predict(observations, deterministic=True, action_masks=masks)
        total_start = time.perf_counter_ns()
        for _ in range(BATCH_CALLS):
            model.predict(observations, deterministic=True, action_masks=masks)
        elapsed_seconds = (time.perf_counter_ns() - total_start) / 1_000_000_000.0
        decisions = batch_size * BATCH_CALLS
        batch_rows.append(
            {
                "batch_size": batch_size,
                "batch_calls": BATCH_CALLS,
                "decisions": decisions,
                "elapsed_seconds": elapsed_seconds,
                "decisions_per_second": decisions / elapsed_seconds,
                "mean_batch_ms": 1000.0 * elapsed_seconds / BATCH_CALLS,
                "amortized_ms_per_decision": 1000.0 * elapsed_seconds / decisions,
            }
        )

    tracemalloc.start()
    for _ in range(1000):
        model.predict(reference_observation, deterministic=True, action_masks=reference_mask)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    write_csv(EVIDENCE / "scheduler_component_latency.csv", latency_rows)
    write_csv(EVIDENCE / "scheduler_component_latency_blocks.csv", block_rows)
    write_csv(EVIDENCE / "scheduler_batch_throughput.csv", batch_rows)
    manifest = {
        "purpose": "Scheduler-side CPU overhead microbenchmark",
        "model": MODEL.name,
        "model_sha256": sha256(MODEL),
        "code_root": "src",
        "policy_parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "fp32_parameter_storage_bytes": parameter_count * 4,
        "python_tracemalloc_current_bytes_after_1000_predict_calls": current_bytes,
        "python_tracemalloc_peak_bytes_for_1000_predict_calls": peak_bytes,
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "node_counts": list(NODE_COUNTS),
        "padded_node_budget": 12,
        "action_count": 5,
        "warmup_calls_per_component": WARMUP_CALLS,
        "measured_calls_per_component": MEASURED_CALLS,
        "measurement_blocks": BLOCKS,
        "batch_sizes": list(BATCH_SIZES),
        "batch_calls": BATCH_CALLS,
        "evaluation_seed": EVALUATION_SEED,
        "timing_clock": "time.perf_counter_ns",
        "scope": (
            "CPU scheduler microbenchmark only; simulator service latency and external LLM serving "
            "latency are excluded"
        ),
    }
    (EVIDENCE / "scheduler_overhead_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    for row in latency_rows:
        print(row)
    for row in batch_rows:
        print(row)


if __name__ == "__main__":
    main()
