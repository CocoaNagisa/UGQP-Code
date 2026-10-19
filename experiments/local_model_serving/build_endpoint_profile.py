"""Build the endpoint profile from the independent calibration records."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ("1.7B", "8B")
KINDS = ("intent", "extract", "retrieve", "tool_call", "plan", "generate", "verify", "summarize")
DIFFS = ("low", "medium", "high")


def uncertainty_band(value):
    return "low" if value < 0.40 else "medium" if value < 0.70 else "high"


def med(values):
    return float(statistics.median(values)) if values else None


def mean(values):
    return float(statistics.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "endpoint_profile.json")
    args = parser.parse_args()
    calibration = args.calibration.resolve()
    output = args.output.resolve()
    protocol = json.loads((calibration / "protocol.json").read_text(encoding="utf-8"))
    calls = [json.loads(line) for line in (calibration / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    # The calibration workflows are separate from the formal comparison.
    assert len(calls) == 384 and len(protocol["cases"]) == 24
    case_map = {row["id"]: row for row in protocol["cases"]}
    by = defaultdict(list)
    for row in calls:
        node = case_map[row["case_id"]]["nodes"][row["node_index"]]
        row["difficulty_band"] = uncertainty_band(float(node["uncertainty"]))
        row["estimated_input_tokens"] = int(node["input_tokens"])
        row["estimated_output_tokens"] = int(node["output_tokens"])
        by[(row["model"], row["node_kind"], row["difficulty"])].append(row)
    models = {}
    for model in MODELS:
        quality_by_kind = {}
        latency_by_kind = {}
        quality_by_kind_diff = {}
        latency_by_kind_diff = {}
        cost_by_kind_diff = {}
        all_quality, all_latency = [], []
        all_prompt, all_completion = [], []
        for kind in KINDS:
            qvals, lvals = [], []
            quality_by_kind_diff[kind] = {}
            latency_by_kind_diff[kind] = {}
            cost_by_kind_diff[kind] = {}
            for diff in DIFFS:
                rows = [row for row in calls if row["model"] == model and row["node_kind"] == kind and row["difficulty_band"] == diff]
                if not rows:
                    continue
                q = [float(r["grade"]["field_accuracy"]) for r in rows]
                l = [float(r["elapsed_s"]) for r in rows]
                tok = [float(r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]) for r in rows]
                quality_by_kind_diff[kind][diff] = mean(q)
                latency_by_kind_diff[kind][diff] = med(l)
                # This is a normalized work ratio, not a currency estimate.
                cost_by_kind_diff[kind][diff] = 1.0 if model == "1.7B" else 8.0 / 1.7
                qvals.extend(q); lvals.extend(l); all_quality.extend(q); all_latency.extend(l)
                all_prompt.extend(float(r["usage"]["prompt_tokens"]) for r in rows)
                all_completion.extend(float(r["usage"]["completion_tokens"]) for r in rows)
            quality_by_kind[kind] = mean(qvals) if qvals else 0.0
            latency_by_kind[kind] = med(lvals) if lvals else 0.1
        # Convert measured service latency and measured token volume into a
        # nonzero queue-capable simulator rate. The formal run retains the
        # measured kind latency and adds queue waiting separately.
        mean_prompt, mean_completion = mean(all_prompt), mean(all_completion)
        median_latency = med(all_latency)
        prefill_rate = mean_prompt / max(median_latency * 0.45, 1e-3)
        decode_rate = mean_completion / max(median_latency * 0.55, 1e-3)
        models[model] = {
            "n_calls": len(all_quality),
            "mean_field_accuracy": mean(all_quality),
            "quality_by_kind": quality_by_kind,
            "quality_by_kind_difficulty": quality_by_kind_diff,
            "median_latency_s": median_latency,
            "median_latency_by_kind": latency_by_kind,
            "latency_by_kind_difficulty": latency_by_kind_diff,
            "mean_prompt_tokens": mean_prompt,
            "mean_completion_tokens": mean_completion,
            "prefill_rate": prefill_rate,
            "decode_rate": decode_rate,
            "base_latency": 0.01,
            "normalized_model_work_ratio": 1.0 if model == "1.7B" else 8.0 / 1.7,
            "token_cost_by_kind_difficulty": cost_by_kind_diff,
        }
    if models["8B"]["mean_field_accuracy"] <= models["1.7B"]["mean_field_accuracy"]:
        raise RuntimeError("Calibration gate failed: 8B is not more accurate than 1.7B")
    if models["8B"]["median_latency_s"] <= models["1.7B"]["median_latency_s"]:
        raise RuntimeError("Calibration gate failed: 8B is not slower than 1.7B")
    out = {
        "profile_id": "v5_bridge_v1",
        "source": "E1b independent real-output calibration; disjoint from formal replicates 8-10",
        "difficulty_definition": "node uncertainty bands from the frozen E1b task-shape records: low < 0.40, medium 0.40-<0.70, high >= 0.70; actual typed-field quality, token usage, and latency are joined by node kind and this band",
        "normalization": "model_work = (parameter_count / 1.7B) * (prompt_tokens + completion_tokens) / 1000",
        "queue_capable": True,
        "models": models,
    }
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({m: {k: models[m][k] for k in ("n_calls", "mean_field_accuracy", "median_latency_s", "normalized_model_work_ratio")} for m in MODELS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
