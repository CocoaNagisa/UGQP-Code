"""Calibrate the 1.7B and 8B endpoints on independent eight-stage DAG workflows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np


SEED = 20261020
BOOTSTRAPS = 3000
FAMILIES = ("flood", "traffic", "pump", "lighting")
DIFFICULTIES = ("low", "medium", "high")
REPLICATES = (4, 5)
MODELS = {
    "1.7B": {"url": "http://127.0.0.1:8102", "served_name": "qwen3-17"},
    "8B": {"url": "http://127.0.0.1:8104", "served_name": "qwen3-8b"},
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def generate_cases(helper, taskgen, e1_module) -> list[dict]:
    cases = []
    shape_rng = random.Random(SEED)
    serial = 1
    for family in FAMILIES:
        for difficulty in DIFFICULTIES:
            for replicate in REPLICATES:
                facts = e1_module.make_facts(family, difficulty, replicate)
                resources = e1_module.make_resources(
                    family, difficulty, serial, random.Random(SEED + serial)
                )
                scene = helper.make_scene(family, f"E1B{serial:02d}", facts, resources)
                items = helper.specs(scene)
                nodes = []
                for idx, item in enumerate(items):
                    input_tokens, output_tokens, uncertainty = taskgen.sample_node_shape(
                        shape_rng, item["kind"], e1_module.TASKGEN_SCENARIO[family]
                    )
                    privacy = shape_rng.choices([1, 2, 3], weights=[0.22, 0.46, 0.32])[0]
                    risk = min(1.0, max(0.0, uncertainty * shape_rng.uniform(0.75, 1.30)))
                    node = taskgen.Node(
                        idx, item["kind"], input_tokens, output_tokens, uncertainty,
                        privacy, risk, shape_rng.uniform(0.65, 1.45),
                        predecessors=list(item["pred"]),
                        successors=[j for j, other in enumerate(items) if idx in other["pred"]],
                    )
                    nodes.append(node)
                taskgen.compute_graph_features(nodes)
                cases.append({
                    "id": scene["id"], "family": family, "difficulty": difficulty,
                    "replicate": replicate, "scene": scene,
                    "items": items, "nodes": [node.__dict__ for node in nodes],
                })
                serial += 1
    if len(cases) != 24 or len({row["id"] for row in cases}) != 24:
        raise AssertionError("Expected 24 unique calibration workflows")
    return cases


def endpoint_models(e1_module) -> dict:
    return e1_module.endpoint_models()


def prepare(args) -> None:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving existing calibration directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    helper_path = Path(args.helper).resolve()
    e1_path = Path(args.e1_script).resolve()
    code_root = Path(args.code_root).resolve()
    import sys
    sys.path.insert(0, str(code_root / "src"))
    helper = load_module(helper_path, "e1b_readonly_helper")
    e1_module = load_module(e1_path, "e1b_source_generator")
    from llm_offload_env import taskgen
    from llm_offload_env.config import make_config
    from llm_offload_env.serving import default_profiles
    cases = generate_cases(helper, taskgen, e1_module)
    endpoint_data = endpoint_models(e1_module)
    profile_map = {"1.7B": "local", "8B": "edge_s"}
    protocol = {
        "protocol": "E1b independent endpoint and Eq. (7) calibration over the complete eight-stage DAG",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "sealed_before_generation",
        "independent_cases": 24,
        "stratification": "4 event families x 3 predeclared difficulty tiers x 2 new replicates (4 and 5), disjoint from E1 replicates 1-3 and E2 replicates 6-7",
        "node_kinds": ["intent", "extract", "retrieve", "retrieve", "tool_call", "plan", "verify", "summarize"],
        "models": MODELS,
        "endpoint_identities": endpoint_data,
        "profile_mapping": profile_map,
        "decoding": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": SEED,
                     "max_tokens": 256, "enable_thinking": False, "stream": True},
        "scoring": "Exact typed fields and schema through the frozen helper; no LLM judge. Each workflow uses actual model outputs as predecessor context.",
        "proxy": "Unchanged archived tradeoff_v2 quality terms and propagation strength, applied recursively to the eight-stage DAG; no proxy coefficient fitting.",
        "cases": cases,
        "source_hashes": {
            "script": file_digest(Path(__file__).resolve()),
            "helper": file_digest(helper_path),
            "e1_script": file_digest(e1_path),
            "config": file_digest(code_root / "src/llm_offload_env/config.py"),
            "serving": file_digest(code_root / "src/llm_offload_env/serving.py"),
            "taskgen": file_digest(code_root / "src/llm_offload_env/taskgen.py"),
        },
        "quality_propagation_strength": make_config().quality_propagation_strength,
        "profile_quality_base": {name: default_profiles("tradeoff_v2")[profile].quality_base
                                 for name, profile in profile_map.items()},
    }
    write_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(file_digest(output / "protocol.json") + "\n", encoding="ascii")
    print(json.dumps({"sealed": str(output), "cases": len(cases), "endpoint_identities": endpoint_data},
                     ensure_ascii=False), flush=True)


def check_seal(output: Path, helper_path: Path, e1_path: Path, code_root: Path) -> dict:
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if file_digest(protocol_path) != (output / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("Protocol hash mismatch")
    current = {
        "script": file_digest(Path(__file__).resolve()),
        "helper": file_digest(helper_path),
        "e1_script": file_digest(e1_path),
        "config": file_digest(code_root / "src/llm_offload_env/config.py"),
        "serving": file_digest(code_root / "src/llm_offload_env/serving.py"),
        "taskgen": file_digest(code_root / "src/llm_offload_env/taskgen.py"),
    }
    if current != protocol["source_hashes"]:
        raise RuntimeError("Sealed source hash mismatch")
    return protocol


def proxy_values(case: dict, model: str, protocol: dict, serving, cfg) -> list[float]:
    profile = serving.default_profiles("tradeoff_v2")[protocol["profile_mapping"][model]]
    strength = cfg.quality_propagation_strength
    predicted = []
    for data in case["nodes"]:
        node = type("ProxyNode", (), data)()
        base = float(serving.quality(node, profile))
        prior = float(np.mean([predicted[index] for index in node.predecessors])) if node.predecessors else 1.0
        penalty = strength * (1.0 - prior) * node.downstream_risk * max(0.0, 1.0 - profile.capability)
        predicted.append(float(min(1.0, max(0.0, base - penalty))))
    return predicted


def run(args) -> None:
    output = Path(args.output).resolve()
    helper_path = Path(args.helper).resolve()
    e1_path = Path(args.e1_script).resolve()
    code_root = Path(args.code_root).resolve()
    protocol = check_seal(output, helper_path, e1_path, code_root)
    log = output / ("calls_smoke.jsonl" if args.smoke else "calls.jsonl")
    if log.exists():
        raise FileExistsError(f"Preserving existing calls: {log}")
    helper = load_module(helper_path, "e1b_runtime_helper")
    e1_module = load_module(e1_path, "e1b_runtime_source")
    import sys
    sys.path.insert(0, str(code_root / "src"))
    from llm_offload_env.config import make_config
    from llm_offload_env import serving
    cfg = make_config()
    cases = protocol["cases"]
    selected_cases = cases[:1] if args.smoke else cases
    for case_index, case in enumerate(selected_cases):
        model_order = ("1.7B", "8B") if case_index % 2 == 0 else ("8B", "1.7B")
        for model_name in model_order:
            endpoint = protocol["models"][model_name]
            outputs = {}
            predicted = proxy_values(case, model_name, protocol, serving, cfg)
            for index, item in enumerate(case["items"]):
                prompt = helper.prompt_for(case["scene"], case["items"], index, outputs)
                result = e1_module.stream_call(endpoint["url"], endpoint["served_name"], prompt, SEED)
                expected = item["expected"]
                if index == 6:
                    expected = helper.conditional_verify(case["scene"], outputs)
                grade = helper.grade(result["output"], expected)
                feedback = helper.contract_feedback(case["scene"], case["items"], index,
                                                     {**outputs, index: result["output"]}, result["output"])
                outputs[index] = result["output"]
                append_jsonl(log, {
                    "case_id": case["id"], "family": case["family"], "difficulty": case["difficulty"],
                    "model": model_name, "served_name": endpoint["served_name"], "node_index": index,
                    "node_kind": item["kind"], "predecessors": item["pred"], "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "expected": expected, "output": result["output"], "grade": grade,
                    "actual_quality": grade["field_accuracy"], "proxy_quality": predicted[index],
                    "contract_feedback": feedback, "usage": result["usage"],
                    "finish_reason": result["finish_reason"], "elapsed_s": result["elapsed_s"],
                    "ttft_s": result["ttft_s"], "seed": SEED,
                })
            print(f"completed {case_index + 1}/{len(selected_cases)} {case['id']} {model_order}", flush=True)
    if not args.smoke:
        analyze(args)


def average_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    x, y = average_ranks(left), average_ranks(right)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def bootstrap_mean_ci(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(array, size=len(array), replace=True))) for _ in range(BOOTSTRAPS)]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def proxy_correlation(rows: list[dict], model: str, node_kind: str | None = None) -> dict:
    selected = [row for row in rows if row["model"] == model and
                (node_kind is None or row["node_kind"] == node_kind)]
    case_ids = sorted({row["case_id"] for row in selected})
    point = spearman([r["proxy_quality"] for r in selected], [r["actual_quality"] for r in selected])
    by_case = {case: [r for r in selected if r["case_id"] == case] for case in case_ids}
    rng = np.random.default_rng(SEED + (11 if model == "1.7B" else 13) + len(node_kind or "all"))
    draws = []
    for _ in range(BOOTSTRAPS):
        sample = rng.choice(case_ids, size=len(case_ids), replace=True)
        sampled_rows = [row for case in sample for row in by_case[case]]
        value = spearman([r["proxy_quality"] for r in sampled_rows],
                         [r["actual_quality"] for r in sampled_rows])
        if value is not None:
            draws.append(value)
    return {"n_cases": len(case_ids), "n_nodes": len(selected), "rho": point,
            "case_cluster_95_ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
            if draws else None,
            "mae": float(np.mean([abs(r["proxy_quality"] - r["actual_quality"]) for r in selected]))}


def analyze(args) -> None:
    output = Path(args.output).resolve()
    rows = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 24 * 8 * 2:
        raise RuntimeError(f"Expected 384 calls, found {len(rows)}")
    cases = sorted({row["case_id"] for row in rows})
    summary = {"status": "completed", "independent_cases": len(cases), "calls": len(rows),
               "models": {}, "by_stage": {}, "paired_case_mean_quality": {}, "strict_workflow": {},
               "proxy_association_pooled": {}, "proxy_association_by_stage": {}}
    for model_index, model in enumerate(("1.7B", "8B")):
        selected = [row for row in rows if row["model"] == model]
        summary["models"][model] = {
            "mean_field_accuracy": float(np.mean([row["actual_quality"] for row in selected])),
            "node_all_fields_rate": float(np.mean([row["grade"]["all_fields_correct"] for row in selected])),
            "schema_rate": float(np.mean([row["grade"]["schema_exact"] for row in selected])),
            "median_latency_s": float(np.median([row["elapsed_s"] for row in selected])),
            "p95_latency_s": float(np.quantile([row["elapsed_s"] for row in selected], 0.95)),
            "errors": sum(row["finish_reason"] is None for row in selected),
            "truncations": sum(row["finish_reason"] == "length" for row in selected),
            "proxy_association": proxy_correlation(rows, model),
        }
        per_case = {case: [row for row in selected if row["case_id"] == case] for case in cases}
        summary["strict_workflow"][model] = float(np.mean([
            all(row["grade"]["all_fields_correct"] for row in per_case[case]) for case in cases
        ]))
        summary["proxy_association_pooled"][model] = proxy_correlation(rows, model)
        for kind in sorted({row["node_kind"] for row in rows}):
            stage_rows = [row for row in selected if row["node_kind"] == kind]
            summary["by_stage"].setdefault(kind, {})[model] = {
                "mean_field_accuracy": float(np.mean([row["actual_quality"] for row in stage_rows])),
                "all_fields_rate": float(np.mean([row["grade"]["all_fields_correct"] for row in stage_rows])),
                "median_latency_s": float(np.median([row["elapsed_s"] for row in stage_rows])),
                "p95_latency_s": float(np.quantile([row["elapsed_s"] for row in stage_rows], 0.95)),
                "proxy_association": proxy_correlation(rows, model, kind),
            }
            summary["proxy_association_by_stage"].setdefault(kind, {})[model] = summary["by_stage"][kind][model]["proxy_association"]
    differences = []
    strict_differences = []
    for case in cases:
        a = [row["actual_quality"] for row in rows if row["model"] == "1.7B" and row["case_id"] == case]
        b = [row["actual_quality"] for row in rows if row["model"] == "8B" and row["case_id"] == case]
        differences.append(float(np.mean(b) - np.mean(a)))
        strict_a = all(row["grade"]["all_fields_correct"] for row in rows if row["model"] == "1.7B" and row["case_id"] == case)
        strict_b = all(row["grade"]["all_fields_correct"] for row in rows if row["model"] == "8B" and row["case_id"] == case)
        strict_differences.append(float(strict_b) - float(strict_a))
    summary["paired_case_mean_quality"] = {
        "8B_minus_1.7B": float(np.mean(differences)),
        "case_cluster_95_ci": bootstrap_mean_ci(differences, SEED + 31),
        "wins_ties_losses": [int(sum(x > 0 for x in differences)), int(sum(x == 0 for x in differences)), int(sum(x < 0 for x in differences))],
    }
    summary["strict_workflow"]["8B_minus_1.7B"] = {
        "difference": float(np.mean(strict_differences)),
        "case_cluster_95_ci": bootstrap_mean_ci(strict_differences, SEED + 37),
        "wins_ties_losses": [int(sum(x > 0 for x in strict_differences)), int(sum(x == 0 for x in strict_differences)), int(sum(x < 0 for x in strict_differences))],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--helper", required=True)
    parser.add_argument("--e1-script", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "run":
        run(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
