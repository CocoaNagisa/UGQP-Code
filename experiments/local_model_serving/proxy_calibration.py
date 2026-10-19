"""Run a frozen paired quality-proxy calibration against two Qwen3 endpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np


MODELS = {
    "1.7B": {"url": "http://127.0.0.1:8102", "served_name": "qwen3-17"},
    "8B": {"url": "http://127.0.0.1:8104", "served_name": "qwen3-8b"},
}
GATE_REFERENCE = {
    "1.7B": {"semantic_mean": 0.6886574074074074, "strict_rate": 0.10416666666666667},
    "8B": {"semantic_mean": 0.8159722222222222, "strict_rate": 0.3541666666666667},
}
SEED = 20261019
BOOTSTRAPS = 2000
SYSTEM = "你是智慧城市LLM-Agent的结构化执行节点。禁止输出思考过程、解释和Markdown。严格按给定规则处理事实，只返回指定字段的一个JSON对象。"
TASKGEN_SCENARIO = {
    "flood": "emergency", "traffic": "traffic",
    "pump": "facility_inspection", "lighting": "municipal_service",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def endpoint_models() -> dict:
    result = {}
    for name, endpoint in MODELS.items():
        with urllib.request.urlopen(endpoint["url"] + "/v1/models", timeout=5) as response:
            body = json.load(response)
        matches = [row for row in body.get("data", []) if row.get("id") == endpoint["served_name"]]
        if len(matches) != 1:
            raise RuntimeError(f"Endpoint identity mismatch for {name}: {body}")
        result[name] = matches[0]
    return result


def make_resources(family: str, difficulty: str, serial: int, rng: random.Random) -> list[dict]:
    count = {"low": 2, "medium": 3, "high": 4}[difficulty]
    rows = []
    for idx in range(count):
        available = not (idx == 0 and difficulty != "low")
        row = {"id": f"{family[:1].upper()}{serial:02d}R{idx + 1}",
               "eta_min": 3 + ((idx * 3 + serial) % 11), "available": available}
        if family == "flood":
            row["flow"] = 16 if idx == count - 1 else 22 + ((serial + idx) % 9)
        elif family == "pump":
            row["qualified"] = idx > 0 or difficulty == "low"
        rows.append(row)
    if difficulty == "high":
        rows[0]["available"] = False
        if family == "flood":
            rows[1]["flow"] = 12
        elif family == "pump":
            rows[1]["qualified"] = False
    return rows


def make_facts(family: str, difficulty: str, replicate: int) -> dict:
    if family == "flood":
        depth = {"low": 5 + replicate, "medium": 9 + replicate, "high": 19 + replicate}[difficulty]
        return {"depth_cm": depth, "injured": False}
    if family == "traffic":
        queue = {"low": 492 + replicate, "medium": 499 + replicate, "high": 485 + 15 * replicate}[difficulty]
        return {"queue_m": queue, "injured": difficulty == "high"}
    if family == "pump":
        values = {
            "low": (2 + min(replicate, 2), 64 + replicate),
            "medium": (4 + replicate, 69 + replicate),
            "high": (7 + replicate, 79 + replicate),
        }
        vibration, temperature = values[difficulty]
        return {"vibration": vibration, "temperature": temperature}
    failed = {"low": 6 + replicate, "medium": 9 + replicate, "high": 13 + replicate}[difficulty]
    return {"failed_lights": failed, "exposed_wire": difficulty == "high"}


def generate_cases(helper, taskgen) -> list[dict]:
    rows = []
    feature_rng = random.Random(SEED)
    case_serial = 1
    for family in ("flood", "traffic", "pump", "lighting"):
        for difficulty in ("low", "medium", "high"):
            for replicate in range(1, 4):
                facts = make_facts(family, difficulty, replicate)
                resources = make_resources(family, difficulty, case_serial, random.Random(SEED + case_serial))
                scene = helper.make_scene(family, f"E1{case_serial:02d}", facts, resources)
                specs = helper.specs(scene)

                nodes = []
                for idx, kind in enumerate(("intent", "extract", "plan")):
                    input_tokens, output_tokens, uncertainty = taskgen.sample_node_shape(
                        feature_rng, kind, TASKGEN_SCENARIO[family]
                    )
                    privacy = feature_rng.choices([1, 2, 3], weights=[0.22, 0.46, 0.32])[0]
                    risk = min(1.0, max(0.0, uncertainty * feature_rng.uniform(0.75, 1.30)))
                    deadline_weight = feature_rng.uniform(0.65, 1.45)
                    nodes.append(taskgen.Node(idx, kind, input_tokens, output_tokens, uncertainty,
                                              privacy, risk, deadline_weight))
                taskgen.add_edge(nodes, 0, 1)
                taskgen.add_edge(nodes, 1, 2)
                taskgen.compute_graph_features(nodes)

                intent_prompt = helper.prompt_for(scene, specs, 0, {})
                extract_prompt = helper.prompt_for(
                    scene, specs, 1, {0: json.dumps({"event": family}, ensure_ascii=False)}
                )
                plan_prompt = "\n".join([
                    "智慧城市事件事实由前驱节点抽取，以下仅提供其实际输出，不提供原始事实。",
                    "前驱节点n2抽取结果：" + "{{EXTRACT_OUTPUT}}",
                    "规则：" + helper.rules(scene),
                    "资源表：" + json.dumps(scene["resources"], ensure_ascii=False),
                    "任务：根据前驱事实、规则和资源约束，独立形成完整调度方案。",
                    helper.schema(specs[5]["expected"]),
                ])
                rows.append({
                    "id": scene["id"], "family": family, "difficulty": difficulty,
                    "replicate": replicate, "scene": scene, "node_specs": specs,
                    "prompts": {"intent": intent_prompt, "extract": extract_prompt, "plan_template": plan_prompt},
                    "node_features": [node.__dict__ for node in nodes],
                })
                case_serial += 1
    if len(rows) != 36 or len({row["id"] for row in rows}) != 36:
        raise AssertionError("E1 must contain 36 unique cases")
    return rows


def profile_mapping(profiles) -> dict:
    mapping = {}
    for model, reference in GATE_REFERENCE.items():
        mapping[model] = min(
            profiles,
            key=lambda key: abs(profiles[key].quality_base - reference["semantic_mean"]),
        )
    if mapping != {"1.7B": "local", "8B": "edge_s"}:
        raise RuntimeError(f"Unexpected predeclared profile mapping: {mapping}")
    return mapping


def proxy_values(case: dict, model: str, profiles: dict, profile_map: dict,
                 propagation: float, quality_fn) -> list[float]:
    profile = profiles[profile_map[model]]
    nodes = case["node_features"]
    predicted = []
    for node in nodes:
        node_obj = type("ProxyNode", (), node)()
        base = float(quality_fn(node_obj, profile))
        predecessors = node_obj.predecessors
        previous_quality = float(np.mean([predicted[idx] for idx in predecessors])) if predecessors else 1.0
        penalty = propagation * (1.0 - previous_quality) * node_obj.downstream_risk * max(0.0, 1.0 - profile.capability)
        predicted.append(float(min(1.0, max(0.0, base - penalty))))
    return predicted


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def spearman_rho(left: list[float], right: list[float]) -> float:
    """Compute Spearman rho with average ranks for tied values."""
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

    x, y = average_ranks(left), average_ranks(right)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def prepare(args) -> None:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving existing E1 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    code_root = Path(args.code_root).resolve()
    helper_path = Path(args.helper).resolve()
    helper = load_module(helper_path, "v11_helpers_read_only")
    import sys
    sys.path.insert(0, str(code_root / "src"))
    from llm_offload_env.config import make_config
    from llm_offload_env.serving import default_profiles
    from llm_offload_env.taskgen import add_edge, compute_graph_features, sample_node_shape, Node

    taskgen = type("TaskGen", (), {"Node": Node, "add_edge": staticmethod(add_edge),
                                  "compute_graph_features": staticmethod(compute_graph_features),
                                  "sample_node_shape": staticmethod(sample_node_shape)})
    cases = generate_cases(helper, taskgen)
    profiles = default_profiles("tradeoff_v2")
    profile_map = profile_mapping(profiles)
    cfg = make_config()
    endpoint_data = endpoint_models()

    protocol = {
        "protocol": "E1 prospective calibration of the archived Eq. (7) quality proxy",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "sealed_before_generation",
        "cases": cases,
        "independent_case_count": 36,
        "stratification": "4 city-event families x 3 predeclared difficulty tiers x 3 cases per cell",
        "case_to_simulator_scenario": TASKGEN_SCENARIO,
        "node_chain": ["intent", "extract", "plan"],
        "models": MODELS,
        "endpoint_identities": endpoint_data,
        "independent_endpoint_gate_reference": GATE_REFERENCE,
        "profile_mapping_rule": "For quality-only calibration, map each endpoint to the archived tradeoff_v2 profile whose quality_base is closest to the independent 48-case semantic mean; freeze before E1 scoring. This is not a model-size-to-action mapping and is not used to claim physical-site equivalence.",
        "profile_mapping": profile_map,
        "decoding": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": SEED,
                     "max_tokens": 256, "enable_thinking": False, "stream": True},
        "quality_scoring": "The archived typed exact-field score; exact JSON schema and all-fields-correct reported separately. No LLM judge.",
        "proxy_scoring": "Original tradeoff_v2 quality terms and archived quality_propagation_strength, applied recursively to the frozen 3-node DAG using only proxy predecessor values. No E1 coefficient fitting.",
        "primary_analysis": "Per-endpoint Spearman correlation of node-level Eq. (7) proxy and actual typed field accuracy; 95% percentile CI by resampling 36 whole cases (2000 bootstrap replicates). Also report MAE/RMSE, paired endpoint quality difference and latency.",
        "not_used_for_tuning": ["E1 cases", "difficulty assignments", "proxy parameters", "E2 tasks", "reward weights", "route outcomes"],
        "source_hashes": {
            "script": digest(Path(__file__).resolve()),
            "helper": digest(helper_path),
            "config_py": digest(code_root / "src/llm_offload_env/config.py"),
            "serving_py": digest(code_root / "src/llm_offload_env/serving.py"),
            "taskgen_py": digest(code_root / "src/llm_offload_env/taskgen.py"),
        },
        "quality_propagation_strength": cfg.quality_propagation_strength,
    }
    write_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(digest(output / "protocol.json") + "\n", encoding="ascii")
    print(json.dumps({"sealed": str(output), "cases": len(cases), "endpoints": endpoint_data,
                      "profile_mapping": profile_map}, ensure_ascii=False), flush=True)


def stream_call(url: str, model: str, prompt: str, seed: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": seed, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(url + "/v1/chat/completions",
                                     data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    content, usage, finish = [], {}, None
    ttft = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                part = choice.get("delta", {}).get("content")
                if part:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    content.append(part)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    return {"output": "".join(content), "usage": usage, "finish_reason": finish,
            "elapsed_s": time.perf_counter() - started, "ttft_s": ttft, "request": payload}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def check_seal(output: Path, helper_path: Path, code_root: Path) -> dict:
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if digest(protocol_path) != (output / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("Sealed E1 protocol hash mismatch")
    hashes = protocol["source_hashes"]
    current = {
        "script": digest(Path(__file__).resolve()), "helper": digest(helper_path),
        "config_py": digest(code_root / "src/llm_offload_env/config.py"),
        "serving_py": digest(code_root / "src/llm_offload_env/serving.py"),
        "taskgen_py": digest(code_root / "src/llm_offload_env/taskgen.py"),
    }
    if current != hashes:
        raise RuntimeError("Sealed E1 source hash mismatch")
    return protocol


def run(args) -> None:
    output = Path(args.output).resolve()
    helper_path = Path(args.helper).resolve()
    code_root = Path(args.code_root).resolve()
    protocol = check_seal(output, helper_path, code_root)
    log = output / "calls.jsonl"
    if log.exists():
        raise FileExistsError(f"Preserving existing call log: {log}")
    helper = load_module(helper_path, "v11_helpers_read_only_run")
    sys_path = str(code_root / "src")
    import sys
    sys.path.insert(0, sys_path)
    from llm_offload_env.config import make_config
    from llm_offload_env.serving import default_profiles, quality
    cfg = make_config()
    profiles = default_profiles("tradeoff_v2")
    profile_map = protocol["profile_mapping"]

    for case_index, case in enumerate(protocol["cases"]):
        order = ("1.7B", "8B") if case_index % 2 == 0 else ("8B", "1.7B")
        for model_name in order:
            endpoint = protocol["models"][model_name]
            actual_predecessors: dict[int, str] = {}
            model_proxy = proxy_values(case, model_name, profiles, profile_map,
                                       cfg.quality_propagation_strength, quality)
            specs = case["node_specs"]
            prompts = case["prompts"]
            for node_index, kind in enumerate(("intent", "extract", "plan")):
                if kind == "plan":
                    prompt = prompts["plan_template"].replace("{{EXTRACT_OUTPUT}}", actual_predecessors[1])
                else:
                    prompt = prompts[kind]
                if kind == "extract":
                    prompt = prompt.replace(json.dumps({"event": case["family"]}, ensure_ascii=False),
                                            actual_predecessors[0])
                result = stream_call(endpoint["url"], endpoint["served_name"], prompt, SEED)
                expected = specs[{"intent": 0, "extract": 1, "plan": 5}[kind]]["expected"]
                grade = helper.grade(result["output"], expected)
                row = {
                    "case_id": case["id"], "family": case["family"], "difficulty": case["difficulty"],
                    "model": model_name, "served_name": endpoint["served_name"], "node": kind,
                    "node_index": node_index, "node_features": case["node_features"][node_index],
                    "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "expected": expected, "output": result["output"], "grade": grade,
                    "actual_quality": grade["field_accuracy"], "proxy_quality": model_proxy[node_index],
                    "usage": result["usage"], "finish_reason": result["finish_reason"],
                    "elapsed_s": result["elapsed_s"], "ttft_s": result["ttft_s"],
                    "seed": SEED, "error": None,
                }
                append_jsonl(log, row)
                actual_predecessors[node_index] = result["output"]
        print(f"completed {case_index + 1}/36 {case['id']}", flush=True)
    analyze(args)


def cluster_spearman(rows: list[dict], model: str, replicates: int = BOOTSTRAPS) -> dict:
    selected = [row for row in rows if row["model"] == model]
    cases = sorted({row["case_id"] for row in selected})
    point = spearman_rho([r["proxy_quality"] for r in selected],
                         [r["actual_quality"] for r in selected])
    rng = np.random.default_rng(SEED + (17 if model == "1.7B" else 29))
    by_case = {case: [row for row in selected if row["case_id"] == case] for case in cases}
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        sample_rows = [row for case in sampled for row in by_case[case]]
        value = spearman_rho([r["proxy_quality"] for r in sample_rows],
                             [r["actual_quality"] for r in sample_rows])
        if math.isfinite(value):
            draws.append(float(value))
    return {"n_cases": len(cases), "n_node_observations": len(selected), "rho": point,
            "case_cluster_bootstrap_95_ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "mae": float(np.mean([abs(r["proxy_quality"] - r["actual_quality"]) for r in selected])),
            "rmse": float(np.sqrt(np.mean([(r["proxy_quality"] - r["actual_quality"]) ** 2 for r in selected])))}


def analyze(args) -> None:
    output = Path(args.output).resolve()
    rows = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 216:
        raise RuntimeError(f"Expected 216 completed E1 calls, found {len(rows)}")
    summary = {"protocol": "E1 prospective calibration of archived Eq. (7) proxy", "status": "completed",
               "n_calls": len(rows), "n_independent_cases": 36, "models": {}, "paired_quality": {},
               "by_node": {}, "by_difficulty": {}, "endpoint_latency": {}}
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        summary["models"][model] = {
            "n": len(subset), "mean_field_accuracy": float(np.mean([r["actual_quality"] for r in subset])),
            "strict_all_fields_rate": float(np.mean([r["grade"]["all_fields_correct"] for r in subset])),
            "schema_rate": float(np.mean([r["grade"]["schema_exact"] for r in subset])),
            "error_count": sum(r["error"] is not None for r in subset),
            "truncation_count": sum(r["finish_reason"] == "length" for r in subset),
            "mean_proxy_quality": float(np.mean([r["proxy_quality"] for r in subset])),
            "latency_median_s": float(np.median([r["elapsed_s"] for r in subset])),
            "latency_p95_s": float(np.quantile([r["elapsed_s"] for r in subset], .95)),
            "output_tokens_mean": float(np.mean([r.get("usage", {}).get("completion_tokens", 0) for r in subset])),
            "proxy_association": cluster_spearman(rows, model),
        }
    per_case = {}
    for case_id in sorted({r["case_id"] for r in rows}):
        per_case[case_id] = {
            model: float(np.mean([r["actual_quality"] for r in rows if r["case_id"] == case_id and r["model"] == model]))
            for model in MODELS
        }
    deltas = [value["8B"] - value["1.7B"] for value in per_case.values()]
    rng = np.random.default_rng(SEED + 41)
    bootstrap = [float(np.mean(rng.choice(deltas, size=len(deltas), replace=True))) for _ in range(BOOTSTRAPS)]
    summary["paired_quality"] = {
        "mean_case_level_delta_8B_minus_1.7B": float(np.mean(deltas)),
        "case_cluster_bootstrap_95_ci": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
        "case_wins_ties_losses": [sum(x > 0 for x in deltas), sum(x == 0 for x in deltas), sum(x < 0 for x in deltas)],
    }
    for key in ("node", "difficulty"):
        values = sorted({r[key] for r in rows})
        summary["by_node" if key == "node" else "by_difficulty"] = {
            value: {model: float(np.mean([r["actual_quality"] for r in rows if r[key] == value and r["model"] == model]))
                    for model in MODELS} for value in values
        }
    for model in MODELS:
        times = [r["elapsed_s"] for r in rows if r["model"] == model]
        summary["endpoint_latency"][model] = {
            "median_s": float(np.median(times)), "p95_s": float(np.quantile(times, .95)),
            "mean_s": float(np.mean(times)),
        }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--helper", required=True)
    parser.add_argument("--code-root", required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "run":
        run(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
