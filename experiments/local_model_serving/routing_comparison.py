from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SEED = 20261021
BOOTSTRAPS = 5000
FAMILIES = ("flood", "traffic", "pump", "lighting")
DIFFICULTIES = ("low", "medium", "high")
REPLICATES = (11, 12, 13)
TRAIN_SEEDS = (2026, 2027, 2028)
TIMESTEPS = 60000
METHODS = ("UGQP-GNN", "DTODRL-joint")
ENDPOINT_ACTIONS = {"local": "1.7B", "edge_s": "8B"}


def uncertainty_band(value: float) -> str:
    return "low" if value < 0.40 else "medium" if value < 0.70 else "high"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_files(code_root: Path) -> dict[str, str]:
    paths = (
        "src/llm_offload_env/config.py",
        "src/llm_offload_env/serving.py",
        "src/llm_offload_env/env.py",
        "src/llm_offload_env/taskgen.py",
        "src/llm_offload_env/gnn_policy.py",
        "src/llm_offload_env/baselines.py",
        "src/train_ppo.py",
        "endpoint_profile.json",
    )
    return {name: digest(code_root / name) for name in paths}


def stable_endpoint_identity(identities: dict) -> dict:
    return {
        model: {
            key: identities[model].get(key)
            for key in ("id", "owned_by", "root", "max_model_len")
        }
        for model in sorted(identities)
    }


def configure_workflow_stages(base_items: list[dict], complexity: str) -> list[dict]:
    if complexity == "compact":
        selected = (0, 1, 5, 6, 7)
        base_items[5]["pred"] = [1]
        base_items[5]["task"] = "依据实际前驱n1中的事实以及规则和资源表，直接生成调度方案的全部字段。"
    elif complexity == "branched":
        selected = (0, 1, 3, 5, 6, 7)
        base_items[5]["pred"] = [1, 3]
        base_items[5]["task"] = "依据实际前驱n1中的事实和n3的资源判断，复核并生成完整调度方案。"
    elif complexity == "full_dag":
        selected = tuple(range(len(base_items)))
    else:
        raise ValueError(f"Unknown workflow complexity: {complexity}")

    original_to_active = {source: active for active, source in enumerate(selected)}
    active_items = []
    for source in selected:
        item = copy.deepcopy(base_items[source])
        item["source_stage_index"] = source
        item["pred"] = [original_to_active[parent] for parent in base_items[source]["pred"]]
        active_items.append(item)
    return active_items


def generate_cases(helper, e1_module, taskgen) -> list[dict]:
    cases = []
    serial = 61
    complexity_tiers = ("compact", "branched", "full_dag")
    for family in FAMILIES:
        for difficulty in DIFFICULTIES:
            for replicate in REPLICATES:
                facts = e1_module.make_facts(family, difficulty, replicate)
                resources = e1_module.make_resources(
                    family, difficulty, serial, random.Random(SEED + serial)
                )
                scene = helper.make_scene(family, f"E2{serial:02d}", facts, resources)
                base_items = helper.specs(scene)
                family_index = FAMILIES.index(family)
                difficulty_index = DIFFICULTIES.index(difficulty)
                replicate_index = REPLICATES.index(replicate)
                complexity = complexity_tiers[
                    (family_index + difficulty_index + replicate_index) % len(complexity_tiers)
                ]
                items = configure_workflow_stages(base_items, complexity)
                shape_rng = random.Random(SEED + serial * 100_003)
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
                    node.difficulty = uncertainty_band(float(uncertainty))
                    nodes.append(node)
                taskgen.compute_graph_features(nodes)
                deadline_factor = shape_rng.uniform(0.82, 1.22)
                cases.append({
                    "id": scene["id"], "family": family, "difficulty": difficulty,
                    "replicate": replicate, "scene": scene, "items": items,
                    "base_items": base_items, "workflow_complexity": complexity,
                    "nodes": [node.__dict__ for node in nodes],
                    "deadline_factor": deadline_factor,
                })
                serial += 1
    if len(cases) != 36 or len({case["id"] for case in cases}) != 36:
        raise RuntimeError("Expected 36 independent v5 workflows")
    counts = {tier: sum(case["workflow_complexity"] == tier for case in cases) for tier in complexity_tiers}
    if counts != {"compact": 12, "branched": 12, "full_dag": 12}:
        raise RuntimeError(f"Unbalanced workflow complexity schedule: {counts}")
    for family in FAMILIES:
        for difficulty in DIFFICULTIES:
            cell = [case for case in cases if case["family"] == family and case["difficulty"] == difficulty]
            if len(cell) != 3 or {case["workflow_complexity"] for case in cell} != set(complexity_tiers):
                raise RuntimeError(f"Complexity schedule is not balanced for {family}/{difficulty}")
    return cases


def prepare(args) -> None:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving existing E2 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    helper_path = Path(args.helper).resolve()
    e1_path = Path(args.e1_script).resolve()
    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root / "src"))
    helper = load_module(helper_path, "e2_readonly_helper")
    e1_module = load_module(e1_path, "e2_case_source")
    from llm_offload_env import taskgen

    cases = generate_cases(helper, e1_module, taskgen)
    endpoint_identities = e1_module.endpoint_models()
    profile_path = code_root / "endpoint_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("profile_id") != "v5_bridge_v1" or not profile.get("queue_capable"):
        raise RuntimeError("Missing or invalid v5 bridge profile")
    for model in ("1.7B", "8B"):
        if profile["models"][model]["n_calls"] != 192:
            raise RuntimeError(f"Unexpected calibration volume for {model}")
    pretrain = {
        "experiment": "v5 bridge-calibrated, queue-aware two-endpoint workflow routing",
        "status": "sealed_before_formal_evaluation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "independent_workflows": len(cases),
        "stratification": "4 event families x 3 predeclared factual-difficulty tiers x 3 fresh replicates (11, 12, 13), disjoint from E1/E1b, v4, and exploratory v5 replicates. Every family-by-difficulty cell contains one compact, one branched, and one full-DAG workflow; 12 cases per graph complexity.",
        "complexity_assignment": "For each case, complexity index = (family index + factual-difficulty index + replicate index) mod 3; replicate index follows frozen order (11, 12, 13). Assignment is independent of methods and outputs.",
        "workflow": "Three predeclared variants of the same validated semantic pipeline: compact 5-stage intent/extract/plan/verify/summarize; branched 6-stage intent/extract/resource-retrieve/plan/verify/summarize; full 8-stage intent/extract/two-retrieve/tool-call/plan/verify/summarize. Original stage prompts and typed scoring are retained; compact variants omit semantically redundant intermediate stages. Every variant is a topologically ordered DAG.",
        "actions": ENDPOINT_ACTIONS,
        "endpoint_identities": endpoint_identities,
        "decoding": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 256, "enable_thinking": False, "stream": True},
        "scoring": "Frozen typed-field scorer; report mean node field accuracy, per-node strict fields/schema, and all-node strict workflow rate. No LLM judge.",
        "primary_scope": "Two real Qwen3 endpoints on the RTX 5090 host. Formal calls are sequential; simulator queues represent exogenous per-endpoint workload and are not measured physical host queue delays.",
        "resource_metric": "Normalized model-work cost = sum((parameter_count / 1.7B) * (actual prompt_tokens + actual completion_tokens) / 1000); lower is less model work. This is not energy, currency, or measured GPU time.",
        "training": {
            "methods": ["UGQP-GNN", "DTODRL-joint"], "seeds": list(TRAIN_SEEDS),
            "timesteps_per_seed": TIMESTEPS, "n_envs": 4, "device": "cpu",
            "environment": {"profile_mode": "v5_bridge", "queue_mode": "bursty", "queue_scale": 1.5, "min_nodes": 8, "max_nodes": 8, "task_graph_mode": "bridge_eight", "arrival_rate": 0.7, "network_scale": 1.0, "deadline_scale": 1.55, "quality_propagation_strength": 1.5},
            "reward_weights": {"latency": 1.0, "cost": 0.2, "quality": 2.4, "deadline": 1.2, "balance": 0.18, "latency_normalizer_s": 5.0},
            "profile_rule": "Measured real endpoint field accuracy, full-call latency, and token usage joined by node kind and frozen uncertainty band from disjoint E1b cases; model work is normalized by parameter count. Nonzero bursty per-endpoint queues are enabled in training and evaluation.",
        },
        "analysis_unit": "Independent workflow; nodes and three training seeds are repeated measures, not independent cases.",
        "cases": cases,
        "profile_sha256": digest(profile_path),
        "external_source_sha256": {"helper": digest(helper_path), "e1_case_generator": digest(e1_path)},
        "code_source_sha256": source_files(code_root),
    }
    write_json(output / "pretrain_protocol.json", pretrain)
    (output / "pretrain_protocol.sha256").write_text(digest(output / "pretrain_protocol.json") + "\n", encoding="ascii")
    print(json.dumps({"output": str(output), "workflows": len(cases), "endpoint_identities": endpoint_identities}, ensure_ascii=False))


def check_pretrain_protocol(output: Path, code_root: Path) -> dict:
    path = output / "pretrain_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if digest(path) != (output / "pretrain_protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("Pretraining protocol hash mismatch")
    if digest(code_root / "endpoint_profile.json") != protocol["profile_sha256"]:
        raise RuntimeError("Calibrated profile changed after E2 task generation")
    if source_files(code_root) != protocol["code_source_sha256"]:
        raise RuntimeError("E2 code changed after pretraining protocol seal")
    return protocol


def train_policies(args) -> None:
    output = Path(args.output).resolve()
    code_root = Path(args.code_root).resolve()
    protocol = check_pretrain_protocol(output, code_root)
    sys.path.insert(0, str(code_root / "src"))
    from train_ppo import train

    checkpoint = code_root / "models" / "dtodrl_gat_pretrained.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    manifest = {"training_protocol_sha256": digest(output / "pretrain_protocol.json"), "policies": []}
    manifest_path = output / "training_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Preserving existing training manifest: {manifest_path}")
    for seed in protocol["training"]["seeds"]:
        for method, encoder, freeze in (("UGQP-GNN", "gnn", True), ("DTODRL-joint", "dtodrl", False)):
            model_dir = output / "models" / method.lower().replace("-", "_") / f"seed_{seed}"
            model_path = train(
                model_dir, protocol["training"]["timesteps_per_seed"], seed=seed,
                variant="full", arrival_rate=0.7, network_scale=1.0, deadline_scale=1.55,
                encoder=encoder, queue_mode="bursty", queue_scale=1.5, ent_coef=0.05,
                profile_mode="v5_bridge", n_envs=4, min_nodes=8, max_nodes=8,
                task_graph_mode="bridge_eight", quality_propagation_strength=1.5,
                dtodrl_checkpoint=checkpoint if encoder == "dtodrl" else None,
                dtodrl_freeze_gat=freeze, device="cpu",
            )
            manifest["policies"].append({
                "method": method, "seed": seed, "path": str(model_path.relative_to(output)),
                "sha256": digest(model_path),
            })
            print(f"trained {method} seed={seed} sha256={manifest['policies'][-1]['sha256']}", flush=True)
    write_json(manifest_path, manifest)


def seal(args) -> None:
    output = Path(args.output).resolve()
    code_root = Path(args.code_root).resolve()
    pretrain = check_pretrain_protocol(output, code_root)
    manifest = json.loads((output / "training_manifest.json").read_text(encoding="utf-8"))
    if manifest["training_protocol_sha256"] != digest(output / "pretrain_protocol.json"):
        raise RuntimeError("Training manifest refers to a different protocol")
    endpoint_identities = load_module(Path(args.e1_script).resolve(), "e2_endpoint_identity").endpoint_models()
    if stable_endpoint_identity(endpoint_identities) != stable_endpoint_identity(pretrain["endpoint_identities"]):
        raise RuntimeError("Endpoint identity changed after E2 protocol preparation")
    run_protocol = {
        **pretrain,
        "status": "sealed_before_real_e2_calls",
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint_identities_at_seal": endpoint_identities,
        "stable_endpoint_identity_at_seal": stable_endpoint_identity(endpoint_identities),
        "training_manifest_sha256": digest(output / "training_manifest.json"),
        "policy_sha256": manifest["policies"],
        "e2_runner_sha256": digest(Path(__file__).resolve()),
    }
    path = output / "protocol.json"
    if path.exists():
        raise FileExistsError(f"Preserving existing E2 run protocol: {path}")
    write_json(path, run_protocol)
    (output / "protocol.sha256").write_text(digest(path) + "\n", encoding="ascii")
    print(json.dumps({"protocol_sha256": digest(path), "policies": len(manifest["policies"])}, ensure_ascii=False))


def check_run_protocol(output: Path, code_root: Path, e1_path: Path) -> dict:
    path = output / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if digest(path) != (output / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("E2 run protocol hash mismatch")
    if protocol["e2_runner_sha256"] != digest(Path(__file__).resolve()):
        raise RuntimeError("E2 runner changed after protocol sealing")
    if protocol["code_source_sha256"] != source_files(code_root):
        raise RuntimeError("E2 source changed after protocol sealing")
    if protocol["profile_sha256"] != digest(code_root / "endpoint_profile.json"):
        raise RuntimeError("Calibrated profiles changed after protocol sealing")
    endpoint_identities = load_module(e1_path, "e2_live_endpoint_identity").endpoint_models()
    if stable_endpoint_identity(endpoint_identities) != protocol["stable_endpoint_identity_at_seal"]:
        raise RuntimeError("Live endpoint identity differs from the sealed protocol")
    for row in protocol["policy_sha256"]:
        model_path = output / row["path"]
        if digest(model_path) != row["sha256"]:
            raise RuntimeError(f"Policy checkpoint hash mismatch: {model_path}")
    return protocol


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(array, size=len(array), replace=True))) for _ in range(BOOTSTRAPS)]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def make_task(case: dict, taskgen, task_id: int):
    from llm_offload_env.taskgen import Task, Node

    nodes = [Node(**data) for data in case["nodes"]]
    nominal = sum(0.05 + node.input_tokens / 2800.0 + node.output_tokens / 115.0 for node in nodes)
    task = Task(task_id, {"flood": "emergency", "traffic": "traffic", "pump": "facility_inspection", "lighting": "municipal_service"}[case["family"]], nodes, nominal * case["deadline_factor"] * 1.55)
    return task


def capture_gpu() -> dict | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return {"raw": result.stdout.strip()}
    except (OSError, subprocess.SubprocessError):
        return None


def execute_one(case: dict, method: str, model, endpoint_map: dict, helper, e1_module,
                taskgen, seed: int, training_seed: int | None, output_path: Path) -> dict:
    from llm_offload_env.config import make_config, ACTION_TO_ID, ACTIONS
    from llm_offload_env.env import LLMAgentOffloadEnv
    from llm_offload_env.baselines import select_action

    cfg = make_config(
        seed=seed, profile_mode="v5_bridge", min_nodes=8, max_nodes=8,
        task_graph_mode="bridge_eight", queue_mode="bursty", queue_scale=1.5,
        arrival_rate=0.7, network_scale=1.0, deadline_scale=1.55,
    )
    env = LLMAgentOffloadEnv(cfg)
    task = make_task(case, taskgen, seed)
    observation, _ = env.reset(seed=seed, options={"task": task})
    outputs = {}
    base_items = case["base_items"]
    per_node = []
    started = time.perf_counter()
    for index, item in enumerate(case["items"]):
        source_index = item["source_stage_index"]
        if method == "GOA-DAG":
            action = select_action(env, "goa_dag", random.Random(seed + index))
        else:
            action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
            action = int(np.asarray(action).reshape(-1)[0])
        action_name = ACTIONS[action]
        model_name = endpoint_map[action_name]
        endpoint = e1_module.MODELS[model_name]
        prompt = helper.prompt_for(case["scene"], base_items, source_index, outputs)
        expected = item["expected"]
        if source_index == 6:
            expected = helper.conditional_verify(case["scene"], outputs)
        response = e1_module.stream_call(endpoint["url"], endpoint["served_name"], prompt, seed + index)
        grade = helper.grade(response["output"], expected)
        feedback = helper.contract_feedback(
            case["scene"], base_items, source_index,
            {**outputs, source_index: response["output"]}, response["output"],
        )
        outputs[source_index] = response["output"]
        prior_finish = max((env.completion[pred] for pred in item["pred"]), default=0.0)
        observation, _, terminated, _, _ = env.step(action)
        actual_latency = float(response["elapsed_s"])
        env.completion[index] = prior_finish + actual_latency
        env.realized_quality[index] = float(grade["field_accuracy"])
        env.node_results[-1]["quality"] = float(grade["field_accuracy"])
        env.node_results[-1]["base_quality"] = float(grade["field_accuracy"])
        env.node_results[-1]["latency"] = actual_latency
        env.node_results[-1]["finish_time"] = env.completion[index]
        observation = env._obs()
        row = {
            "case_id": case["id"], "family": case["family"], "difficulty": case["difficulty"],
            "workflow_complexity": case["workflow_complexity"],
            "method": method, "training_seed": training_seed, "node_index": index,
            "source_stage_index": source_index, "node_kind": item["kind"],
            "predecessors": item["pred"],
            "action": action_name, "endpoint": model_name, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "expected": expected, "output": response["output"], "grade": grade,
            "contract_feedback": feedback, "usage": response["usage"],
            "finish_reason": response["finish_reason"], "elapsed_s": actual_latency,
            "ttft_s": response["ttft_s"], "seed": seed + index,
            "normalized_model_work_cost": ((1.0 if model_name == "1.7B" else 8.0 / 1.7)
                * (int(response["usage"]["prompt_tokens"]) + int(response["usage"]["completion_tokens"])) / 1000.0),
            "actual_prompt_tokens": int(response["usage"]["prompt_tokens"]),
            "actual_completion_tokens": int(response["usage"]["completion_tokens"]),
            "node_difficulty_band": uncertainty_band(float(case["nodes"][index]["uncertainty"])),
        }
        append_jsonl(output_path, row)
        per_node.append(row)
        if bool(terminated) != (index == len(case["items"]) - 1):
            raise RuntimeError(f"Unexpected DAG termination state for {case['id']} at node {index}")
    wall = time.perf_counter() - started
    return {
        "case_id": case["id"], "family": case["family"], "difficulty": case["difficulty"],
        "workflow_complexity": case["workflow_complexity"],
        "node_count": len(case["items"]),
        "method": method, "training_seed": training_seed,
        "mean_node_field_accuracy": statistics.mean(row["grade"]["field_accuracy"] for row in per_node),
        "all_fields_correct": all(row["grade"]["all_fields_correct"] for row in per_node),
        "all_schema_correct": all(row["grade"]["schema_exact"] for row in per_node),
        "terminal_node_field_accuracy": per_node[-1]["grade"]["field_accuracy"],
        "model_usage": {name: sum(row["endpoint"] == name for row in per_node) for name in ("1.7B", "8B")},
        "actual_request_time_sum_s": sum(row["elapsed_s"] for row in per_node),
        "normalized_model_work_cost": sum(row["normalized_model_work_cost"] for row in per_node),
        "workflow_wall_time_s": wall,
        "model_errors": sum(row["finish_reason"] is None for row in per_node),
        "truncations": sum(row["finish_reason"] == "length" for row in per_node),
    }


def execute(args) -> None:
    output = Path(args.output).resolve()
    code_root = Path(args.code_root).resolve()
    e1_path = Path(args.e1_script).resolve()
    protocol = check_run_protocol(output, code_root, e1_path)
    output_path = output / ("calls_smoke.jsonl" if args.smoke else "calls.jsonl")
    workflow_path = output / ("workflows_smoke.jsonl" if args.smoke else "workflows.jsonl")
    if args.smoke:
        if output_path.exists() or workflow_path.exists():
            raise FileExistsError(f"Preserving existing E2 smoke output: {output_path}")
    else:
        if output_path.exists() != workflow_path.exists():
            raise RuntimeError("Only one formal E2 execution log exists; preserve it and use a fresh run directory")
        if not output_path.exists():
            smoke_calls = output / "calls_smoke.jsonl"
            smoke_workflows = output / "workflows_smoke.jsonl"
            if smoke_calls.exists() != smoke_workflows.exists():
                raise RuntimeError("Incomplete smoke outputs; preserve them and use a fresh run directory")
            if smoke_calls.exists():
                shutil.copyfile(smoke_calls, output_path)
                shutil.copyfile(smoke_workflows, workflow_path)
    sys.path.insert(0, str(code_root / "src"))
    helper = load_module(Path(args.helper).resolve(), "e2_execute_helper")
    e1_module = load_module(e1_path, "e2_execute_source")
    from llm_offload_env import taskgen
    from sb3_contrib import MaskablePPO

    endpoint_map = protocol["actions"]
    models = {}
    selections = []
    if args.smoke:
        selected_cases = protocol["cases"][:1]
        selections = [(method, TRAIN_SEEDS[0]) for method in METHODS]
    else:
        selected_cases = protocol["cases"]
        selections = [(method, seed) for method in ("UGQP-GNN", "DTODRL-joint") for seed in TRAIN_SEEDS]
    policy_manifest = {row["path"]: row for row in protocol["policy_sha256"]}
    for method, seed in selections:
        record = next(row for row in protocol["policy_sha256"] if row["method"] == method and row["seed"] == seed)
        model_path = output / record["path"]
        models[(method, seed)] = MaskablePPO.load(model_path, device="cpu")
    write_json(output / ("gpu_smoke.json" if args.smoke else "gpu_before.json"), capture_gpu())
    completed_workflows = 0
    for case in selected_cases:
        case_selections = list(selections)
        if not args.smoke:
            random.Random(SEED + case["replicate"]).shuffle(case_selections)
        for method, seed in case_selections:
            if not args.smoke and workflow_path.exists():
                completed_keys = {
                    (row["case_id"], row["method"], row["training_seed"])
                    for row in (json.loads(line) for line in workflow_path.read_text(encoding="utf-8").splitlines())
                }
                if (case["id"], method, seed) in completed_keys:
                    continue
            model = models[(method, seed)]
            record = execute_one(
                case, method, model, endpoint_map, helper, e1_module, taskgen,
                SEED + case["replicate"] * 1000, seed, output_path,
            )
            append_jsonl(workflow_path, record)
            completed_workflows += 1
            print(f"completed {completed_workflows} new workflows {case['id']} {method} seed={seed}", flush=True)
    write_json(output / ("gpu_smoke_after.json" if args.smoke else "gpu_after.json"), capture_gpu())
    expected_workflows = len(selected_cases) * len(selections)
    expected_calls = sum(len(case["items"]) for case in selected_cases) * len(selections)
    actual_calls = len(output_path.read_text(encoding="utf-8").splitlines())
    actual_workflows = len(workflow_path.read_text(encoding="utf-8").splitlines())
    if args.smoke and (actual_workflows != expected_workflows or actual_calls != expected_calls):
        raise RuntimeError(f"Smoke expected {expected_workflows} workflows/{expected_calls} calls; got {actual_workflows}/{actual_calls}")
    if not args.smoke and (actual_workflows != expected_workflows or actual_calls != expected_calls):
        raise RuntimeError(f"Formal E2 expected {expected_workflows} workflows/{expected_calls} calls; got {actual_workflows}/{actual_calls}")
    print(json.dumps({"workflows": actual_workflows, "calls": actual_calls, "output": str(output_path)}))


def analyze(args) -> None:
    output = Path(args.output).resolve()
    path = output / "workflows.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    call_rows = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    execution_count = len(TRAIN_SEEDS) * 2
    expected_runs = len(protocol["cases"]) * execution_count
    expected_calls = sum(len(case["items"]) for case in protocol["cases"]) * execution_count
    if len(rows) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} workflow runs, found {len(rows)}")
    if len(call_rows) != expected_calls:
        raise RuntimeError(f"Expected {expected_calls} node calls, found {len(call_rows)}")
    by_case_method = {}
    for method in METHODS:
        for case in sorted({row["case_id"] for row in rows}):
            selected = [row for row in rows if row["method"] == method and row["case_id"] == case]
            by_case_method[(case, method)] = {
                key: statistics.mean(float(row[key]) for row in selected)
                for key in ("mean_node_field_accuracy", "all_fields_correct", "all_schema_correct", "terminal_node_field_accuracy", "actual_request_time_sum_s", "normalized_model_work_cost", "workflow_wall_time_s", "model_errors", "truncations")
            }
            by_case_method[(case, method)]["model_usage_8B"] = statistics.mean(row["model_usage"]["8B"] / row["node_count"] for row in selected)
            by_case_method[(case, method)]["workflow_complexity"] = selected[0]["workflow_complexity"]
    case_ids = sorted({row["case_id"] for row in rows})
    summary = {"status": "completed", "independent_workflows": len(case_ids), "workflow_runs": len(rows), "node_calls": len(call_rows), "methods": {}, "complexity_strata": {}, "difficulty_strata": {}, "paired_case_differences": {}}
    metrics = ("mean_node_field_accuracy", "all_fields_correct", "all_schema_correct", "terminal_node_field_accuracy", "actual_request_time_sum_s", "normalized_model_work_cost", "workflow_wall_time_s", "model_usage_8B")
    for method in METHODS:
        values = {metric: [by_case_method[(case, method)][metric] for case in case_ids] for metric in metrics}
        summary["methods"][method] = {
            "mean_node_field_accuracy": statistics.mean(values["mean_node_field_accuracy"]),
            "all_fields_workflow_rate": statistics.mean(values["all_fields_correct"]),
            "all_schema_workflow_rate": statistics.mean(values["all_schema_correct"]),
            "terminal_node_field_accuracy": statistics.mean(values["terminal_node_field_accuracy"]),
            "mean_actual_request_time_sum_s": statistics.mean(values["actual_request_time_sum_s"]),
            "mean_normalized_model_work_cost": statistics.mean(values["normalized_model_work_cost"]),
            "median_workflow_wall_time_s": statistics.median(values["workflow_wall_time_s"]),
            "p95_workflow_wall_time_s": float(np.quantile(values["workflow_wall_time_s"], 0.95)),
            "mean_8B_node_share": statistics.mean(values["model_usage_8B"]),
            "mean_model_errors_per_workflow": statistics.mean([by_case_method[(case, method)]["model_errors"] for case in case_ids]),
            "mean_truncations_per_workflow": statistics.mean([by_case_method[(case, method)]["truncations"] for case in case_ids]),
        }
        summary["complexity_strata"][method] = {}
        for tier in ("compact", "branched", "full_dag"):
            tier_cases = [case for case in case_ids if by_case_method[(case, method)]["workflow_complexity"] == tier]
            summary["complexity_strata"][method][tier] = {
                "n_workflows": len(tier_cases),
                "mean_node_field_accuracy": statistics.mean(by_case_method[(case, method)]["mean_node_field_accuracy"] for case in tier_cases),
                "all_fields_workflow_rate": statistics.mean(by_case_method[(case, method)]["all_fields_correct"] for case in tier_cases),
                "median_workflow_wall_time_s": statistics.median(by_case_method[(case, method)]["workflow_wall_time_s"] for case in tier_cases),
                "mean_8B_node_share": statistics.mean(by_case_method[(case, method)]["model_usage_8B"] for case in tier_cases),
                "mean_normalized_model_work_cost": statistics.mean(by_case_method[(case, method)]["normalized_model_work_cost"] for case in tier_cases),
            }
        summary["difficulty_strata"][method] = {}
        case_difficulty = {row["id"]: row["difficulty"] for row in protocol["cases"]}
        for tier in DIFFICULTIES:
            tier_cases = [case for case in case_ids if case_difficulty[case] == tier]
            summary["difficulty_strata"][method][tier] = {
                "n_workflows": len(tier_cases),
                "mean_node_field_accuracy": statistics.mean(by_case_method[(case, method)]["mean_node_field_accuracy"] for case in tier_cases),
                "all_fields_workflow_rate": statistics.mean(by_case_method[(case, method)]["all_fields_correct"] for case in tier_cases),
                "mean_8B_node_share": statistics.mean(by_case_method[(case, method)]["model_usage_8B"] for case in tier_cases),
                "mean_normalized_model_work_cost": statistics.mean(by_case_method[(case, method)]["normalized_model_work_cost"] for case in tier_cases),
            }
    for comparator in ("DTODRL-joint",):
        paired = {}
        for metric in metrics:
            differences = [by_case_method[(case, "UGQP-GNN")][metric] - by_case_method[(case, comparator)][metric] for case in case_ids]
            paired[metric] = {
                "UGQP_minus_comparator": statistics.mean(differences),
                "workflow_cluster_bootstrap_95_ci": bootstrap_ci(differences, SEED + len(comparator) + len(metric)),
            }
        route_map = {}
        for row in call_rows:
            route_map.setdefault((row["case_id"], row["method"], row["training_seed"]), []).append(
                (row["node_index"], row["endpoint"])
            )
        disagreement_by_case = []
        for case in case_ids:
            seed_rates = []
            for train_seed in TRAIN_SEEDS:
                left = [endpoint for _, endpoint in sorted(route_map[(case, "UGQP-GNN", train_seed)])]
                right_seed = train_seed
                right = [endpoint for _, endpoint in sorted(route_map[(case, comparator, right_seed)])]
                seed_rates.append(sum(a != b for a, b in zip(left, right)) / max(len(left), 1))
            disagreement_by_case.append(statistics.mean(seed_rates))
        paired["route_action_disagreement_rate"] = statistics.mean(disagreement_by_case)
        summary["paired_case_differences"][f"UGQP-GNN_vs_{comparator}"] = paired
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "seal", "execute", "analyze"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--helper", required=True)
    parser.add_argument("--e1-script", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "train":
        train_policies(args)
    elif args.phase == "seal":
        seal(args)
    elif args.phase == "execute":
        execute(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
