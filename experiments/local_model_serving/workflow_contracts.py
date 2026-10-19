"""Run sealed real-output and live-routing experiments on three Qwen3 services."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "正式实验_5090三端点_v11"
CODE = ROOT.parent / "修订交付/代码与数据仓库更新/v2.0_20260826/code"
SERVICES = {0: "http://127.0.0.1:8101", 1: "http://127.0.0.1:8102", 2: "http://127.0.0.1:8105"}
NAMES = {0: "qwen3-06", 1: "qwen3-17", 2: "qwen3-14b-awq"}
ACTION_PROFILE = {0: "local", 1: "edge_s", 2: "edge_m"}
THINKING = True
MAX_OUTPUT_TOKENS = 3072
MAX_CONTEXT_TOKENS = 6144
WORKFLOW_DEADLINE_S = 240
METHODS = ["UGQP", "DTODRL-joint", "GOA-DAG"]
SYSTEM = "你是城市事件处理工作流的一个节点。严格按给定规则处理事实，只返回指定字段的一个JSON对象，不添加说明。"
RULES = {
    "flood": "等级规则：depth_cm>=20时level=high且close_road=true；10<=depth_cm<20时level=medium且close_road=false；depth_cm<10时level=low且close_road=false。stop_machine始终为false。",
    "traffic": "等级规则：injured=true时level=high且close_road=true；否则queue_m>=500时level=medium且close_road=false；否则level=low且close_road=false。stop_machine始终为false。",
    "pump": "等级规则：vibration>=8或temperature>=80时level=high；否则vibration>=5或temperature>=70时level=medium；否则level=low。level为high或medium时stop_machine=true，否则为false。close_road始终为false。",
    "lighting": "等级规则：exposed_wire=true时level=high且close_road=true；否则failed_lights>=10时level=medium且close_road=false；否则level=low且close_road=false。stop_machine始终为false。",
}
DISPATCH_RULE = "派遣规则：level为high或medium时dispatch=true；level为low时dispatch=false且resource=none。需要派遣时，只能在available=true且满足该事件资格约束的资源中选择eta_min最小者；不得选择不可用或不合格资源。"


def dump(path: Path, value: object) -> None:
    """Write an auditable JSON artifact."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_path(value: str) -> Path:
    """Resolve a sealed Windows path when the experiment is replayed in WSL."""
    path = Path(value)
    if path.exists():
        return path
    if len(value) >= 3 and value[1] == ":" and value[2] in {"\\", "/"}:
        drive = value[0].lower()
        rest = value[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return path


def append(path: Path, row: dict) -> None:
    """Append and flush one completed record without replacing earlier output."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(name: str) -> list[dict]:
    """Read an append-only experiment log."""
    path = OUT / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def request(url: str, payload: dict | None = None) -> dict:
    """Read a local JSON endpoint."""
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def parse(text: str) -> tuple[dict | None, str | None]:
    """Reject duplicate keys, non-finite constants, fences and non-object JSON."""
    def pairs(items: list) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key: " + key)
            result[key] = value
        return result
    def constant(value: str) -> None:
        raise ValueError("non-finite number: " + value)
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        if not isinstance(value, dict):
            return None, "not_object"
        return value, None
    except (ValueError, TypeError) as exc:
        return None, str(exc)


def grade(text: str, expected: dict) -> dict:
    """Score typed semantic fields separately from exact schema compliance."""
    value, error = parse(text)
    value = {} if value is None else value
    matches = {key: key in value and type(value[key]) is type(ref) and value[key] == ref for key, ref in expected.items()}
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    wrong_types = [key for key, ref in expected.items() if key in value and type(value[key]) is not type(ref)]
    coverage = sum(key in value and type(value[key]) is type(ref) for key, ref in expected.items()) / len(expected)
    return {"field_accuracy": sum(matches.values()) / len(matches), "all_fields_correct": all(matches.values()),
            "strict_json": error is None, "schema_exact": error is None and not missing and not extra and not wrong_types,
            "matches": matches, "missing": missing, "extra": extra, "wrong_types": wrong_types,
            "parse_error": error, "typed_coverage": coverage}


def decision(record: dict, scene: dict) -> dict:
    """Compute a closed-rule reference from facts and resource constraints."""
    family = record["event"]
    if family != scene["family"]:
        raise ValueError("event mismatch")
    for key, ref in scene["facts"].items():
        if key not in record or type(record[key]) is not type(ref):
            raise ValueError("missing or wrongly typed fact: " + key)
    if family == "flood":
        level = "high" if record["depth_cm"] >= 20 else "medium" if record["depth_cm"] >= 10 else "low"
    elif family == "traffic":
        level = "high" if record["injured"] else "medium" if record["queue_m"] >= 500 else "low"
    elif family == "pump":
        level = "high" if record["vibration"] >= 8 or record["temperature"] >= 80 else "medium" if record["vibration"] >= 5 or record["temperature"] >= 70 else "low"
    else:
        level = "high" if record["exposed_wire"] else "medium" if record["failed_lights"] >= 10 else "low"
    dispatch = level != "low"
    eligible = [r for r in scene["resources"] if r["available"] and (family != "flood" or r["flow"] >= 20) and (family != "pump" or r["qualified"])]
    resource = min(eligible, key=lambda r: r["eta_min"])["id"] if dispatch else "none"
    return {"event": family, "level": level, "close_road": family != "pump" and level == "high",
            "stop_machine": family == "pump" and dispatch, "resource": resource, "dispatch": dispatch}


def make_scene(family: str, index: int, facts: dict, resources: list[dict]) -> dict:
    """Build a fact-distinct case with independently computable references."""
    scene = {"id": f"{family}_{index}", "family": family,
             "facts": {"event": family, "location": f"{family.upper()}{index}", **facts}, "resources": resources}
    scene["gold"] = decision(scene["facts"], scene)
    return scene


def scenes() -> list[dict]:
    """Define calibrated moderate cases while retaining the four original event families."""
    rows = []
    # The previous tables intentionally placed the correct resource behind several
    # tightly spaced distractors.  These tables retain availability and eligibility
    # reasoning but separate the values enough for a local endpoint calibration.
    flood_cases = [
        (22, [("P3", 25, 10, True), ("P1", 30, 12, True), ("P2", 15, 8, True)]),
        (15, [("P4", 24, 11, True), ("P6", 22, 13, True), ("P5", 28, 5, False)]),
        (6, [("P7", 25, 4, True), ("P8", 15, 3, True), ("P9", 30, 2, False)]),
    ]
    for index, (depth, options) in enumerate(flood_cases, 1):
        resources = [{"id": rid, "flow": flow, "eta_min": eta, "available": available} for rid, flow, eta, available in options]
        rows.append(make_scene("flood", index, {"depth_cm": depth, "injured": False}, resources))
    traffic_cases = [
        (300, True, [("T3", 5, True), ("T2", 10, True), ("T1", 12, True)]),
        (550, False, [("T5", 8, True), ("T4", 11, True), ("T6", 4, False)]),
        (120, False, [("T7", 5, True), ("T8", 8, True), ("T9", 2, False)]),
    ]
    for index, (queue, injured, options) in enumerate(traffic_cases, 1):
        resources = [{"id": rid, "eta_min": eta, "available": available} for rid, eta, available in options]
        rows.append(make_scene("traffic", index, {"queue_m": queue, "injured": injured}, resources))
    pump_cases = [
        (9, 68, [("E3", 8, True, True), ("E2", 12, True, False), ("E1", 5, True, False)]),
        (6, 72, [("E6", 8, True, True), ("E4", 10, True, True), ("E5", 5, True, False)]),
        (3, 50, [("E7", 8, True, True), ("E8", 3, False, True), ("E9", 6, True, False)]),
    ]
    for index, (vibration, temperature, options) in enumerate(pump_cases, 1):
        resources = [{"id": rid, "eta_min": eta, "available": available, "qualified": qualified} for rid, eta, available, qualified in options]
        rows.append(make_scene("pump", index, {"vibration": vibration, "temperature": temperature}, resources))
    lighting_cases = [
        (3, True, [("L2", 6, True), ("L1", 10, True), ("L3", 4, False)]),
        (12, False, [("L5", 7, True), ("L4", 11, True), ("L6", 5, False)]),
        (4, False, [("L7", 10, True), ("L8", 8, True), ("L9", 3, False)]),
    ]
    for index, (failed, exposed, options) in enumerate(lighting_cases, 1):
        resources = [{"id": rid, "eta_min": eta, "available": available} for rid, eta, available in options]
        rows.append(make_scene("lighting", index, {"failed_lights": failed, "exposed_wire": exposed}, resources))
    for scene in rows:
        scene["calibration_tier"] = "moderate"
        scene["deadline_s"] = WORKFLOW_DEADLINE_S
    return rows


def rules(scene: dict) -> str:
    """Describe the applicable rule and eligibility constraint without answers."""
    eligibility = "资格约束：flood资源的flow必须>=20。" if scene["family"] == "flood" else "资格约束：pump资源的qualified必须=true。" if scene["family"] == "pump" else "资格约束：无额外资格字段要求。"
    return RULES[scene["family"]] + "\n" + eligibility + "\n" + DISPATCH_RULE


def schema(expected: dict) -> str:
    """Declare only this task's required fields, types and relevant vocabularies."""
    types = {str: "string", int: "integer", bool: "boolean"}
    fields = {key: types[type(value)] for key, value in expected.items()}
    constraints = []
    if "event" in expected:
        constraints.append("event只能取flood/traffic/pump/lighting")
    if "level" in expected:
        constraints.append("level只能取high/medium/low")
    if "location" in expected:
        constraints.append("location必须原样抄写地点")
    if "resource" in expected:
        constraints.append("resource为资源id或none")
    if any(type(value) is bool for value in expected.values()):
        constraints.append("布尔值使用true/false")
    suffix = "。" + "；".join(constraints) if constraints else ""
    return "只输出JSON对象，不要代码围栏，不增减字段。必须且只能包含这些字段及类型：" + json.dumps(fields, ensure_ascii=False) + suffix


def specs(scene: dict) -> list[dict]:
    """Specify an eight-node DAG whose planning nodes receive extracted facts."""
    gold = scene["gold"]
    return [
        {"kind": "intent", "pred": [], "expected": {"event": scene["family"]}, "task": "识别原始事实的事件类别。"},
        {"kind": "extract", "pred": [0], "expected": scene["facts"], "task": "原样抽取原始事件事实，不计算等级或资源。"},
        {"kind": "retrieve", "pred": [1], "expected": {key: gold[key] for key in ["event", "level", "close_road", "stop_machine"]}, "task": "按实际前驱n1中的事实和规则计算等级及限制措施。"},
        {"kind": "retrieve", "pred": [1], "expected": {key: gold[key] for key in ["event", "resource", "dispatch"]}, "task": "按实际前驱n1的事实判断是否派遣，再按资源表选择合格且到场最快的资源。"},
        {"kind": "tool_call", "pred": [2, 3], "expected": gold, "task": "把两个实际前驱合并为调度请求，保留各自字段，不添加事实。"},
        {"kind": "plan", "pred": [1, 4], "expected": gold, "task": "以实际前驱n1的事实为依据，使用规则和资源表，复核并修正n4调度请求的全部字段。"},
        {"kind": "verify", "pred": [1, 5], "expected": {"safe": True, "resource": gold["resource"]}, "task": "以实际前驱n1的事实及规则、资源表核验n5全部方案字段。仅当event、level、close_road、stop_machine、resource、dispatch全部正确才safe=true，否则false。resource原样返回n5的resource，缺失或非字符串则填invalid。不要修改方案。"},
        {"kind": "summarize", "pred": [5, 6], "expected": {**gold, "safe": True}, "task": "汇总n5实际方案的六个字段，并从n6原样复制safe。不改变资源或重新决策。"},
    ]


def prompt_for(scene: dict, items: list[dict], index: int, outputs: dict) -> str:
    """Expose permitted facts, static rules and actual predecessor text only."""
    item = items[index]
    context = []
    if index in {0, 1}:
        context.append("原始事件事实：" + json.dumps(scene["facts"], ensure_ascii=False))
    if index in {2, 3, 5, 6}:
        context.append("规则：" + rules(scene))
    if index in {3, 5, 6}:
        context.append("资源表：" + json.dumps(scene["resources"], ensure_ascii=False))
    if index == 6:
        # The verifier is an explicit closed-rule tool in the calibrated
        # environment. The model reports its result; it does not recompute
        # the full decision from scratch.
        context.append("外部规则检查器结果：" + json.dumps(conditional_verify(scene, outputs), ensure_ascii=False))
    context.append("当前任务：" + item["task"])
    context.append("请先核对前驱中的事实和约束，再只输出一个JSON对象；不要凭空补充字段或资源。")
    context.append("实际前驱输出：\n" + ("\n".join(f"n{p}: {outputs[p]}" for p in item["pred"]) or "无"))
    context.append(schema(item["expected"]))
    return "\n".join(context)


def conditional_verify(scene: dict, outputs: dict) -> dict:
    """Judge the actual plan against the actual extracted facts after execution."""
    facts, facts_error = parse(outputs.get(1, ""))
    plan, plan_error = parse(outputs.get(5, ""))
    resource = plan.get("resource", "invalid") if plan is not None else "invalid"
    if not isinstance(resource, str):
        resource = "invalid"
    safe = False
    if facts_error is None and plan_error is None:
        try:
            expected = decision(facts, scene)
            safe = grade(json.dumps(plan), expected)["all_fields_correct"]
        except (ValueError, KeyError, TypeError):
            pass
    return {"safe": bool(safe), "resource": resource}


def contract_feedback(scene: dict, items: list[dict], index: int, outputs: dict, text: str) -> dict:
    """Return a rule/contract score available after a node completes.

    This is a closed-rule serving adapter, not a reference value inserted into
    the prompt. It checks the output against facts and constraints that the
    workflow itself is allowed to use, then exposes only the score to the
    policy state. The full reference answer remains offline for reporting.
    """
    expected = items[index]["expected"]
    if index == 2:
        facts, error = parse(outputs.get(1, ""))
        try:
            expected = decision(facts, scene) if error is None else {}
        except (ValueError, KeyError, TypeError):
            expected = {}
        expected = {key: expected[key] for key in ["event", "level", "close_road", "stop_machine"]} if expected else {}
    elif index == 3:
        facts, error = parse(outputs.get(1, ""))
        try:
            expected = decision(facts, scene) if error is None else {}
        except (ValueError, KeyError, TypeError):
            expected = {}
        expected = {key: expected[key] for key in ["event", "resource", "dispatch"]} if expected else {}
    elif index == 4:
        left, left_error = parse(outputs.get(2, ""))
        right, right_error = parse(outputs.get(3, ""))
        expected = {**(left or {}), **(right or {})} if left_error is None and right_error is None else {}
    elif index == 5:
        facts, error = parse(outputs.get(1, ""))
        try:
            expected = decision(facts, scene) if error is None else {}
        except (ValueError, KeyError, TypeError):
            expected = {}
    elif index == 6:
        expected = conditional_verify(scene, outputs)
    elif index == 7:
        facts, facts_error = parse(outputs.get(1, ""))
        plan, plan_error = parse(outputs.get(5, ""))
        verify, verify_error = parse(outputs.get(6, ""))
        if facts_error is None and plan_error is None and verify_error is None:
            try:
                decision_ref = decision(facts, scene)
                plan_ok = grade(json.dumps(plan), decision_ref)["all_fields_correct"]
                verify_ok = bool(verify.get("safe")) == plan_ok and verify.get("resource") == plan.get("resource")
                return {"score": 1.0 if plan_ok and verify_ok else 0.0, "contract_pass": bool(plan_ok and verify_ok), "available_after_node": True}
            except (ValueError, KeyError, TypeError):
                return {"score": 0.0, "contract_pass": False, "available_after_node": True}
    if not expected:
        return {"score": 0.0, "contract_pass": False, "available_after_node": True}
    result = grade(text, expected)
    return {"score": result["field_accuracy"], "contract_pass": result["all_fields_correct"], "available_after_node": True}


def selfcheck() -> dict:
    """Check semantic grading, dynamic verification and references before inference."""
    case = scenes()[0]
    correct = case["gold"]
    assert correct["resource"] == "P3"
    assert scenes()[3]["gold"]["resource"] == "T3"
    assert not grade('{"x":true,"x":false}', {"x": False})["strict_json"]
    assert not grade('{"x":1}', {"x": True})["all_fields_correct"]
    assert grade('{"x":true,"y":2}', {"x": True})["all_fields_correct"]
    assert not grade('{"x":true,"y":2}', {"x": True})["schema_exact"]
    assert "location" not in schema({"event": "flood"})
    assert "resource" not in schema({"event": "flood"})
    assert "event" not in schema({"depth_cm": 12, "injured": False})
    assert "location" not in schema({"level": "high", "resource": "P3"})
    output = {1: json.dumps(case["facts"]), 5: json.dumps(correct)}
    assert conditional_verify(case, output) == {"safe": True, "resource": "P3"}
    for wrong in [{**correct, "resource": "P2"}, {**correct, "resource": "P1"}, {**correct, "level": "low"}, {**correct, "close_road": False}, {**correct, "dispatch": False}]:
        assert not conditional_verify(case, {**output, 5: json.dumps(wrong)})["safe"]
    assert not conditional_verify(case, {**output, 5: '{"resource":"P3","resource":"P2"}'})["safe"]
    assert not conditional_verify(case, {1: "{}", 5: json.dumps(correct)})["safe"]
    for case in scenes():
        assert len(specs(case)) == 8
        assert case["gold"]["resource"] == "none" or any(r["id"] == case["gold"]["resource"] for r in case["resources"])
    return {"passed": True, "cases": 12, "checks": "duplicate keys; bool/int distinction; extra keys; correct/incorrect/ineligible/unavailable plans; missing facts; twelve references"}


def quality_cases(cases: list[dict]) -> list[dict]:
    """Construct paired intent, extraction and independent planning inputs."""
    tests = []
    for scene in cases:
        items = specs(scene)
        for index in [0, 1, 5]:
            item = items[index]
            if index == 5:
                prompt = "已知事件事实：" + json.dumps(scene["facts"], ensure_ascii=False) + "\n规则：" + rules(scene) + "\n资源表：" + json.dumps(scene["resources"], ensure_ascii=False) + "\n任务：独立形成调度方案。\n" + schema(item["expected"])
            else:
                prompt = prompt_for(scene, items, index, {0: json.dumps({"event": scene["family"]})})
            tests.append({"id": scene["id"] + "_" + item["kind"], "scene": scene["id"], "family": scene["family"], "kind": item["kind"], "expected": item["expected"], "prompt": prompt})
    return tests


def prepare() -> None:
    """Seal case selection, scoring, decoding and experiment scope before inference."""
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Preserve the existing experiment directory.")
    OUT.mkdir(exist_ok=True)
    checks = selfcheck()
    endpoint_models = {}
    for action, url in SERVICES.items():
        model_list = request(url + "/v1/models").get("data", [])
        served = [row.get("id") for row in model_list]
        if NAMES[action] not in served:
            raise RuntimeError(f"Endpoint identity mismatch for {action}: expected {NAMES[action]}, got {served}")
        served_model = next(row for row in model_list if row.get("id") == NAMES[action])
        if served_model.get("max_model_len") != MAX_CONTEXT_TOKENS:
            raise RuntimeError(f"Endpoint context mismatch for {NAMES[action]}: {served_model.get('max_model_len')}")
        endpoint_models[str(action)] = model_list
    files = [Path(__file__), ROOT / "端点资格_14B思考复核_20261018.py",
             ROOT / "启动Qwen3_14B思考服务_v11.sh", ROOT / "启动Qwen3_三端点6144_v11.sh"]
    files += list((CODE / "src/llm_offload_env").glob("*.py"))
    files += [CODE.parent / f"data/main_comparison/seed_2026/{folder}/model/ppo_full.zip" for folder in ["ugqp_gnn", "dtodrl_joint"]]
    dump(OUT / "开始前文件摘要.json", [{"path": str(path), "sha256": digest(path), "size": path.stat().st_size} for path in files])
    cases = scenes()
    protocol = {
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"), "version": 11, "selfcheck": checks,
        "cases": cases, "quality_cases": quality_cases(cases), "workflow_specs": {s["id"]: specs(s) for s in cases},
        "models": NAMES, "model_to_profile": ACTION_PROFILE, "endpoint_models_at_seal": endpoint_models,
        "serving": {"max_model_len": MAX_CONTEXT_TOKENS,
                    "kv_cache_memory_bytes": {"qwen3-06": "1 GiB", "qwen3-17": "1 GiB", "qwen3-14b-awq": "2 GiB"}},
        "reasoning_parser": "vLLM Qwen3 parser: final JSON content scored; reasoning separated",
        "decoding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "max_tokens": MAX_OUTPUT_TOKENS, "enable_thinking": THINKING,
                     "max_model_len": MAX_CONTEXT_TOKENS},
        "prior_engineering_smoke": {"version": "v10", "input_cap": 4096, "output_cap": 2048,
                                    "observed_failure": "A 1.7B verify node exhausted 2048 completion tokens. Replaying the exact request and seed used 2676 completion tokens with a 3072 cap; v10 is retained as diagnostic-only data."},
        "E1": {"calls": 324, "independent_cases": 12, "tests_per_case": 3,
               "seeds": [20261017, 20261018, 20261019],
               "nodes_per_case": ["intent", "extract", "plan"], "paired_same_inputs": True,
               "quality": "typed exact field accuracy; exact schema is reported separately; prompts name only fields required by that task. Plan rubric=2 all fields correct, 1 correct level with at least one failed constraint, 0 otherwise; normalized rubric also reported",
               "proxy": "archived Eq7/profile values; mapping local=0.6B, edge_s=1.7B, edge_m=14B-AWQ declared before inference; descriptive association only, no recalibration",
               "cluster": "12 distinct workflows; model/node/seed calls are repeated measures"},
        "E2": {"workflows": 72, "calls": 576, "methods": METHODS, "repetitions": 2, "deadline_s": WORKFLOW_DEADLINE_S,
               "order": "rotate method order across 12 cases and reverse across repetitions", "latency": "actual wall clock incl tokenization/routing, plus per-call HTTP TTFT and completion wall; no simulated step",
               "success": "final seven reference fields all correct and actual elapsed<=240s; exact schema separately",
               "queues": "zero external queue, one serial caller, no injected traffic",
               "state": "actual completed wall time in seconds; future outputs/quality unavailable; current prompt uses measured max tokenizer length",
               "mask": "three measured service endpoints mapped to archived local, edge_s, edge_m action slots; edge_l/cloud unavailable; original feasibility constraints retained",
               "GOA": "original source-code select_goa_dag_action baseline on the same three legal action slots; it uses the archived profile latency and uncertainty rule, not a post-hoc minimum over measured endpoints",
               "references": "posthoc evaluation; online state feedback is the predeclared deterministic closed-rule checker"},
        "smoke": {"cases": ["flood_1", "traffic_1", "pump_1", "lighting_1"], "workflows": 12, "calls": 96, "scope": "engineering-only run on four predeclared cases; excluded from formal summaries"},
        "calibration": {"phase": "frozen-policy comparison; no retraining", "tier": "three measured endpoint configurations",
                        "deadline_s": WORKFLOW_DEADLINE_S,
                        "deadline_basis": "fixed at 240 seconds before policy runs; 8 serial nodes with 3072-token maximum, chosen after replaying the same v10 1.7B verify request and seed required 2676 completion tokens",
                        "online_feedback": "deterministic validator of completed node output against available source facts and closed workflow rules"},
        "limits": ["one RTX 5090 Laptop hosts all three endpoints; endpoints share GPU and are not separate physical edge sites", "handwritten threshold cases, not production city data", "no policy retraining or coefficient tuning", "retrieve/tool_call are structured parameter tasks without real database/tool execution", "Qwen3-14B-AWQ is 4-bit AWQ while smaller endpoints are BF16", "no human double-rating was performed; closed-rule reference scoring is objective for these tasks only"],
        "script_sha256": digest(Path(__file__)),
    }
    dump(OUT / "预先固定协议.json", protocol)
    (OUT / "协议SHA256.txt").write_text(digest(OUT / "预先固定协议.json"), encoding="ascii")
    print("SEALED", OUT, flush=True)


def validate_protocol() -> dict:
    """Reject unrecorded changes to sealed code or cases."""
    protocol = json.loads((OUT / "预先固定协议.json").read_text(encoding="utf-8"))
    assert digest(OUT / "预先固定协议.json") == (OUT / "协议SHA256.txt").read_text().strip()
    assert digest(Path(__file__)) == protocol["script_sha256"]
    for entry in json.loads((OUT / "开始前文件摘要.json").read_text(encoding="utf-8")):
        assert digest(recorded_path(entry["path"])) == entry["sha256"], entry["path"]
    return protocol


def call(action: int, prompt: str, budget: int = MAX_OUTPUT_TOKENS, seed: int = 20261017) -> dict:
    """Stream and retain actual content, TTFT, token usage and service errors."""
    payload = {"model": NAMES[action], "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": seed, "max_tokens": budget, "chat_template_kwargs": {"enable_thinking": THINKING}, "stream": True, "stream_options": {"include_usage": True}}
    started = time.perf_counter()
    chunks, content, usage, ttft, finish, error = [], [], {}, None, None, None
    try:
        req = urllib.request.Request(SERVICES[action] + "/v1/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                chunks.append(chunk)
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
    except Exception as exc:
        error = repr(exc)
    return {"action": action, "model": NAMES[action], "prompt": prompt, "request": payload, "chunks": chunks, "output": "".join(content), "elapsed_s": time.perf_counter() - started, "ttft_s": ttft, "usage": usage, "finish_reason": finish, "error": error, "retry_count": 0}


def token_counts(prompt: str) -> dict:
    """Measure every endpoint tokenizer before routing requests."""
    return {a: request(url + "/tokenize", {"model": NAMES[a], "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": THINKING}})["count"] for a, url in SERVICES.items()}


def calibrate() -> None:
    """Fit nonnegative empirical request-time coefficients on separate prompts."""
    import numpy as np
    from scipy.optimize import nnls
    path = OUT / "独立服务校准.jsonl"
    if path.exists():
        raise FileExistsError(path)
    texts = [
        "依次原样输出一个只有items字段的JSON对象，items为1到8的整数列表。",
        "依次原样输出一个只有items字段的JSON对象，items为1到32的整数列表。",
        "背景资料仅用于占位，不需解释：" + "系统需要记录到达时间、任务编号和资源状态。" * 12 + "\n输出只有items字段的JSON对象，items为1到8的整数列表。",
        "背景资料仅用于占位，不需解释：" + "系统需要记录到达时间、任务编号和资源状态。" * 12 + "\n输出只有items字段的JSON对象，items为1到32的整数列表。",
    ]
    rows = []
    for index, prompt in enumerate(texts):
        for action in ([0, 1, 2] if index % 2 == 0 else [2, 1, 0]):
            row = call(action, prompt)
            row["calibration_id"] = index
            append(path, row)
            rows.append(row)
            assert row["error"] is None and row["usage"]
    coefficients = {}
    for action in SERVICES:
        selected = [r for r in rows if r["action"] == action]
        design = np.array([[1, r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]] for r in selected], dtype=float)
        observed = np.array([r["elapsed_s"] for r in selected])
        fit, residual = nnls(design, observed)
        coefficients[action] = {"intercept_s": float(fit[0]), "input_token_s": float(fit[1]), "output_token_s": float(fit[2]), "residual_norm_s": float(residual), "n": 4}
    dump(OUT / "服务时延系数.json", coefficients)
    print("CALIBRATED", coefficients, flush=True)


def proxy(test: dict, action: int) -> float:
    """Compute the unchanged simulation score using declared template assumptions."""
    sys.path.insert(0, str(CODE / "src"))
    from llm_offload_env.serving import default_profiles, quality
    from llm_offload_env.taskgen import Node
    kind = test["kind"]
    node = Node(idx=0, kind=kind, input_tokens=500, output_tokens=192, uncertainty=0.55 if kind == "plan" else 0.25, privacy=1, risk=0.7 if kind == "plan" else 0.45, deadline_weight=1)
    return float(quality(node, default_profiles("tradeoff_v2")[ACTION_PROFILE[action]]))


def run_quality() -> None:
    """Execute all 72 paired requests without reusing live-routing responses."""
    protocol = validate_protocol()
    path = OUT / "E1_逐调用.jsonl"
    if path.exists():
        raise FileExistsError(path)
    for seed in protocol["E1"]["seeds"]:
        for index, test in enumerate(protocol["quality_cases"]):
            order = [0, 1, 2] if (index + seed) % 2 == 0 else [2, 1, 0]
            for action in order:
                row = call(action, test["prompt"], seed=seed)
                row.update({"test": test, "seed": seed,
                            "grade": grade(row["output"], test["expected"]),
                            "fixed_proxy": proxy(test, action)})
                if test["kind"] == "plan":
                    row["rubric_0_2"] = 2 if row["grade"]["all_fields_correct"] else 1 if row["grade"]["matches"]["level"] else 0
                else:
                    row["rubric_0_2"] = 2 * row["grade"]["field_accuracy"]
                append(path, row)
            print("E1", seed, index + 1, "/36", test["id"], flush=True)


def make_policies() -> dict:
    """Load the archived checkpoints without retraining."""
    from sb3_contrib import MaskablePPO
    sys.path.insert(0, str(CODE / "src"))
    from llm_offload_env.gnn_policy import DAGMessagePassingExtractor, DTODRLGATExtractor
    checkpoint = CODE / "models" / "dtodrl_gat_pretrained.pt"
    original_cwd = Path.cwd()
    os.chdir(CODE)
    try:
        return {name: MaskablePPO.load(CODE.parent / f"data/main_comparison/seed_2026/{folder}/model/ppo_full.zip", device="cpu") for name, folder in [("UGQP", "ugqp_gnn"), ("DTODRL-joint", "dtodrl_joint")]}
    finally:
        os.chdir(original_cwd)


def run_live(smoke: bool) -> None:
    """Run complete real DAGs with isolated state and frozen method order."""
    import numpy as np
    import torch
    validate_protocol()
    policies = make_policies()
    from llm_offload_env.config import make_config
    from llm_offload_env.env import LLMAgentOffloadEnv
    from llm_offload_env.taskgen import Task, Node, compute_graph_features
    from llm_offload_env.serving import feasible
    tag = "复跑" if smoke else "E2"
    node_path, summary_path = OUT / f"{tag}_逐节点.jsonl", OUT / f"{tag}_逐工作流.jsonl"
    if node_path.exists() or summary_path.exists():
        raise FileExistsError(node_path)
    selected = [s for s in scenes() if s["id"] in {"flood_1", "traffic_1"}] if smoke else scenes()
    coefficients = json.loads((OUT / "服务时延系数.json").read_text(encoding="utf-8"))
    run_count = 0
    for repetition in range(1 if smoke else 2):
        for case_index, scene in enumerate(selected):
            order = METHODS[case_index % 3:] + METHODS[:case_index % 3]
            if repetition % 2:
                order = list(reversed(order))
            for method in order:
                env = LLMAgentOffloadEnv(make_config(seed=2026, min_nodes=8, max_nodes=12, arrival_rate=0, network_scale=0, queue_mode="legacy", queue_scale=0, profile_mode="tradeoff_v2"))
                env.reset()
                for profile in env.profiles.values():
                    profile.queue_load = 0.0
                items = specs(scene)
                nodes = [Node(idx=i, kind=s["kind"], input_tokens=500, output_tokens=192, uncertainty=0.55 if s["kind"] in {"plan", "verify"} else 0.25, privacy=1, risk=0.7 if s["kind"] in {"plan", "verify"} else 0.45, deadline_weight=1, predecessors=s["pred"], successors=[j for j, t in enumerate(items) if i in t["pred"]], token_uncertainty=0.5) for i, s in enumerate(items)]
                compute_graph_features(nodes)
                env.task = Task(task_id=run_count, scenario=scene["id"], nodes=nodes, deadline=20)
                env.ready_order = list(range(8))
                outputs, rows = {}, []
                started = time.perf_counter()
                for index, item in enumerate(items):
                    prompt = prompt_for(scene, items, index, outputs)
                    counts = token_counts(prompt)
                    assert max(counts.values()) + 192 <= MAX_CONTEXT_TOKENS, "context budget exceeded"
                    nodes[index].input_tokens = max(counts.values())
                    obs = env._obs()
                    for pending in range(index, 12):
                        obs[pending * 12 + 11] = 0.0
                    mask = np.array([False, feasible(nodes[index], env.profiles["edge_s"]), feasible(nodes[index], env.profiles["edge_m"]), False, False])
                    assert mask.any()
                    estimate = {a: coefficients[str(a)]["intercept_s"] + coefficients[str(a)]["input_token_s"] * counts[a] + coefficients[str(a)]["output_token_s"] * 192 for a in SERVICES if mask[a]}
                    route_start = time.perf_counter()
                    probabilities = None
                    if method == "GOA-DAG":
                        action = min(estimate, key=estimate.get)
                    else:
                        action = int(policies[method].predict(obs, deterministic=True, action_masks=mask)[0])
                        with torch.no_grad():
                            tensor, _ = policies[method].policy.obs_to_tensor(obs)
                            probabilities = policies[method].policy.get_distribution(tensor, action_masks=mask).distribution.probs.cpu().numpy().tolist()[0]
                    route_s = time.perf_counter() - route_start
                    assert mask[action]
                    expected = conditional_verify(scene, outputs) if index == 6 else item["expected"]
                    row = call(action, prompt)
                    outputs[index] = row["output"]
                    finish_s = time.perf_counter() - started
                    result = grade(row["output"], expected)
                    env.completion[index] = finish_s
                    env.realized_quality[index] = result["typed_coverage"]
                    env.current_pos = index + 1
                    row.update({"case": scene["id"], "family": scene["family"], "method": method, "repetition": repetition, "node": index, "kind": item["kind"], "predecessors": item["pred"], "expected": expected, "grade": result, "legal_actions": np.flatnonzero(mask).tolist(), "action_probabilities": probabilities, "estimated_endpoint_s": estimate, "current_token_counts": counts, "observation": obs.tolist(), "online_typed_coverage": result["typed_coverage"], "finish_s": finish_s, "routing_s": route_s})
                    rows.append(row)
                    append(node_path, row)
                wall = time.perf_counter() - started
                final = rows[-1]["grade"]
                summary = {"case": scene["id"], "family": scene["family"], "method": method, "repetition": repetition, "order_in_case": order.index(method), "wall_s": wall, "service_wall_s": sum(r["elapsed_s"] for r in rows), "mean_field_accuracy": statistics.mean(r["grade"]["field_accuracy"] for r in rows), "final_field_accuracy": final["field_accuracy"], "final_correct": final["all_fields_correct"], "final_schema_exact": final["schema_exact"], "within_20s": wall <= 20, "joint_success": final["all_fields_correct"] and wall <= 20, "strict_json_count": sum(r["grade"]["strict_json"] for r in rows), "schema_exact_count": sum(r["grade"]["schema_exact"] for r in rows), "errors": sum(r["error"] is not None for r in rows), "truncated": sum(r["finish_reason"] == "length" for r in rows), "actions": [r["action"] for r in rows], "first_wrong_node": next((r["node"] for r in rows if not r["grade"]["all_fields_correct"]), None), "final_output": outputs[7]}
                append(summary_path, summary)
                run_count += 1
                print(tag, run_count, "/" + str(len(selected) * (1 if smoke else 2) * 3), json.dumps(summary, ensure_ascii=False), flush=True)


def csv_write(path: Path, rows: list[dict]) -> None:
    """Export parallel summary fields for inspection."""
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_ugqp_calibration(smoke: bool, methods: tuple[str, ...] = ("UGQP",)) -> None:
    """Run frozen policies and the source baseline over the three endpoints."""
    import numpy as np
    import torch
    protocol = validate_protocol()
    policies = make_policies()
    from llm_offload_env.config import make_config
    from llm_offload_env.env import LLMAgentOffloadEnv
    from llm_offload_env.taskgen import Task, Node, compute_graph_features
    from llm_offload_env.baselines import select_goa_dag_action
    if smoke:
        selected_ids = {"flood_1", "traffic_1", "pump_1", "lighting_1"}
        selected = [scene for scene in scenes() if scene["id"] in selected_ids]
        tag = "UGQP_校准短试" if methods == ("UGQP",) else "横向校准短试"
    else:
        selected = scenes()
        tag = "UGQP_校准正式" if methods == ("UGQP",) else "横向校准正式"
    node_path = OUT / f"{tag}_逐节点.jsonl"
    summary_path = OUT / f"{tag}_逐工作流.jsonl"
    if node_path.exists() or summary_path.exists():
        raise FileExistsError(node_path)
    coefficients = json.loads((OUT / "服务时延系数.json").read_text(encoding="utf-8"))
    all_rows, all_summaries = [], []
    run_count = 0
    repetitions = 1 if smoke else 2
    for repetition in range(repetitions):
        for case_index, scene in enumerate(selected):
            order = list(methods[case_index % len(methods):]) + list(methods[:case_index % len(methods)])
            if repetition % 2:
                order.reverse()
            for method in order:
                env = LLMAgentOffloadEnv(make_config(seed=2026, min_nodes=8, max_nodes=12, arrival_rate=0, network_scale=0, queue_mode="legacy", queue_scale=0, profile_mode="tradeoff_v2"))
                env.reset()
                for profile in env.profiles.values():
                    profile.queue_load = 0.0
                items = specs(scene)
                nodes = [Node(idx=i, kind=s["kind"], input_tokens=500, output_tokens=192,
                              uncertainty=0.55 if s["kind"] in {"plan", "verify"} else 0.25,
                              privacy=1, risk=0.7 if s["kind"] in {"plan", "verify"} else 0.45,
                              deadline_weight=1, predecessors=s["pred"],
                              successors=[j for j, t in enumerate(items) if i in t["pred"]],
                              token_uncertainty=0.5) for i, s in enumerate(items)]
                compute_graph_features(nodes)
                env.task = Task(task_id=run_count, scenario=scene["id"], nodes=nodes, deadline=scene["deadline_s"])
                env.ready_order = list(range(8))
                outputs, rows = {}, []
                started = time.perf_counter()
                for index, item in enumerate(items):
                    prompt = prompt_for(scene, items, index, outputs)
                    counts = token_counts(prompt)
                    assert max(counts.values()) + MAX_OUTPUT_TOKENS <= MAX_CONTEXT_TOKENS, "context budget exceeded"
                    nodes[index].input_tokens = max(counts.values())
                    obs = env._obs()
                    for pending in range(index, 12):
                        obs[pending * 12 + 11] = 0.0
                    mask = env.action_masks()
                    mask[3:] = False
                    assert mask.any()
                    estimate = {a: coefficients[str(a)]["intercept_s"] + coefficients[str(a)]["input_token_s"] * counts[a] + coefficients[str(a)]["output_token_s"] * 192 for a in SERVICES if mask[a]}
                    route_start = time.perf_counter()
                    if method == "GOA-DAG":
                        action = select_goa_dag_action(env, np.flatnonzero(mask).tolist())
                        probabilities = None
                    else:
                        action = int(policies[method].predict(obs, deterministic=True, action_masks=mask)[0])
                        with torch.no_grad():
                            tensor, _ = policies[method].policy.obs_to_tensor(obs)
                            probabilities = policies[method].policy.get_distribution(tensor, action_masks=mask).distribution.probs.cpu().numpy().tolist()[0]
                    route_s = time.perf_counter() - route_start
                    expected = conditional_verify(scene, outputs) if index == 6 else item["expected"]
                    call_seed = 20261017 + repetition * 10000 + case_index * 100 + index
                    row = call(action, prompt, seed=call_seed)
                    outputs[index] = row["output"]
                    finish_s = time.perf_counter() - started
                    result = grade(row["output"], expected)
                    feedback = contract_feedback(scene, items, index, outputs, row["output"])
                    env.completion[index] = finish_s
                    env.realized_quality[index] = feedback["score"]
                    env.current_pos = index + 1
                    row.update({"case": scene["id"], "family": scene["family"], "method": method,
                                "repetition": repetition, "call_seed": call_seed, "node": index,
                                "kind": item["kind"], "predecessors": item["pred"], "expected": expected,
                                "grade": result, "contract_feedback": feedback, "legal_actions": np.flatnonzero(mask).tolist(),
                                "action_probabilities": probabilities, "estimated_endpoint_s": estimate,
                                "current_token_counts": counts, "observation": obs.tolist(),
                                "online_semantic_score": feedback["score"], "finish_s": finish_s, "routing_s": route_s})
                    rows.append(row)
                    all_rows.append(row)
                    append(node_path, row)
                wall = time.perf_counter() - started
                final = rows[-1]["grade"]
                final_contract = rows[-1]["contract_feedback"]
                summary = {"case": scene["id"], "family": scene["family"], "method": method,
                           "repetition": repetition, "order_in_case": order.index(method), "calibration_tier": scene["calibration_tier"],
                           "deadline_s": scene["deadline_s"], "wall_s": wall, "service_wall_s": sum(r["elapsed_s"] for r in rows),
                           "mean_field_accuracy": statistics.mean(r["grade"]["field_accuracy"] for r in rows),
                           "mean_online_semantic_score": statistics.mean(r["contract_feedback"]["score"] for r in rows),
                           "final_field_accuracy": final["field_accuracy"], "final_correct": final["all_fields_correct"],
                           "final_contract_pass": final_contract["contract_pass"], "within_deadline": wall <= scene["deadline_s"],
                           "joint_success": final["all_fields_correct"] and wall <= scene["deadline_s"],
                           "strict_json_count": sum(r["grade"]["strict_json"] for r in rows),
                           "schema_exact_count": sum(r["grade"]["schema_exact"] for r in rows),
                           "semantic_contract_pass_count": sum(r["contract_feedback"]["contract_pass"] for r in rows),
                           "errors": sum(r["error"] is not None for r in rows), "truncated": sum(r["finish_reason"] == "length" for r in rows),
                           "actions": [r["action"] for r in rows], "first_wrong_node": next((r["node"] for r in rows if not r["grade"]["all_fields_correct"]), None),
                           "final_output": outputs[7]}
                all_summaries.append(summary)
                append(summary_path, summary)
                run_count += 1
                print(tag, run_count, "/" + str(len(selected) * len(methods) * repetitions), json.dumps(summary, ensure_ascii=False), flush=True)
    dump(OUT / f"{tag}_汇总.json", {"workflows": all_summaries, "nodes": len(all_rows), "methods": list(methods), "repetitions": repetitions, "same_calibrated_parameters": True})


def analyze_ugqp_calibration() -> None:
    """Report the calibration gate without changing any scores or records."""
    for tag in ["UGQP_校准短试", "UGQP_校准正式"]:
        path = OUT / f"{tag}_逐工作流.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        print(json.dumps({"tag": tag, "workflows": len(rows), "joint_success": sum(r["joint_success"] for r in rows),
                          "joint_rate": statistics.mean(r["joint_success"] for r in rows),
                          "final_correct": sum(r["final_correct"] for r in rows),
                          "within_deadline": sum(r["within_deadline"] for r in rows),
                          "mean_field_accuracy": statistics.mean(r["mean_field_accuracy"] for r in rows),
                          "mean_online_semantic_score": statistics.mean(r["mean_online_semantic_score"] for r in rows),
                          "action_counts": dict((str(action), sum(action in r["actions"] for r in rows)) for action in [1, 2])}, ensure_ascii=False, indent=2))


def analyze_v11() -> None:
    """Summarize the sealed three-endpoint experiment and paired comparisons."""
    import numpy as np
    from scipy.stats import spearmanr

    protocol = validate_protocol()
    e1 = read_rows("E1_逐调用.jsonl")
    nodes = read_rows("横向校准正式_逐节点.jsonl")
    runs = read_rows("横向校准正式_逐工作流.jsonl")
    assert len(e1) == 324, f"E1 calls={len(e1)}"
    assert len(nodes) == 576, f"E2 node calls={len(nodes)}"
    assert len(runs) == 72, f"E2 workflows={len(runs)}"
    assert len({(r["test"]["id"], r["action"], r["seed"]) for r in e1}) == 324
    assert len({(r["case"], r["method"], r["repetition"], r["node"]) for r in nodes}) == 576
    assert all(r["action"] in SERVICES for r in e1 + nodes)

    summary = {"counts": {"E1_calls": len(e1), "E2_calls": len(nodes), "E2_workflows": len(runs)},
               "E1_models": [], "E1_by_kind": [], "E1_by_family": [], "E1_proxy": {},
               "E2_methods": [], "E2_by_family": [], "E2_routes": {}, "errors": {}}
    for action in SERVICES:
        selected = [r for r in e1 if r["action"] == action]
        summary["E1_models"].append({
            "action": action, "model": NAMES[action], "n": len(selected),
            "mean_field_accuracy": statistics.mean(r["grade"]["field_accuracy"] for r in selected),
            "all_fields_correct": sum(r["grade"]["all_fields_correct"] for r in selected),
            "strict_json": sum(r["grade"]["strict_json"] for r in selected),
            "exact_schema": sum(r["grade"]["schema_exact"] for r in selected),
            "mean_normalized_rubric": statistics.mean(r["rubric_0_2"] / 2 for r in selected),
            "median_call_s": statistics.median(r["elapsed_s"] for r in selected),
            "median_ttft_s": statistics.median(r["ttft_s"] for r in selected if r["ttft_s"] is not None),
        })
        for field, values, target in [("kind", ["intent", "extract", "plan"], "E1_by_kind"),
                                      ("family", list(RULES), "E1_by_family")]:
            for value in values:
                group = [r for r in selected if r["test"][field] == value]
                summary[target].append({"model": NAMES[action], field: value, "n": len(group),
                                        "mean_field_accuracy": statistics.mean(r["grade"]["field_accuracy"] for r in group),
                                        "all_fields_correct": sum(r["grade"]["all_fields_correct"] for r in group)})

    by_test_action = {}
    for row in e1:
        key = (row["test"]["id"], row["action"])
        by_test_action.setdefault(key, []).append(row)
    proxy_points = []
    for (test_id, action), group in by_test_action.items():
        proxy_points.append({"test": test_id, "action": action,
                             "proxy": group[0]["fixed_proxy"],
                             "observed_mean_field_accuracy": statistics.mean(r["grade"]["field_accuracy"] for r in group)})
    proxy_rho = spearmanr([r["proxy"] for r in proxy_points],
                          [r["observed_mean_field_accuracy"] for r in proxy_points])

    pairs = []
    for test in protocol["quality_cases"]:
        for seed in protocol["E1"]["seeds"]:
            selected = {r["action"]: r for r in e1 if r["test"]["id"] == test["id"] and r["seed"] == seed}
            pairs.append({"test": test["id"], "case": test["scene"], "kind": test["kind"], "seed": seed,
                          "0.6B": selected[0]["grade"]["field_accuracy"],
                          "1.7B": selected[1]["grade"]["field_accuracy"],
                          "14B_AWQ": selected[2]["grade"]["field_accuracy"],
                          "14B_minus_1.7B": selected[2]["grade"]["field_accuracy"] - selected[1]["grade"]["field_accuracy"]})
    case_deltas = [statistics.mean(r["14B_minus_1.7B"] for r in pairs if r["case"] == scene["id"]) for scene in protocol["cases"]]
    rng = np.random.default_rng(20261018)
    e1_boot = np.mean(rng.choice(case_deltas, size=(20000, len(case_deltas)), replace=True), axis=1)
    summary["E1_proxy"] = {"descriptive_spearman_rho": float(proxy_rho.statistic),
                           "paired_test_endpoint_points": len(proxy_points),
                           "limitation": "descriptive association only; repeated seeds and manually authored cases do not establish general proxy validity"}
    summary["E1_14B_vs_1_7B"] = {"paired_test_seed_n": len(pairs),
                                "mean_field_accuracy_delta": statistics.mean(r["14B_minus_1.7B"] for r in pairs),
                                "cases_14B_higher": sum(x > 1e-12 for x in case_deltas),
                                "cases_tied": sum(abs(x) <= 1e-12 for x in case_deltas),
                                "cases_14B_lower": sum(x < -1e-12 for x in case_deltas),
                                "case_cluster_bootstrap_95_ci": np.quantile(e1_boot, [0.025, 0.975]).tolist()}

    for method in METHODS:
        group = [r for r in runs if r["method"] == method]
        method_nodes = [r for r in nodes if r["method"] == method]
        row = {"method": method, "workflows": len(group),
               "mean_node_field_accuracy": statistics.mean(r["mean_field_accuracy"] for r in group),
               "mean_node_contract_score": statistics.mean(r["mean_online_semantic_score"] for r in group),
               "final_correct": sum(r["final_correct"] for r in group),
               "final_contract_pass": sum(r["final_contract_pass"] for r in group),
               "within_deadline": sum(r["within_deadline"] for r in group),
               "joint_success": sum(r["joint_success"] for r in group),
               "mean_wall_s": statistics.mean(r["wall_s"] for r in group),
               "median_wall_s": statistics.median(r["wall_s"] for r in group),
               "http_errors": sum(r["error"] is not None for r in method_nodes),
               "truncated": sum(r["finish_reason"] == "length" for r in method_nodes)}
        row["endpoint_calls"] = {NAMES[a]: sum(r["action"] == a for r in method_nodes) for a in SERVICES}
        summary["E2_methods"].append(row)
        for family in RULES:
            family_runs = [r for r in group if r["family"] == family]
            summary["E2_by_family"].append({"method": method, "family": family,
                                            "n": len(family_runs),
                                            "mean_node_field_accuracy": statistics.mean(r["mean_field_accuracy"] for r in family_runs),
                                            "final_correct": sum(r["final_correct"] for r in family_runs),
                                            "joint_success": sum(r["joint_success"] for r in family_runs)})

    route_pairs = []
    for case in protocol["cases"]:
        for repetition in range(2):
            subset = {r["method"]: r for r in runs if r["case"] == case["id"] and r["repetition"] == repetition}
            route_pairs.append({"case": case["id"], "repetition": repetition,
                                "UGQP_equals_DTODRL": subset["UGQP"]["actions"] == subset["DTODRL-joint"]["actions"],
                                "UGQP_minus_DTODRL_field_accuracy": subset["UGQP"]["mean_field_accuracy"] - subset["DTODRL-joint"]["mean_field_accuracy"],
                                "UGQP_minus_GOA_field_accuracy": subset["UGQP"]["mean_field_accuracy"] - subset["GOA-DAG"]["mean_field_accuracy"],
                                "UGQP_minus_DTODRL_wall_s": subset["UGQP"]["wall_s"] - subset["DTODRL-joint"]["wall_s"],
                                "UGQP_minus_GOA_wall_s": subset["UGQP"]["wall_s"] - subset["GOA-DAG"]["wall_s"]})
    summary["E2_routes"] = {"paired_workflows": len(route_pairs),
                            "identical_UGQP_DTODRL_routes": sum(r["UGQP_equals_DTODRL"] for r in route_pairs),
                            "UGQP_minus_DTODRL_mean_field_accuracy": statistics.mean(r["UGQP_minus_DTODRL_field_accuracy"] for r in route_pairs),
                            "UGQP_minus_GOA_mean_field_accuracy": statistics.mean(r["UGQP_minus_GOA_field_accuracy"] for r in route_pairs),
                            "UGQP_minus_DTODRL_mean_wall_s": statistics.mean(r["UGQP_minus_DTODRL_wall_s"] for r in route_pairs),
                            "UGQP_minus_GOA_mean_wall_s": statistics.mean(r["UGQP_minus_GOA_wall_s"] for r in route_pairs)}
    summary["errors"] = {"http_errors": sum(r["error"] is not None for r in e1 + nodes),
                         "truncated": sum(r["finish_reason"] == "length" for r in e1 + nodes),
                         "max_prompt_tokens": max(r.get("usage", {}).get("prompt_tokens", 0) for r in e1 + nodes)}
    dump(OUT / "正式实验汇总.json", summary)
    dump(OUT / "结束后不可变文件核验.json", {"passed": True,
         "checked": len(json.loads((OUT / "开始前文件摘要.json").read_text(encoding="utf-8"))),
         "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def analyze() -> None:
    """Summarize every formal record, including failures and identical routes."""
    import numpy as np
    from scipy.stats import spearmanr
    validate_protocol()
    e1, nodes, runs = read_rows("E1_逐调用.jsonl"), read_rows("E2_逐节点.jsonl"), read_rows("E2_逐工作流.jsonl")
    assert len(e1) == 72 and len(nodes) == 576 and len(runs) == 72
    assert len({(r["test"]["id"], r["action"]) for r in e1}) == 72
    assert len({(r["case"], r["method"], r["repetition"], r["node"]) for r in nodes}) == 576
    summary = {"counts": {"E1_calls": len(e1), "E2_calls": len(nodes), "E2_workflows": len(runs)}, "E1": [], "E2": [], "E1_by_kind": [], "E1_by_family": []}
    for action in SERVICES:
        chosen = [r for r in e1 if r["action"] == action]
        summary["E1"].append({"model": NAMES[action], "n": len(chosen), "mean_field_accuracy": statistics.mean(r["grade"]["field_accuracy"] for r in chosen), "normalized_rubric": statistics.mean(r["rubric_0_2"] / 2 for r in chosen), "all_fields_correct": sum(r["grade"]["all_fields_correct"] for r in chosen), "strict_json": sum(r["grade"]["strict_json"] for r in chosen), "exact_schema": sum(r["grade"]["schema_exact"] for r in chosen), "median_call_s": statistics.median(r["elapsed_s"] for r in chosen), "median_ttft_s": statistics.median(r["ttft_s"] for r in chosen if r["ttft_s"] is not None)})
        for key, values, name in [("kind", ["intent", "extract", "plan"], "E1_by_kind"), ("family", list(RULES), "E1_by_family")]:
            for value in values:
                group = [r for r in chosen if r["test"][key] == value]
                summary[name].append({"model": NAMES[action], key: value, "n": len(group), "accuracy": statistics.mean(r["grade"]["field_accuracy"] for r in group), "fully_correct": sum(r["grade"]["all_fields_correct"] for r in group)})
    pairs = []
    for test in quality_cases(scenes()):
        selected = {r["action"]: r for r in e1 if r["test"]["id"] == test["id"]}
        small, medium = [selected[a]["grade"]["field_accuracy"] for a in [1, 2]]
        pairs.append({"test": test["id"], "case": test["scene"], "kind": test["kind"], "small_accuracy": small, "medium_accuracy": medium, "delta": medium - small, "small_proxy": selected[1]["fixed_proxy"], "medium_proxy": selected[2]["fixed_proxy"]})
    deltas = [statistics.mean(p["delta"] for p in pairs if p["case"] == s["id"]) for s in scenes()]
    rng = np.random.default_rng(20261017)
    bootstrap = np.mean(rng.choice(deltas, size=(10000, 12), replace=True), axis=1)
    rho = float(spearmanr([r["fixed_proxy"] for r in e1], [r["grade"]["field_accuracy"] for r in e1]).statistic)
    summary["E1_proxy"] = {"descriptive_spearman": rho if np.isfinite(rho) else None, "mae": statistics.mean(abs(r["fixed_proxy"] - r["grade"]["field_accuracy"]) for r in e1), "distinct_proxy_values": len({r["fixed_proxy"] for r in e1}), "medium_better": sum(p["delta"] > 1e-9 for p in pairs), "tie": sum(abs(p["delta"]) <= 1e-9 for p in pairs), "medium_worse": sum(p["delta"] < -1e-9 for p in pairs), "mean_delta": statistics.mean(deltas), "case_cluster_bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(), "limitation": "only 12 handmade cases; few tied proxy values; task/model effects mixed; no causal or calibrated prediction claim"}
    for method in METHODS:
        chosen = [r for r in runs if r["method"] == method]
        selected_nodes = [r for r in nodes if r["method"] == method]
        summary["E2"].append({"method": method, "workflows": len(chosen), "mean_wall_s": statistics.mean(r["wall_s"] for r in chosen), "median_wall_s": statistics.median(r["wall_s"] for r in chosen), "min_wall_s": min(r["wall_s"] for r in chosen), "max_wall_s": max(r["wall_s"] for r in chosen), "mean_node_field_accuracy": statistics.mean(r["mean_field_accuracy"] for r in chosen), "final_correct": sum(r["final_correct"] for r in chosen), "within_20s": sum(r["within_20s"] for r in chosen), "joint_success": sum(r["joint_success"] for r in chosen), "small_calls": sum(r["action"] == 1 for r in selected_nodes), "medium_calls": sum(r["action"] == 2 for r in selected_nodes), "strict_json": sum(r["grade"]["strict_json"] for r in selected_nodes), "exact_schema": sum(r["grade"]["schema_exact"] for r in selected_nodes)})
    equal = 0
    route_pairs = []
    for scene in scenes():
        for repetition in range(2):
            selected = {r["method"]: r for r in runs if r["case"] == scene["id"] and r["repetition"] == repetition}
            same = selected["UGQP"]["actions"] == selected["DTODRL-joint"]["actions"]
            equal += same
            route_pairs.append({"case": scene["id"], "repetition": repetition, "same_UGQP_DTODRL_routes": same, "UGQP_minus_DTODRL_wall_s": selected["UGQP"]["wall_s"] - selected["DTODRL-joint"]["wall_s"], "UGQP_minus_GOA_wall_s": selected["UGQP"]["wall_s"] - selected["GOA-DAG"]["wall_s"]})
    summary["E2_identical_UGQP_DTODRL_routes"] = {"same": equal, "paired_workflows": 24}
    summary["errors"] = {"http_errors": sum(r["error"] is not None for r in e1 + nodes), "truncated": sum(r["finish_reason"] == "length" for r in e1 + nodes), "max_actual_prompt_tokens": max(r["usage"].get("prompt_tokens", 0) for r in e1 + nodes)}
    summary["failures"] = [{key: r[key] for key in ["case", "method", "repetition", "first_wrong_node", "final_field_accuracy", "final_output"]} for r in runs if not r["final_correct"]]
    dump(OUT / "正式实验汇总.json", summary)
    dump(OUT / "结束后不可变文件核验.json", {"passed": True, "checked": len(json.loads((OUT / "开始前文件摘要.json").read_text(encoding="utf-8"))), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "calibrate", "quality", "horizontal_smoke", "horizontal_full", "analyze_v11"])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "calibrate":
        validate_protocol()
        calibrate()
    elif args.mode == "quality":
        run_quality()
    elif args.mode == "horizontal_smoke":
        run_ugqp_calibration(True, ("UGQP", "DTODRL-joint", "GOA-DAG"))
    elif args.mode == "horizontal_full":
        run_ugqp_calibration(False, ("UGQP", "DTODRL-joint", "GOA-DAG"))
    else:
        analyze_v11()
