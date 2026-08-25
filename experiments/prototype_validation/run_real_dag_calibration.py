"""Collect structured serving traces for four smart-city workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from inference.model_utils import get_async_model


WORKFLOWS: Dict[str, List[Dict[str, Any]]] = {
    "emergency": [
        {"id": "n1", "pred": [], "kind": "intent", "label": "事件类型识别", "task": "判断这是一条城市应急事件，并输出 event_type 和 severity。"},
        {"id": "n2", "pred": ["n1"], "kind": "extract", "label": "地点对象抽取", "task": "抽取事件地点、受影响对象和时间。"},
        {"id": "n3", "pred": ["n2"], "kind": "retrieve", "label": "设施检索规划", "task": "列出需要查询的排水设施和道路状态字段。"},
        {"id": "n4", "pred": ["n2"], "kind": "retrieve", "label": "资源检索规划", "task": "列出需要查询的处置人员、车辆和可用性字段。"},
        {"id": "n5", "pred": ["n3", "n4"], "kind": "tool_call", "label": "资源查询参数生成", "task": "为城市资源查询生成结构化 API 参数。"},
        {"id": "n6", "pred": ["n5"], "kind": "plan", "label": "应急处置规划", "task": "根据前面结果生成有优先级的应急处置步骤。"},
        {"id": "n7", "pred": ["n6"], "kind": "verify", "label": "风险合规校验", "task": "检查处置步骤是否存在明显安全、隐私或资源约束冲突。"},
        {"id": "n8", "pred": ["n7"], "kind": "summarize", "label": "调度摘要", "task": "生成给城市调度员使用的简短结构化摘要。"},
    ],
    "traffic": [
        {"id": "n1", "pred": [], "kind": "intent", "label": "交通事件识别", "task": "识别交通事件类型、拥堵等级和是否需要快速响应。"},
        {"id": "n2", "pred": ["n1"], "kind": "extract", "label": "路段要素抽取", "task": "抽取道路、路口、方向、时间和受影响交通方式。"},
        {"id": "n3", "pred": ["n2"], "kind": "retrieve", "label": "交通状态检索", "task": "列出需要查询的实时路况和排队长度字段。"},
        {"id": "n4", "pred": ["n2"], "kind": "retrieve", "label": "信号公交检索", "task": "列出需要查询的信号灯和公交运行字段。"},
        {"id": "n5", "pred": ["n3", "n4"], "kind": "plan", "label": "交通疏导规划", "task": "生成交通疏导和绕行建议。"},
        {"id": "n6", "pred": ["n5"], "kind": "verify", "label": "规则校验", "task": "检查绕行建议是否违反道路方向、时间窗或公共交通约束。"},
        {"id": "n7", "pred": ["n6"], "kind": "summarize", "label": "交通公告摘要", "task": "生成面向交通调度员的结构化摘要。"},
    ],
    "municipal_service": [
        {"id": "n1", "pred": [], "kind": "intent", "label": "市政请求分类", "task": "识别市民请求的服务类别、紧急程度和办理目标。"},
        {"id": "n2", "pred": ["n1"], "kind": "extract", "label": "申请要素抽取", "task": "抽取申请人诉求、地点、对象和必要时间信息。"},
        {"id": "n3", "pred": ["n2"], "kind": "retrieve", "label": "政策检索规划", "task": "列出需要查询的政策条款和责任部门字段。"},
        {"id": "n4", "pred": ["n2"], "kind": "retrieve", "label": "历史工单检索规划", "task": "列出需要查询的历史工单和办理状态字段。"},
        {"id": "n5", "pred": ["n3", "n4"], "kind": "tool_call", "label": "工单参数生成", "task": "生成市政工单系统需要的结构化字段。"},
        {"id": "n6", "pred": ["n5"], "kind": "plan", "label": "服务办理规划", "task": "生成办理顺序、责任部门和预计反馈节点。"},
        {"id": "n7", "pred": ["n6"], "kind": "verify", "label": "合规性校验", "task": "检查办理建议是否缺少必要字段或违反隐私和政策约束。"},
        {"id": "n8", "pred": ["n7"], "kind": "summarize", "label": "市民答复摘要", "task": "生成清晰、克制、可执行的市民答复摘要。"},
    ],
    "facility_inspection": [
        {"id": "n1", "pred": [], "kind": "intent", "label": "设施异常识别", "task": "判断设施异常类型和巡检紧急程度。"},
        {"id": "n2", "pred": ["n1"], "kind": "extract", "label": "设施信息抽取", "task": "抽取设施编号、位置、异常现象和发现时间。"},
        {"id": "n3", "pred": ["n2"], "kind": "retrieve", "label": "资产历史检索规划", "task": "列出需要查询的资产台账和历史维护字段。"},
        {"id": "n4", "pred": ["n2"], "kind": "retrieve", "label": "传感器检索规划", "task": "列出需要查询的异常振动和运行状态字段。"},
        {"id": "n5", "pred": ["n3", "n4"], "kind": "tool_call", "label": "巡检查询参数生成", "task": "生成资产和维护系统的结构化查询参数。"},
        {"id": "n6", "pred": ["n5"], "kind": "plan", "label": "维修巡检规划", "task": "生成巡检、隔离和维修的先后步骤。"},
        {"id": "n7", "pred": ["n6"], "kind": "verify", "label": "安全校验", "task": "检查维修建议是否遗漏停用、人员安全或访问权限约束。"},
        {"id": "n8", "pred": ["n7"], "kind": "summarize", "label": "工单摘要", "task": "生成供巡检班组使用的结构化工单摘要。"},
    ],
}

NODE_MAX_TOKENS = {
    "intent": 64,
    "extract": 80,
    "retrieve": 128,
    "tool_call": 128,
    "plan": 320,
    "verify": 160,
    "summarize": 128,
}

SCENARIO_INPUTS = {
    "emergency": "某路口因持续暴雨出现明显积水，已经影响公交通行，请判断风险、查询附近排水设施和可用处置人员并生成处置建议。",
    "traffic": "早高峰某主干道发生车辆事故，两个相邻路口排队长度快速增加，请分析交通影响并给出疏导建议。",
    "municipal_service": "市民反映小区附近公共照明连续两晚故障，希望查询历史工单、确认责任部门并给出办理答复。",
    "facility_inspection": "城市泵站传感器报告异常振动，请查询设备历史维护信息，安排巡检并生成安全维修工单。",
}


def _extract_json(text: str) -> Any:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _gpu_memory() -> Dict[str, Any]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        used_text, total_text, util_text, power_text = [x.strip() for x in raw.splitlines()[0].split(",")]
        return {
            "used_mib": int(used_text),
            "total_mib": int(total_text),
            "utilization_percent": int(util_text),
            "power_w": None if power_text == "[N/A]" else float(power_text),
        }
    except Exception:
        return {}


async def generate_one(engine: Any, lora_request: Any, prompt: str, request_id: str, max_tokens: int) -> Dict[str, Any]:
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        output_kind=RequestOutputKind.DELTA,
    )
    started = time.perf_counter()
    first_token = None
    final_text = ""
    prompt_tokens = None
    output_tokens = 0
    async for output in engine.generate(request_id=request_id, prompt=prompt, sampling_params=sampling, lora_request=lora_request):
        if first_token is None and output.outputs and output.outputs[0].text:
            first_token = time.perf_counter()
        if output.outputs:
            completion = output.outputs[0]
            final_text += completion.text
            token_ids = getattr(completion, "token_ids", None)
            if token_ids is not None:
                output_tokens += len(token_ids)
        prompt_ids = getattr(output, "prompt_token_ids", None)
        if prompt_ids is not None:
            prompt_tokens = len(prompt_ids)
    finished = time.perf_counter()
    decode_time = None if first_token is None else max(0.0, finished - first_token)
    return {
        "text": final_text,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_s": None if first_token is None else first_token - started,
        "latency_s": finished - started,
        "tpot_s": None if output_tokens <= 1 or decode_time is None else decode_time / (output_tokens - 1),
        "decode_tokens_per_s": None if output_tokens <= 1 or not decode_time else (output_tokens - 1) / decode_time,
    }


async def run(args: argparse.Namespace) -> Path:
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine, lora_request, _ = get_async_model(
        model_path=args.model,
        lora_path=None,
        verbose=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    records: List[Dict[str, Any]] = []
    warmup = await generate_one(
        engine,
        lora_request,
        "<|im_start|>system\n只输出JSON，不进行思考。<|im_end|>\n<|im_start|>user\n输出{\"warmup\":true} /no_think<|im_end|>\n<|im_start|>assistant\n",
        "warmup",
        16,
    )
    print("warmup", json.dumps(warmup, ensure_ascii=False))
    for repeat in range(args.repeats):
        for scenario, nodes in WORKFLOWS.items():
            outputs_by_node: Dict[str, str] = {}
            workflow_started = time.perf_counter()
            for node in nodes:
                predecessor_context = "\n".join(
                    f"{pred}: {outputs_by_node.get(pred, '')}" for pred in node["pred"]
                ) or "无"
                user_content = (
                    f"场景：{scenario}\n用户请求：{SCENARIO_INPUTS[scenario]}\n"
                    f"节点：{node['id']}；节点类型：{node['kind']}；节点任务：{node['task']}\n"
                    f"前驱节点：{node['pred']}\n前驱结果：{predecessor_context[-6000:]}\n"
                    "请执行当前节点，只输出一个 JSON 对象。/no_think"
                )
                prompt = (
                    "<|im_start|>system\n你是智慧城市LLM-Agent的结构化执行节点。"
                    "禁止输出思考过程、解释和Markdown。JSON必须包含status、node_type和result字段。<|im_end|>\n"
                    f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
                )
                max_tokens = min(args.max_tokens, NODE_MAX_TOKENS[node["kind"]])
                metrics = await generate_one(engine, lora_request, prompt, f"{scenario}-{repeat}-{node['id']}", max_tokens)
                parsed = _extract_json(metrics["text"])
                required_keys_valid = isinstance(parsed, dict) and all(key in parsed for key in ("status", "node_type", "result"))
                record = {
                    "workflow_id": f"{scenario}-{repeat}",
                    "scenario": scenario,
                    "node_id": node["id"],
                    "node_type": node["kind"],
                    "node_label": node["label"],
                    "predecessors": node["pred"],
                    "repeat": repeat,
                    "endpoint": "laptop-wsl-vllm",
                    "model": args.model,
                    "max_tokens": max_tokens,
                    "prompt_chars": len(prompt),
                    "prompt_tokens": metrics["prompt_tokens"],
                    "output_tokens": metrics["output_tokens"],
                    "ttft_s": metrics["ttft_s"],
                    "latency_s": metrics["latency_s"],
                    "tpot_s": metrics["tpot_s"],
                    "decode_tokens_per_s": metrics["decode_tokens_per_s"],
                    "json_valid": isinstance(parsed, dict),
                    "required_keys_valid": required_keys_valid,
                    "raw_text": metrics["text"],
                    "gpu": _gpu_memory(),
                    "timestamp": time.time(),
                }
                records.append(record)
                outputs_by_node[node["id"]] = json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else metrics["text"]
                print(json.dumps({k: record[k] for k in ["scenario", "repeat", "node_id", "node_type", "prompt_tokens", "output_tokens", "ttft_s", "latency_s", "decode_tokens_per_s", "json_valid", "required_keys_valid"]}, ensure_ascii=False))
            print(f"workflow {scenario}-{repeat} latency_s={time.perf_counter() - workflow_started:.3f}")
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--output", default="data/prototype_validation/real_dag_calibration.jsonl")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    args = parser.parse_args()
    path = asyncio.run(run(args))
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
