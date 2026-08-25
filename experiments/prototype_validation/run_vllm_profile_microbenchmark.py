"""Fit simulator-compatible prefill/decode rates from a real vLLM engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from inference.model_utils import get_async_model


def ols(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / max(denominator, 1e-12)
    intercept = y_mean - slope * x_mean
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - residual / max(total, 1e-12)
    return intercept, slope, r2


def build_prompt(tokenizer: Any, target_tokens: int, nonce: str) -> Tuple[str, int]:
    system = "<|im_start|>system\n你是吞吐测试节点，只按要求生成。<|im_end|>\n"
    suffix = "\n只回复数字1。/no_think<|im_end|>\n<|im_start|>assistant\n"
    unit = f" 城市事件{nonce}包含道路积水交通设施工单巡检数据。"
    body = "<|im_start|>user\n"
    while len(tokenizer.encode(system + body + suffix)) < target_tokens:
        body += unit
    prompt = system + body + suffix
    return prompt, len(tokenizer.encode(prompt))


async def measure(engine: Any, prompt: str, request_id: str, output_tokens: int) -> Dict[str, Any]:
    params = SamplingParams(
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
        temperature=0.0,
        top_p=1.0,
        output_kind=RequestOutputKind.DELTA,
    )
    started = time.perf_counter()
    first = None
    actual_output = 0
    prompt_tokens = None
    async for output in engine.generate(request_id=request_id, prompt=prompt, sampling_params=params, lora_request=None):
        if output.outputs and output.outputs[0].text and first is None:
            first = time.perf_counter()
        if output.outputs:
            token_ids = getattr(output.outputs[0], "token_ids", None)
            if token_ids is not None:
                actual_output += len(token_ids)
        prompt_ids = getattr(output, "prompt_token_ids", None)
        if prompt_ids is not None:
            prompt_tokens = len(prompt_ids)
    finished = time.perf_counter()
    ttft = None if first is None else first - started
    decode_time = None if first is None else finished - first
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": actual_output,
        "ttft_s": ttft,
        "latency_s": finished - started,
        "decode_tokens_per_s": None if not decode_time or actual_output <= 1 else (actual_output - 1) / decode_time,
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine, _, _ = get_async_model(
        model_path=args.model,
        lora_path=None,
        verbose=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
    )
    warmup_prompt, _ = build_prompt(tokenizer, 64, "warmup")
    await measure(engine, warmup_prompt, "warmup", 8)

    rows: List[Dict[str, Any]] = []
    for target in args.input_lengths:
        for repeat in range(args.repeats):
            prompt, actual = build_prompt(tokenizer, target, f"prefill-{target}-{repeat}")
            result = await measure(engine, prompt, f"prefill-{target}-{repeat}", 1)
            rows.append({"mode": "prefill", "target_input_tokens": target, "repeat": repeat, **result})
            print(json.dumps(rows[-1], ensure_ascii=False))

    decode_prompt, _ = build_prompt(tokenizer, args.decode_input_tokens, "decode")
    for target in args.output_lengths:
        for repeat in range(args.repeats):
            result = await measure(engine, decode_prompt + f"\n批次{target}-{repeat}", f"decode-{target}-{repeat}", target)
            rows.append({"mode": "decode", "target_output_tokens": target, "repeat": repeat, **result})
            print(json.dumps(rows[-1], ensure_ascii=False))

    prefill_rows = [row for row in rows if row["mode"] == "prefill" and row["ttft_s"] is not None]
    prefill_intercept, prefill_slope, prefill_r2 = ols(
        [float(row["prompt_tokens"]) for row in prefill_rows],
        [float(row["ttft_s"]) for row in prefill_rows],
    )
    decode_rows = [row for row in rows if row["mode"] == "decode"]
    decode_intercept, decode_slope, decode_r2 = ols(
        [float(row["output_tokens"]) for row in decode_rows],
        [float(row["latency_s"] - row["ttft_s"]) for row in decode_rows],
    )
    summary = {
        "model": args.model,
        "endpoint": "laptop-wsl-vllm-inprocess",
        "prefix_caching": False,
        "repeats": args.repeats,
        "prefill_fit": {
            "base_latency_s": prefill_intercept,
            "seconds_per_input_token": prefill_slope,
            "prefill_tokens_per_s": None if prefill_slope <= 0 else 1.0 / prefill_slope,
            "r2": prefill_r2,
        },
        "decode_fit": {
            "decode_intercept_s": decode_intercept,
            "seconds_per_output_token": decode_slope,
            "decode_tokens_per_s": None if decode_slope <= 0 else 1.0 / decode_slope,
            "r2": decode_r2,
        },
        "rows": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--output", default="data/prototype_validation/vllm_profile_microbenchmark.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[64, 128, 256, 512, 768])
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--decode-input-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
