"""Measure concurrent vLLM behavior without modifying the deployed service."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from inference.model_utils import get_async_model


def build_prompt(tokenizer: Any, target_tokens: int, nonce: str) -> Tuple[str, int]:
    """Build a non-cached prompt close to a target token count."""
    system = "<|im_start|>system\n你是智慧城市服务节点。<|im_end|>\n"
    suffix = "\n连续输出简短处置要点。/no_think<|im_end|>\n<|im_start|>assistant\n"
    unit = f" 城市任务{nonce}包含道路设施工单风险约束与调度信息。"
    body = "<|im_start|>user\n"
    while len(tokenizer.encode(system + body + suffix)) < target_tokens:
        body += unit
    prompt = system + body + suffix
    return prompt, len(tokenizer.encode(prompt))


async def measure_request(
    engine: Any,
    prompt: str,
    request_id: str,
    output_tokens: int,
) -> Dict[str, Any]:
    """Measure one request submitted to an existing in-process engine."""
    params = SamplingParams(
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
        temperature=0.0,
        top_p=1.0,
        output_kind=RequestOutputKind.DELTA,
    )
    started = time.perf_counter()
    first_token = None
    prompt_tokens = None
    generated_tokens = 0
    async for output in engine.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=params,
        lora_request=None,
    ):
        if output.outputs and output.outputs[0].text and first_token is None:
            first_token = time.perf_counter()
        if output.outputs:
            token_ids = getattr(output.outputs[0], "token_ids", None)
            if token_ids is not None:
                generated_tokens += len(token_ids)
        prompt_ids = getattr(output, "prompt_token_ids", None)
        if prompt_ids is not None:
            prompt_tokens = len(prompt_ids)
    finished = time.perf_counter()
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": generated_tokens,
        "ttft_s": None if first_token is None else first_token - started,
        "latency_s": finished - started,
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run fixed-size request batches at several concurrency levels."""
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine, _, _ = get_async_model(
        model_path=args.model,
        lora_path=None,
        verbose=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
    )
    warmup_prompt, _ = build_prompt(tokenizer, 128, "warmup")
    await measure_request(engine, warmup_prompt, "warmup", 16)

    rows: List[Dict[str, Any]] = []
    batches: List[Dict[str, Any]] = []
    for concurrency in args.concurrency:
        for repeat in range(args.repeats):
            requests = []
            for index in range(concurrency):
                nonce = f"c{concurrency}-r{repeat}-i{index}"
                prompt, _ = build_prompt(tokenizer, args.input_tokens, nonce)
                requests.append(
                    measure_request(
                        engine,
                        prompt,
                        nonce,
                        args.output_tokens,
                    )
                )
            batch_started = time.perf_counter()
            results = await asyncio.gather(*requests)
            batch_elapsed = time.perf_counter() - batch_started
            total_output = sum(int(row["output_tokens"]) for row in results)
            batch = {
                "concurrency": concurrency,
                "repeat": repeat,
                "batch_elapsed_s": batch_elapsed,
                "aggregate_decode_tokens_per_s": total_output / batch_elapsed,
            }
            batches.append(batch)
            for index, result in enumerate(results):
                rows.append(
                    {
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "request_index": index,
                        **result,
                    }
                )
            print(json.dumps(batch, ensure_ascii=False))

    single_latency = statistics.mean(
        float(row["latency_s"]) for row in rows if row["concurrency"] == 1
    )
    single_throughput = statistics.mean(
        float(row["aggregate_decode_tokens_per_s"])
        for row in batches
        if row["concurrency"] == 1
    )
    by_concurrency: Dict[str, Dict[str, float]] = {}
    for concurrency in args.concurrency:
        selected_rows = [row for row in rows if row["concurrency"] == concurrency]
        selected_batches = [row for row in batches if row["concurrency"] == concurrency]
        mean_throughput = statistics.mean(
            float(row["aggregate_decode_tokens_per_s"]) for row in selected_batches
        )
        by_concurrency[str(concurrency)] = {
            "mean_ttft_s": statistics.mean(float(row["ttft_s"]) for row in selected_rows),
            "mean_latency_s": statistics.mean(float(row["latency_s"]) for row in selected_rows),
            "mean_queue_delay_proxy_s": statistics.mean(
                max(0.0, float(row["latency_s"]) - single_latency) for row in selected_rows
            ),
            "aggregate_decode_tokens_per_s": mean_throughput,
            "ideal_linear_efficiency": mean_throughput / max(concurrency * single_throughput, 1e-12),
        }
    return {
        "model": args.model,
        "endpoint": "laptop-wsl-vllm-inprocess-isolated-test",
        "input_tokens_target": args.input_tokens,
        "output_tokens_target": args.output_tokens,
        "repeats": args.repeats,
        "by_concurrency": by_concurrency,
        "batches": batches,
        "rows": rows,
    }


def main() -> None:
    """Parse arguments and write the concurrency calibration result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--output", default="data/prototype_validation/vllm_concurrency_calibration.json")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("WROTE", output)


if __name__ == "__main__":
    main()
