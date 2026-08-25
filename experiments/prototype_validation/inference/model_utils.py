"""Provide the asynchronous vLLM engine used by the prototype scripts."""

from __future__ import annotations

from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


_ENGINE: AsyncLLM | None = None


def get_async_model(
    model_path: str,
    lora_path: str | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> tuple[AsyncLLM, None, None]:
    """Return a reusable in-process vLLM engine."""
    if lora_path is not None:
        raise ValueError("The released prototype protocol does not use LoRA adapters")
    global _ENGINE
    if _ENGINE is None:
        engine_args = AsyncEngineArgs(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=float(kwargs.pop("gpu_memory_utilization", 0.60)),
            max_model_len=int(kwargs.pop("max_model_len", 2048)),
            dtype=str(kwargs.pop("dtype", "float16")),
            enable_prefix_caching=bool(kwargs.pop("enable_prefix_caching", False)),
            **kwargs,
        )
        _ENGINE = AsyncLLM.from_engine_args(engine_args)
        if verbose:
            print(f"Loaded vLLM model: {model_path}")
    return _ENGINE, None, None
