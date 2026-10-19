from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path

from .taskgen import Node


def _difficulty_band(node: Node) -> str:
    explicit = getattr(node, "difficulty", None)
    if explicit in {"low", "medium", "high"}:
        return explicit
    uncertainty = float(getattr(node, "uncertainty", 0.0))
    return "low" if uncertainty < 0.40 else "medium" if uncertainty < 0.70 else "high"


@dataclass
class ModelProfile:
    name: str
    prefill_rate: float
    decode_rate: float
    base_latency: float
    quality_base: float
    token_cost: float
    privacy_allow: int
    capability: float
    queue_load: float = 0.0
    prefill_backlog: float = 0.0
    decode_backlog: float = 0.0
    eta_pre: float = 0.88
    eta_dec: float = 0.92
    quality_by_kind: dict[str, float] | None = None
    latency_by_kind: dict[str, float] | None = None
    quality_by_kind_difficulty: dict[str, dict[str, float]] | None = None
    latency_by_kind_difficulty: dict[str, dict[str, float]] | None = None
    token_cost_by_kind_difficulty: dict[str, dict[str, float]] | None = None
    empirical_endpoint: bool = False
    calibrated_endpoint: bool = False

    def clone(self) -> "ModelProfile":
        return ModelProfile(**self.__dict__)


def default_profiles(profile_mode: str = "tradeoff_v2") -> dict[str, ModelProfile]:
    if profile_mode == "v5_bridge":
        profile_path = Path(__file__).resolve().parents[2] / "endpoint_profile.json"
        data = json.loads(profile_path.read_text(encoding="utf-8"))["models"]
        result = {}
        for action, model_name in (("local", "1.7B"), ("edge_s", "8B")):
            model = data[model_name]
            result[action] = ModelProfile(
                name=action,
                prefill_rate=float(model["prefill_rate"]),
                decode_rate=float(model["decode_rate"]),
                base_latency=float(model["base_latency"]),
                quality_base=float(model["mean_field_accuracy"]),
                token_cost=float(model["normalized_model_work_ratio"]),
                privacy_allow=3,
                capability=float(model["mean_field_accuracy"]),
                quality_by_kind=model["quality_by_kind"],
                latency_by_kind=model["median_latency_by_kind"],
                quality_by_kind_difficulty=model.get("quality_by_kind_difficulty"),
                latency_by_kind_difficulty=model.get("latency_by_kind_difficulty"),
                token_cost_by_kind_difficulty=model.get("token_cost_by_kind_difficulty"),
                calibrated_endpoint=True,
            )
        return result
    if profile_mode == "e2_empirical":
        profile_path = Path(__file__).resolve().parents[2] / "e2_calibrated_profiles.json"
        data = json.loads(profile_path.read_text(encoding="utf-8"))["models"]
        result = {}
        for action, model_name in (("local", "1.7B"), ("edge_s", "8B")):
            model = data[model_name]
            result[action] = ModelProfile(
                name=action,
                prefill_rate=1e6,
                decode_rate=1e6,
                base_latency=0.0,
                quality_base=float(model["mean_field_accuracy"]),
                token_cost=0.0,
                privacy_allow=3,
                capability=float(model["mean_field_accuracy"]),
                quality_by_kind=model["quality_by_kind"],
                latency_by_kind=model["median_latency_by_kind"],
                empirical_endpoint=True,
            )
        return result
    if profile_mode == "legacy_v1":
        return {
            "local": ModelProfile("local", 1800.0, 55.0, 0.040, 0.72, 0.05, 3, 0.46),
            "edge_s": ModelProfile("edge_s", 3000.0, 90.0, 0.080, 0.84, 0.30, 2, 0.70),
            "edge_m": ModelProfile("edge_m", 4200.0, 120.0, 0.120, 0.90, 0.45, 2, 0.84),
            "edge_l": ModelProfile("edge_l", 5200.0, 150.0, 0.180, 0.94, 0.65, 2, 0.94),
            "cloud": ModelProfile("cloud", 7000.0, 220.0, 0.350, 0.97, 1.00, 1, 1.00),
        }
    if profile_mode != "tradeoff_v2":
        raise ValueError(f"Unsupported profile_mode: {profile_mode}")

    return {
        "local": ModelProfile("local", 1800.0, 55.0, 0.040, 0.72, 0.05, 3, 0.46),
        "edge_s": ModelProfile("edge_s", 4800.0, 160.0, 0.080, 0.84, 0.30, 2, 0.70),
        "edge_m": ModelProfile("edge_m", 4000.0, 115.0, 0.120, 0.90, 0.45, 2, 0.84),
        "edge_l": ModelProfile("edge_l", 3000.0, 75.0, 0.180, 0.94, 0.65, 2, 0.94),
        "cloud": ModelProfile("cloud", 6500.0, 180.0, 0.350, 0.97, 1.00, 1, 1.00),
    }


def feasible(node: Node, profile: ModelProfile) -> bool:
    if profile.empirical_endpoint or profile.calibrated_endpoint:
        return True
    if node.privacy > profile.privacy_allow:
        return False
    if profile.name == "local" and node.kind in {"plan", "generate"} and node.uncertainty > 0.32:
        return False
    if profile.name == "local" and node.input_tokens + node.output_tokens > 900:
        return False
    if profile.capability + 0.16 < node.risk:
        return False
    return True


def serving_wait(profile: ModelProfile) -> float:
    if profile.name == "local" and not profile.calibrated_endpoint:
        return 0.0
    pre = profile.prefill_backlog / max(profile.eta_pre * profile.prefill_rate, 1e-6)
    dec = profile.decode_backlog / max(profile.eta_dec * profile.decode_rate, 1e-6)
    return max(0.0, pre + dec)


def refresh_queue_load(profile: ModelProfile) -> None:
    profile.queue_load = serving_wait(profile)


def _add_wait_workload(profile: ModelProfile, wait_seconds: float, prefill_share: float) -> None:
    """Convert a waiting-time increment into prefill/decode token backlog."""
    if (profile.name == "local" and not profile.calibrated_endpoint) or wait_seconds <= 0.0:
        return
    prefill_share = min(1.0, max(0.0, prefill_share))
    profile.prefill_backlog += wait_seconds * prefill_share * profile.eta_pre * profile.prefill_rate
    profile.decode_backlog += wait_seconds * (1.0 - prefill_share) * profile.eta_dec * profile.decode_rate


def initialize_background_queues(
    profiles: dict[str, ModelProfile],
    rng: random.Random,
    arrival_rate: float,
    queue_mode: str,
    queue_scale: float,
) -> dict[str, int]:
    """Initialize reproducible exogenous multi-tenant serving workload."""
    if queue_mode not in {"legacy", "bursty"}:
        raise ValueError(f"Unsupported queue_mode: {queue_mode}")

    burst_remaining = {name: 0 for name in profiles}
    if queue_mode == "legacy":
        return burst_remaining

    heterogeneity = {
        "local": 1.00,
        "edge_s": 1.18,
        "edge_m": 1.00,
        "edge_l": 0.88,
        "cloud": 0.78,
    }
    base_wait = queue_scale * (0.25 + 1.25 * max(arrival_rate, 0.0))
    for name, profile in profiles.items():
        if name == "local" and not profile.calibrated_endpoint:
            continue
        target_wait = base_wait * heterogeneity[name] * rng.uniform(0.35, 1.55)
        _add_wait_workload(profile, target_wait, rng.uniform(0.28, 0.44))
        refresh_queue_load(profile)
    return burst_remaining


def advance_background_queues(
    profiles: dict[str, ModelProfile],
    rng: random.Random,
    arrival_rate: float,
    queue_mode: str,
    queue_scale: float,
    burst_probability: float,
    burst_multiplier: float,
    burst_min_steps: int,
    burst_max_steps: int,
    burst_remaining: dict[str, int],
) -> None:
    """Advance background queues by one scheduling step.

    Existing work is drained first. New external requests then arrive at all
    remote locations. Persistent per-server bursts give the queue observation
    a causal routing role while preserving a fixed load path for each seed.
    """
    if queue_mode == "legacy":
        decay_queues(profiles)
        return
    if queue_mode != "bursty":
        raise ValueError(f"Unsupported queue_mode: {queue_mode}")
    if burst_min_steps < 1 or burst_max_steps < burst_min_steps:
        raise ValueError("Invalid queue burst duration range")

    heterogeneity = {
        "local": 1.00,
        "edge_s": 1.18,
        "edge_m": 1.00,
        "edge_l": 0.88,
        "cloud": 0.78,
    }
    base_increment = queue_scale * (0.10 + 0.22 * max(arrival_rate, 0.0))

    for name, profile in profiles.items():
        if name == "local" and not profile.calibrated_endpoint:
            continue

        profile.prefill_backlog *= 0.76
        profile.decode_backlog *= 0.82

        remaining = int(burst_remaining.get(name, 0))
        if remaining <= 0 and rng.random() < burst_probability:
            remaining = rng.randint(burst_min_steps, burst_max_steps)
        multiplier = burst_multiplier if remaining > 0 else 1.0

        wait_increment = (
            base_increment
            * heterogeneity[name]
            * rng.uniform(0.65, 1.35)
            * multiplier
        )
        _add_wait_workload(profile, wait_increment, rng.uniform(0.28, 0.44))
        refresh_queue_load(profile)
        burst_remaining[name] = max(0, remaining - 1)


def latency(node: Node, profile: ModelProfile, rng: random.Random, network_scale: float = 1.0) -> float:
    if profile.calibrated_endpoint:
        by_diff = profile.latency_by_kind_difficulty or {}
        base = float((by_diff.get(node.kind) or {}).get(_difficulty_band(node),
                     (profile.latency_by_kind or {}).get(node.kind, profile.base_latency)))
        # Calibration latency is the service component; queue and a small
        # reproducible jitter remain active so routing observes resource load.
        return max(0.001, (base + serving_wait(profile)) * rng.uniform(0.94, 1.10))
    if profile.empirical_endpoint:
        return float(profile.latency_by_kind[node.kind])
    comm = 0.0
    if profile.name.startswith("edge"):
        bandwidth = rng.uniform(8.0, 95.0) * network_scale
        comm = (node.input_tokens + 0.22 * node.output_tokens) / max(bandwidth * 900.0, 1.0)
    elif profile.name == "cloud":
        bandwidth = rng.uniform(12.0, 110.0) * network_scale
        comm = 0.20 + (node.input_tokens + 0.28 * node.output_tokens) / max(bandwidth * 750.0, 1.0)
    queue_wait = serving_wait(profile)
    prefill = profile.base_latency + node.input_tokens / profile.prefill_rate
    decode = node.output_tokens / profile.decode_rate
    return max(0.001, (comm + queue_wait + prefill + decode) * rng.uniform(0.94, 1.10))


def expected_latency(node: Node, profile: ModelProfile, network_scale: float = 1.0) -> float:
    """Deterministic latency estimate for decision-time heuristic baselines.

    The estimate uses the mean configured bandwidth and no serving jitter. It
    reads the live queue backlog, so it is recomputed after every DAG decision.
    """
    if profile.calibrated_endpoint:
        by_diff = profile.latency_by_kind_difficulty or {}
        return float((by_diff.get(node.kind) or {}).get(_difficulty_band(node),
                     (profile.latency_by_kind or {}).get(node.kind, profile.base_latency)) + serving_wait(profile))
    if profile.empirical_endpoint:
        return float(profile.latency_by_kind[node.kind])
    comm = 0.0
    if profile.name.startswith("edge"):
        mean_bandwidth = 0.5 * (8.0 + 95.0) * network_scale
        comm = (node.input_tokens + 0.22 * node.output_tokens) / max(
            mean_bandwidth * 900.0,
            1.0,
        )
    elif profile.name == "cloud":
        mean_bandwidth = 0.5 * (12.0 + 110.0) * network_scale
        comm = 0.20 + (node.input_tokens + 0.28 * node.output_tokens) / max(
            mean_bandwidth * 750.0,
            1.0,
        )

    queue_wait = serving_wait(profile)
    prefill = profile.base_latency + node.input_tokens / profile.prefill_rate
    decode = node.output_tokens / profile.decode_rate
    return max(0.001, comm + queue_wait + prefill + decode)


def quality(node: Node, profile: ModelProfile) -> float:
    if profile.calibrated_endpoint:
        by_diff = profile.quality_by_kind_difficulty or {}
        kind_values = by_diff.get(node.kind, {})
        band = _difficulty_band(node)
        value = kind_values.get(band)
        if value is None:
            value = (profile.quality_by_kind or {}).get(node.kind, profile.quality_base)
        return float(min(1.0, max(0.0, value)))
    if profile.empirical_endpoint:
        return float(profile.quality_by_kind[node.kind])
    complexity_penalty = 0.11 if node.kind in {"plan", "generate"} else 0.07 if node.kind == "verify" else 0.035
    uncertainty_penalty = node.uncertainty * (0.40 if profile.name == "local" else 0.17 if profile.name.startswith("edge") else 0.10)
    tool_penalty = 0.04 if node.kind == "tool_call" and profile.name == "local" else 0.0
    privacy_bonus = 0.025 if node.privacy == 3 and profile.name == "local" else 0.0
    return min(1.0, max(0.0, profile.quality_base - complexity_penalty - uncertainty_penalty - tool_penalty + privacy_bonus))


def cost(node: Node, profile: ModelProfile) -> float:
    if profile.calibrated_endpoint:
        tokens = node.input_tokens + node.output_tokens
        by_diff = profile.token_cost_by_kind_difficulty or {}
        ratio = float((by_diff.get(node.kind) or {}).get(_difficulty_band(node), profile.token_cost))
        return ratio * tokens / 1000.0
    if profile.empirical_endpoint:
        return 0.0
    network_cost = 0.0 if profile.name == "local" else 0.03 * (node.input_tokens + node.output_tokens) / 1000.0
    if profile.name == "cloud":
        network_cost *= 2.4
    return profile.token_cost * (node.input_tokens + node.output_tokens) / 1000.0 + network_cost


def update_queue(profile: ModelProfile, node: Node, service_time: float, arrival_rate: float) -> None:
    if profile.empirical_endpoint:
        return
    if profile.name == "local" and not profile.calibrated_endpoint:
        return
    pre_growth = node.input_tokens * (0.16 + 0.05 * arrival_rate)
    dec_growth = node.output_tokens * (0.20 + 0.08 * arrival_rate)
    served_pre = service_time * profile.prefill_rate * profile.eta_pre * 0.10
    served_dec = service_time * profile.decode_rate * profile.eta_dec * 0.10
    profile.prefill_backlog = max(0.0, profile.prefill_backlog * 0.86 + pre_growth - served_pre)
    profile.decode_backlog = max(0.0, profile.decode_backlog * 0.88 + dec_growth - served_dec)
    refresh_queue_load(profile)


def decay_queues(profiles: dict[str, ModelProfile]) -> None:
    for profile in profiles.values():
        if profile.name != "local" or profile.calibrated_endpoint:
            profile.prefill_backlog *= 0.90
            profile.decode_backlog *= 0.92
            refresh_queue_load(profile)
