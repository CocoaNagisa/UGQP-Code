"""Recompute the normalized utility checks from the main metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("success", "quality", "latency", "deadline", "cost", "risk")
WEIGHTS = np.asarray((0.30, 0.30, 0.15, 0.10, 0.10, 0.05), dtype=float)


def load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(METRICS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing metrics: {', '.join(missing)}")
    return frame.set_index("method")


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for metric in METRICS:
        low, high = frame[metric].min(), frame[metric].max()
        if np.isclose(low, high):
            result[metric] = 0.5
        elif metric in {"latency", "deadline", "cost", "risk"}:
            result[metric] = (high - frame[metric]) / (high - low)
        else:
            result[metric] = (frame[metric] - low) / (high - low)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT.parent.parent / "data" / "utility_analysis" / "utility_final_nine_method_metrics.csv",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = load_metrics(args.metrics)
    components = normalize(metrics)
    utilities = components.to_numpy() @ WEIGHTS * 100.0
    result = pd.DataFrame({"method": components.index, "utility": utilities})
    result["rank"] = result["utility"].rank(ascending=False, method="min").astype(int)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
