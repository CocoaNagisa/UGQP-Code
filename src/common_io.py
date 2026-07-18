from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
                               
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
                                    
    ensure_dir(path.parent)
    if not rows:
                                  
        path.write_text("", encoding="utf-8")
        return
                           
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
                                 
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_episode_rows(rows: list[dict[str, Any]], policy_key: str = "policy") -> list[dict[str, Any]]:
                                      
    if not rows:
        return []
                 
    policies = sorted({str(r[policy_key]) for r in rows})
    summary = []
                     
    metric_keys = [
        "total_latency",
        "deadline_violation",
        "avg_quality",
        "success",
        "total_cost",
        "high_risk_local_ratio",
        "offload_ratio",
        "load_variance",
        "reward",
        "mean_chosen_queue_wait",
        "mean_observed_remote_queue",
        "max_observed_remote_queue",
    ]
    for policy in policies:
        subset = [r for r in rows if str(r[policy_key]) == policy]
        row = {"policy": policy, "episodes": len(subset)}
        for key in metric_keys:
                                         
            row[f"mean_{key}"] = sum(float(r[key]) for r in subset) / len(subset)
        summary.append(row)
    return summary
