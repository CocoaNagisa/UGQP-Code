"""Check domain workflow descriptors against the sampled graph range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("node_count", "edge_count", "max_depth_nodes", "branch_nodes", "merge_nodes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    root = args.data
    templates = json.loads((root / "domain_templates.json").read_text(encoding="utf-8"))["templates"]
    template_stats = pd.read_csv(root / "domain_template_stats.csv")
    summary = pd.read_csv(root / "synthetic_structure_summary.csv").set_index("metric")
    checks = []
    for template in templates:
        row = template_stats[template_stats["scenario"] == template["scenario"]].iloc[0]
        for metric in METRICS:
            lower, upper = summary.loc[metric, ["q05", "q95"]]
            value = float(row[metric])
            checks.append({"scenario": template["scenario"], "metric": metric, "within": lower <= value <= upper})
    result = pd.DataFrame(checks)
    if not result["within"].all():
        raise SystemExit("A workflow descriptor falls outside the sampled interval")
    print(f"Checked {len(templates)} workflow templates and {len(result)} descriptors.")


if __name__ == "__main__":
    main()
