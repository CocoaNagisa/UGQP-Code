from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiments_20260716.v12l_r9_controlled_core_causal_ablation import run as base_run
from experiments_20260717.complete_core_ablation_extension import run as extension_run


def invoke(module, arguments: list[str]) -> None:
    previous = sys.argv
    try:
        sys.argv = [module.__file__, *arguments]
        module.main()
    finally:
        sys.argv = previous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base = args.output / "base"
    extension = args.output / "extension"
    common = ["--protocol", args.protocol]
    if args.resume:
        common.append("--resume")
    invoke(base_run, [*common, "--root", str(base)])
    extension_run.FULL_REFERENCE_ROOT = base.resolve()
    invoke(extension_run, [*common, "--root", str(extension)])


if __name__ == "__main__":
    main()

