"""CLI: cei-simulate — run reference fleet simulation and ablations."""

from __future__ import annotations

import argparse
import json
import sys

from cei.metrics import format_ablation_table, windowed_means
from cei.simulate import run_ablations, run_simulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CEI hierarchical MoE reference simulator")
    parser.add_argument("--steps", type=int, default=400, help="Requests per run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("learned", "local", "random", "heuristic", "ablate"),
        default="ablate",
        help="Single mode or full A0–A3 ablation table",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text table")
    args = parser.parse_args(argv)

    if args.mode == "ablate":
        results = run_ablations(steps=args.steps, seed=args.seed)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("CEI ablation results (A0 local / A1 random / A2 heuristic / A3 learned)")
            print(format_ablation_table(results))
            local_u = results["local"]["utility_mean"]
            learned_u = results["learned"]["utility_mean"]
            print(f"\nLearned vs local utility delta: {learned_u - local_u:+.4f}")
        return 0

    fleet, result = run_simulation(steps=args.steps, seed=args.seed, mode=args.mode)
    summary = result.summary()
    summary["windows"] = windowed_means(result)
    summary["learner_version"] = fleet.learner.version
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"mode={args.mode} steps={args.steps}")
        for k, v in summary.items():
            if k == "mode":
                continue
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
