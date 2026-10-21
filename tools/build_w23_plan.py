from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(REPO_ROOT / "AutoGPTQ"),
)

from auto_gptq.mixed_precision import (
    allocate_w2_w3,
    load_layer_scores,
    save_mixed_precision_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a transformer-block-level W2/W3 allocation "
            "plan from DLM layer sensitivity scores."
        )
    )

    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="JSON file containing one score per transformer block.",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
    )

    budget_group = parser.add_mutually_exclusive_group(
        required=True
    )

    budget_group.add_argument(
        "--num-w3",
        type=int,
        help="Number of transformer blocks assigned W3.",
    )

    budget_group.add_argument(
        "--average-bits",
        type=float,
        help="Target average block bit width in [2.0, 3.0].",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scores = load_layer_scores(
        score_file=args.scores,
        num_layers=args.num_layers,
    )

    plan = allocate_w2_w3(
        layer_scores=scores,
        num_w3_layers=args.num_w3,
        target_average_bits=args.average_bits,
    )

    output_path = save_mixed_precision_plan(
        plan=plan,
        output_file=args.output,
    )

    print("=" * 72)
    print("W2/W3 MIXED-PRECISION PLAN")
    print("=" * 72)
    print("score name:", plan.score_name)
    print("score source:", plan.score_source)
    print("higher score is more sensitive:",
          plan.higher_is_more_sensitive)
    print("ranking:", list(plan.ranking))
    print("W2 layers:", list(plan.w2_layers))
    print("W3 layers:", list(plan.w3_layers))
    print("average bits:", plan.actual_average_bits)
    print("saved:", output_path)
    print("=" * 72)


if __name__ == "__main__":
    main()
