from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(REPO_ROOT / "AutoGPTQ"),
)

from auto_gptq.mixed_precision import (
    build_sandwich_plan,
    load_sandwich_config,
    save_sandwich_plan,
)


def main() -> None:
    os.chdir(REPO_ROOT)

    config = load_sandwich_config(
        REPO_ROOT
        / "configs"
        / "qdlm_sandwich_w23.json"
    )

    # Quantized weights per LLaDA block:
    # 4 x (4096 -> 4096)
    # 2 x (4096 -> 12288)
    # 1 x (12288 -> 4096)
    block_weight_count = (
        4 * 4096 * 4096
        + 2 * 4096 * 12288
        + 12288 * 4096
    )

    plan = build_sandwich_plan(
        config=config,
        layer_weights=[
            block_weight_count
            for _ in range(32)
        ],
    )

    assert plan.current_w3_layers == tuple(
        range(12, 20)
    ), plan.current_w3_layers

    assert plan.current_w2_layers == (
        tuple(range(0, 12))
        + tuple(range(20, 32))
    )

    assert abs(
        plan.current_average_bits - 2.25
    ) < 1e-12

    assert plan.final_w1_layers == (
        0,
        1,
        2,
        3,
        28,
        29,
        30,
        31,
    ), plan.final_w1_layers

    assert plan.final_w3_layers == tuple(
        range(12, 20)
    )

    assert abs(
        plan.final_preview_average_bits - 2.0
    ) < 1e-12

    output = save_sandwich_plan(
        plan=plan,
        output_file=config.plan_output,
    )

    current_histogram = Counter(
        plan.current_bits
    )

    final_histogram = Counter(
        plan.final_preview_bits
    )

    print("=" * 72)
    print("DLM SANDWICH GPTQ: STRUCTURAL CHECK PASS")
    print("=" * 72)
    print(
        "current W2 layers:",
        list(plan.current_w2_layers),
    )
    print(
        "current W3 layers:",
        list(plan.current_w3_layers),
    )
    print(
        "current block histogram:",
        dict(sorted(current_histogram.items())),
    )
    print(
        "current average bits:",
        plan.current_average_bits,
    )
    print()
    print(
        "final W1 layers:",
        list(plan.final_w1_layers),
    )
    print(
        "final W2 layers:",
        list(plan.final_w2_layers),
    )
    print(
        "final W3 layers:",
        list(plan.final_w3_layers),
    )
    print(
        "final block histogram:",
        dict(sorted(final_histogram.items())),
    )
    print(
        "final preview average bits:",
        plan.final_preview_average_bits,
    )
    print()
    print("plan:", output)
    print("=" * 72)


if __name__ == "__main__":
    main()
