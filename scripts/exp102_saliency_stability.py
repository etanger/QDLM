import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


INPUT = Path(
    "step_logs/activation_saliency/"
    "exp101_activation_saliency_map_limit2.jsonl"
)

OUTPUT = Path(
    "step_logs/activation_saliency/"
    "exp102_saliency_stability.json"
)

LAYERS = [0, 4, 8, 12, 16, 20, 24, 28]
MODULES = [
    "attn_out",
    "ff_proj",
    "up_proj",
    "ff_out",
]
STEPS = [0, 16, 32, 48, 63]


def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0

    return len(a & b) / len(a | b)


examples = []

with INPUT.open() as f:
    for line in f:
        if line.strip():
            examples.append(
                json.loads(line)
            )


# records_by_example[e][(layer, module, step)] = stats
records_by_example = []

for obj in examples:
    cur = {}

    for r in obj["records"]:
        stats = (
            r["masked"]
            if r["masked"] is not None
            else r["all"]
        )

        cur[
            (
                r["layer"],
                r["module"],
                r["step"],
            )
        ] = stats

    records_by_example.append(cur)


summary = {}


for layer in LAYERS:
    summary[str(layer)] = {}

    for module in MODULES:

        # ------------------------------------------------------------
        # 1. Global top-1 dominance:
        #    Across all samples and timesteps, how often is the same
        #    channel the top-1 channel?
        # ------------------------------------------------------------
        all_top1 = []

        for ex in records_by_example:
            for step in STEPS:
                s = ex[
                    (
                        layer,
                        module,
                        step,
                    )
                ]

                all_top1.append(
                    s["top_indices"][0]
                )

        global_counter = Counter(
            all_top1
        )

        global_top1_channel, global_top1_count = (
            global_counter.most_common(1)[0]
        )

        global_top1_frequency = (
            global_top1_count
            / len(all_top1)
        )


        # ------------------------------------------------------------
        # 2. Temporal stability within each example.
        #
        #    top1_mode_fraction:
        #      fraction of the 5 timesteps occupied by the most common
        #      top-1 channel.
        #
        #    top5_jaccard:
        #      average pairwise Jaccard among top-5 channel sets
        #      across the 5 timesteps.
        #
        #    top1pct_jaccard:
        #      same, but for top 1% channels.
        # ------------------------------------------------------------
        temporal_top1_mode_fraction = []
        temporal_top5_jaccard = []
        temporal_top1pct_jaccard = []

        for ex in records_by_example:
            top1_by_step = []
            top5_by_step = []
            top1pct_by_step = []

            for step in STEPS:
                s = ex[
                    (
                        layer,
                        module,
                        step,
                    )
                ]

                inds = s["top_indices"]
                dim = s["dim"]

                k_1pct = max(
                    1,
                    (dim + 99) // 100,
                )

                top1_by_step.append(
                    inds[0]
                )

                top5_by_step.append(
                    inds[:5]
                )

                top1pct_by_step.append(
                    inds[:k_1pct]
                )

            c = Counter(
                top1_by_step
            )

            temporal_top1_mode_fraction.append(
                c.most_common(1)[0][1]
                / len(STEPS)
            )

            pair_top5 = []
            pair_top1pct = []

            for a, b in combinations(
                range(len(STEPS)),
                2,
            ):
                pair_top5.append(
                    jaccard(
                        top5_by_step[a],
                        top5_by_step[b],
                    )
                )

                pair_top1pct.append(
                    jaccard(
                        top1pct_by_step[a],
                        top1pct_by_step[b],
                    )
                )

            temporal_top5_jaccard.append(
                np.mean(pair_top5)
            )

            temporal_top1pct_jaccard.append(
                np.mean(pair_top1pct)
            )


        # ------------------------------------------------------------
        # 3. Cross-sample stability.
        #
        #    At each timestep:
        #      which top-1 channel occurs most often across examples?
        #
        #    Then average that dominant frequency across timesteps.
        # ------------------------------------------------------------
        cross_sample_top1_frequency = []

        for step in STEPS:
            top1_at_step = []

            for ex in records_by_example:
                s = ex[
                    (
                        layer,
                        module,
                        step,
                    )
                ]

                top1_at_step.append(
                    s["top_indices"][0]
                )

            c = Counter(
                top1_at_step
            )

            dominant_count = (
                c.most_common(1)[0][1]
            )

            cross_sample_top1_frequency.append(
                dominant_count
                / len(top1_at_step)
            )


        result = {
            "global_top1_channel": int(
                global_top1_channel
            ),

            "global_top1_frequency": float(
                global_top1_frequency
            ),

            "temporal_top1_mode_fraction": float(
                np.mean(
                    temporal_top1_mode_fraction
                )
            ),

            "temporal_top5_jaccard": float(
                np.mean(
                    temporal_top5_jaccard
                )
            ),

            "temporal_top1pct_jaccard": float(
                np.mean(
                    temporal_top1pct_jaccard
                )
            ),

            "cross_sample_top1_frequency": float(
                np.mean(
                    cross_sample_top1_frequency
                )
            ),
        }

        summary[str(layer)][module] = result


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    f"Loaded examples: {len(examples)}"
)

for module in MODULES:
    print()
    print("=" * 105)
    print(module)
    print("=" * 105)

    print(
        f"{'Layer':>6} "
        f"{'Top1 global':>12} "
        f"{'Time Top1':>12} "
        f"{'Time J@5':>12} "
        f"{'Time J@1%':>12} "
        f"{'Sample Top1':>12} "
        f"{'Top1 ID':>10}"
    )

    for layer in LAYERS:
        r = summary[
            str(layer)
        ][module]

        print(
            f"{layer:6d} "
            f"{r['global_top1_frequency']:12.3f} "
            f"{r['temporal_top1_mode_fraction']:12.3f} "
            f"{r['temporal_top5_jaccard']:12.3f} "
            f"{r['temporal_top1pct_jaccard']:12.3f} "
            f"{r['cross_sample_top1_frequency']:12.3f} "
            f"{r['global_top1_channel']:10d}"
        )

print()
print("Saved:", OUTPUT)
