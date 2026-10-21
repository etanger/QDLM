import json
import math
from pathlib import Path

import torch


LAYERS = [8, 12, 16, 20, 24]
BITS_LIST = [2, 3]

TOP_FRACTIONS = [
    0.001,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
]


def simple_ranks(x):
    order = torch.argsort(x)
    ranks = torch.empty(
        len(x),
        dtype=torch.float64,
    )

    ranks[order] = torch.arange(
        len(x),
        dtype=torch.float64,
    )

    return ranks


def spearman(x, y):
    x = x.double()
    y = y.double()

    rx = simple_ranks(x)
    ry = simple_ranks(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    return (
        torch.dot(rx, ry)
        /
        (
            rx.norm()
            * ry.norm()
            + 1e-12
        )
    ).item()


def load_energy_examples(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            rows.append(
                torch.tensor(
                    obj["channel_energy"],
                    dtype=torch.float64,
                )
            )

    return torch.stack(
        rows,
        dim=0,
    )


def random_control(
    errors,
    k,
    repeats=200,
):
    values = []

    g = torch.Generator()
    g.manual_seed(20260924)

    n = len(errors)

    for _ in range(repeats):
        idx = torch.randperm(
            n,
            generator=g,
        )[:k]

        values.append(
            errors[idx]
            .mean()
            .item()
        )

    values = torch.tensor(
        values,
        dtype=torch.float64,
    )

    return (
        values.mean().item(),
        values.std(
            unbiased=True
        ).item(),
    )


for bits in BITS_LIST:

    print()
    print("=" * 80)
    print(f"W{bits}")
    print("=" * 80)

    for layer in LAYERS:

        energy_path = Path(
            "step_logs/channel_relation/"
            f"inference_energy_layer"
            f"{layer}_w{bits}_limit2.jsonl"
        )

        error_path = Path(
            "step_logs/channel_relation/"
            f"column_error_layer"
            f"{layer}_w{bits}.pt"
        )

        h_path = Path(
            "step_logs/gptq_calibration/"
            f"attn_out_layer"
            f"{layer}_w{bits}.pt"
        )

        energy_examples = (
            load_energy_examples(
                energy_path
            )
        )

        mean_energy = (
            energy_examples.mean(
                dim=0
            )
        )

        err_obj = torch.load(
            error_path,
            map_location="cpu",
        )

        rel_error = (
            err_obj[
                "column_rel_error"
            ]
            .double()
        )

        abs_error = (
            err_obj[
                "column_abs_error"
            ]
            .double()
        )

        fp_norm = (
            err_obj[
                "column_fp_norm"
            ]
            .double()
        )

        h_obj = torch.load(
            h_path,
            map_location="cpu",
        )

        h_diag = (
            h_obj[
                "h_diag_normalized"
            ]
            .double()
        )

        rho_energy_rel = spearman(
            mean_energy,
            rel_error,
        )

        rho_energy_abs = spearman(
            mean_energy,
            abs_error,
        )

        rho_energy_h = spearman(
            mean_energy,
            h_diag,
        )

        rho_h_rel = spearman(
            h_diag,
            rel_error,
        )

        per_example_rho = []

        for e in energy_examples:
            per_example_rho.append(
                spearman(
                    e,
                    rel_error,
                )
            )

        rho_tensor = torch.tensor(
            per_example_rho,
            dtype=torch.float64,
        )

        print()
        print(
            f"Layer {layer}"
        )

        print(
            "  examples:",
            len(energy_examples),
        )

        print(
            "  Spearman("
            "mean inference energy, "
            "relative column error):",
            f"{rho_energy_rel:.4f}",
        )

        print(
            "  Per-example rho:"
            f" mean={rho_tensor.mean().item():.4f},"
            f" std={rho_tensor.std().item():.4f}"
        )

        print(
            "  Spearman("
            "mean inference energy, "
            "absolute column error):",
            f"{rho_energy_abs:.4f}",
        )

        print(
            "  Spearman("
            "mean inference energy, "
            "calibration Hessian):",
            f"{rho_energy_h:.4f}",
        )

        print(
            "  Spearman("
            "calibration Hessian, "
            "relative column error):",
            f"{rho_h_rel:.4f}",
        )

        print()
        print(
            "  Top-energy channel groups:"
        )

        sorted_idx = torch.argsort(
            mean_energy,
            descending=True,
        )

        n = len(mean_energy)

        for frac in TOP_FRACTIONS:

            k = max(
                1,
                math.ceil(
                    n * frac
                ),
            )

            top_idx = (
                sorted_idx[:k]
            )

            top_rel = (
                rel_error[top_idx]
                .mean()
                .item()
            )

            top_abs = (
                abs_error[top_idx]
                .mean()
                .item()
            )

            random_mean, random_std = (
                random_control(
                    rel_error,
                    k,
                )
            )

            print(
                f"    top {100*frac:5.2f}% "
                f"(k={k:4d}) | "
                f"rel={top_rel:.5f} | "
                f"random={random_mean:.5f}"
                f" +/- {random_std:.5f} | "
                f"ratio="
                f"{top_rel / random_mean:.4f} | "
                f"abs={top_abs:.5f}"
            )

