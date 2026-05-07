import os
import math
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def percentile_by_kthvalue(x, q):
    x = x.contiguous()
    n = x.numel()
    k = int(math.ceil(q * n))
    k = max(1, min(k, n))
    return torch.kthvalue(x, k).values.item()


def list_samples(root):
    return sorted(
        d for d in os.listdir(root)
        if d.startswith("sample_") and os.path.isdir(os.path.join(root, d))
    )


def parse_key(key):
    # step_16_layer_31_ff_out_INPUT
    parts = key.split("_")
    step = int(parts[1])
    layer = int(parts[3])
    module = "_".join(parts[4:])
    return step, layer, module


def tail_stats(x):
    x = x.detach().float().abs().flatten().contiguous()

    mean_v = x.mean().item()
    max_v = x.max().item()
    p99 = percentile_by_kthvalue(x, 0.99)
    p999 = percentile_by_kthvalue(x, 0.999)
    p9999 = percentile_by_kthvalue(x, 0.9999)

    k = max(1, int(x.numel() * 0.001))
    top001_mean = torch.topk(x, k).values.mean().item()

    return {
        "mean": mean_v,
        "max": max_v,
        "p99": p99,
        "p999": p999,
        "p9999": p9999,
        "top001_mean": top001_mean,
        "massive_score_1": max_v / (p9999 + 1e-8),   # spike vs tail
        "massive_score_2": max_v / (mean_v + 1e-8),  # spike vs average
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    samples = list_samples(args.root)

    for i, sample in enumerate(samples):
        if i % 20 == 0:
            print(f"[{i}/{len(samples)}] {sample}", flush=True)

        path = os.path.join(args.root, sample, "linear_inputs_selected.pt")
        d = torch.load(path, map_location="cpu")

        for key, x in d.items():
            step, layer, module = parse_key(key)

            if module != "ff_out_INPUT":
                continue

            stats = tail_stats(x)

            row = {
                "sample": sample,
                "step": step,
                "layer": layer,
                "module": module,
            }
            row.update(stats)
            rows.append(row)

        del d

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "ff_out_stats.csv")
    df.to_csv(csv_path, index=False)
    print("saved:", csv_path, flush=True)

    # per-layer summary
    layer_df = (
        df.groupby("layer")
        .agg(
            mean_p9999=("p9999", "mean"),
            std_p9999=("p9999", "std"),
            mean_top001=("top001_mean", "mean"),
            std_top001=("top001_mean", "std"),
            mean_max=("max", "mean"),
            std_max=("max", "std"),
            mean_massive1=("massive_score_1", "mean"),
            std_massive1=("massive_score_1", "std"),
            mean_massive2=("massive_score_2", "mean"),
            std_massive2=("massive_score_2", "std"),
        )
        .reset_index()
    )
    layer_csv = os.path.join(args.outdir, "ff_out_layer_summary.csv")
    layer_df.to_csv(layer_csv, index=False)
    print("saved:", layer_csv, flush=True)

    # step x layer summary
    heat_df = (
        df.groupby(["step", "layer"])
        .agg(
            p9999_mean=("p9999", "mean"),
            top001_mean=("top001_mean", "mean"),
            max_mean=("max", "mean"),
            massive1_mean=("massive_score_1", "mean"),
        )
        .reset_index()
    )
    heat_csv = os.path.join(args.outdir, "ff_out_step_layer_summary.csv")
    heat_df.to_csv(heat_csv, index=False)
    print("saved:", heat_csv, flush=True)

    # plotting
    layers = layer_df["layer"].astype(str).tolist()
    x = np.arange(len(layers))

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # subplot 1: tail scale
    width = 0.25
    axes[0].bar(
        x - width,
        layer_df["mean_p9999"],
        width,
        yerr=layer_df["std_p9999"],
        capsize=4,
        label="p99.99",
    )
    axes[0].bar(
        x,
        layer_df["mean_top001"],
        width,
        yerr=layer_df["std_top001"],
        capsize=4,
        label="top0.1% mean",
    )
    axes[0].bar(
        x + width,
        layer_df["mean_max"],
        width,
        yerr=layer_df["std_max"],
        capsize=4,
        label="max",
    )
    axes[0].set_title(f"{args.name}: ff_out tail magnitude by layer")
    axes[0].set_ylabel("|activation|")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # subplot 2: massive-ness
    width2 = 0.35
    axes[1].bar(
        x - width2 / 2,
        layer_df["mean_massive1"],
        width2,
        yerr=layer_df["std_massive1"],
        capsize=4,
        label="max / p99.99",
    )
    axes[1].bar(
        x + width2 / 2,
        layer_df["mean_massive2"],
        width2,
        yerr=layer_df["std_massive2"],
        capsize=4,
        label="max / mean",
    )
    axes[1].set_title(f"{args.name}: ff_out massive-outlier score by layer")
    axes[1].set_ylabel("score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers)
    axes[1].set_xlabel("Layer")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.outdir, "single_exp_summary.png")
    plt.savefig(fig_path, dpi=250, bbox_inches="tight")
    print("saved:", fig_path, flush=True)


if __name__ == "__main__":
    main()
