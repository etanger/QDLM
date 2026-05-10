import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pick_samples(df, num_samples=5, sample_list=None):
    all_samples = sorted(df["sample"].unique().tolist())
    if sample_list:
        chosen = [s for s in sample_list if s in all_samples]
    else:
        chosen = all_samples[:num_samples]
    return chosen


def get_ylim(series, pad_ratio=0.05):
    vmin = float(series.min())
    vmax = float(series.max())
    if vmin == vmax:
        return vmin - 0.1, vmax + 0.1
    pad = (vmax - vmin) * pad_ratio
    return vmin - pad, vmax + pad


def plot_metric_vs_step(df_sample, metric, out_path, title, ylim=None):
    layers = sorted(df_sample["layer"].unique().tolist())
    steps = sorted(df_sample["step"].unique().tolist())

    plt.figure(figsize=(8, 5.5))

    for layer in layers:
        sub = df_sample[df_sample["layer"] == layer].sort_values("step")
        plt.plot(
            sub["step"].values,
            sub[metric].values,
            marker="o",
            label=f"Layer {layer}",
        )

    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel(metric)
    plt.xticks(steps)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}", flush=True)


def plot_metric_vs_layer(df_sample, metric, out_path, title, ylim=None):
    layers = sorted(df_sample["layer"].unique().tolist())
    steps = sorted(df_sample["step"].unique().tolist())

    plt.figure(figsize=(8, 5.5))

    for step in steps:
        sub = df_sample[df_sample["step"] == step].sort_values("layer")
        plt.plot(
            sub["layer"].values,
            sub[metric].values,
            marker="o",
            label=f"Step {step}",
        )

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(metric)
    plt.xticks(layers)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated sample names, e.g. sample_00000,sample_00001")
    parser.add_argument("--tag", type=str, default="base_vs_exp13")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    required_cols = {"sample", "step", "layer", "cosine", "relative_l2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    sample_list = None
    if args.samples:
        sample_list = [x.strip() for x in args.samples.split(",") if x.strip()]

    chosen_samples = pick_samples(df, num_samples=args.num_samples, sample_list=sample_list)

    print("chosen samples:", chosen_samples, flush=True)

    # 为了让不同 sample 之间更好比较，y 轴统一
    cosine_ylim = get_ylim(df["cosine"])
    l2_ylim = get_ylim(df["relative_l2"])

    for sample in chosen_samples:
        df_sample = df[df["sample"] == sample].copy()

        sample_dir = os.path.join(args.outdir, sample)
        os.makedirs(sample_dir, exist_ok=True)

        # 1) cosine vs timestep
        plot_metric_vs_step(
            df_sample,
            "cosine",
            os.path.join(sample_dir, f"{sample}_cosine_vs_timestep.png"),
            f"{args.tag} | {sample} | Cosine vs Timestep",
            ylim=cosine_ylim,
        )

        # 2) relative_l2 vs timestep
        plot_metric_vs_step(
            df_sample,
            "relative_l2",
            os.path.join(sample_dir, f"{sample}_relative_l2_vs_timestep.png"),
            f"{args.tag} | {sample} | Relative L2 vs Timestep",
            ylim=l2_ylim,
        )

        # 3) cosine vs layer
        plot_metric_vs_layer(
            df_sample,
            "cosine",
            os.path.join(sample_dir, f"{sample}_cosine_vs_layer.png"),
            f"{args.tag} | {sample} | Cosine vs Layer",
            ylim=cosine_ylim,
        )

        # 4) relative_l2 vs layer
        plot_metric_vs_layer(
            df_sample,
            "relative_l2",
            os.path.join(sample_dir, f"{sample}_relative_l2_vs_layer.png"),
            f"{args.tag} | {sample} | Relative L2 vs Layer",
            ylim=l2_ylim,
        )

    print("done.", flush=True)


if __name__ == "__main__":
    main()

