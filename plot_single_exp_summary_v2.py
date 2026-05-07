import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_or_fail(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def median_iqr(df, value):
    g = (
        df.groupby("layer")[value]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    g["err_low"] = g["median"] - g["q25"]
    g["err_high"] = g["q75"] - g["median"]
    return g


def plot_bar(ax, df, value, title, ylabel, layers=None, log=False):
    if layers is not None:
        df = df[df["layer"].isin(layers)].copy()

    g = median_iqr(df, value)
    x = np.arange(len(g))
    labels = g["layer"].astype(str).tolist()

    yerr = np.vstack([g["err_low"].values, g["err_high"].values])

    ax.bar(x, g["median"].values, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    if log:
        ax.set_yscale("log")

    for i, v in enumerate(g["median"].values):
        ax.text(i, v, f"{v:.2g}", ha="center", va="bottom", fontsize=8)


def plot_step_layer_heatmap(ax, df, value, title, layers=None):
    if layers is not None:
        df = df[df["layer"].isin(layers)].copy()

    h = (
        df.groupby(["step", "layer"])[value]
        .median()
        .reset_index()
    )

    steps = sorted(h["step"].unique())
    layers_sorted = sorted(h["layer"].unique())

    arr = np.full((len(layers_sorted), len(steps)), np.nan)

    for i, layer in enumerate(layers_sorted):
        for j, step in enumerate(steps):
            sub = h[(h["layer"] == layer) & (h["step"] == step)]
            if len(sub):
                arr[i, j] = sub[value].iloc[0]

    im = ax.imshow(arr, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_yticks(np.arange(len(layers_sorted)))
    ax.set_yticklabels(layers_sorted)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = load_or_fail(args.csv)

    # 只看 ff_out
    df = df[df["module"] == "ff_out_INPUT"].copy()

    # 更靠谱的 massive score：最大值 / p99.99
    df["massive_score"] = df["max"] / (df["p9999"] + 1e-8)

    middle_layers = [8, 16, 23]
    selected_layers = [1, 8, 16, 23, 27, 31]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    plot_bar(
        axes[0, 0],
        df,
        "p9999",
        "Middle layers: p99.99 tail",
        "|activation|",
        layers=middle_layers,
        log=False,
    )

    plot_bar(
        axes[0, 1],
        df,
        "max",
        "Middle layers: max activation",
        "|activation|",
        layers=middle_layers,
        log=False,
    )

    plot_bar(
        axes[0, 2],
        df,
        "massive_score",
        "Middle layers: max / p99.99",
        "score",
        layers=middle_layers,
        log=False,
    )

    plot_bar(
        axes[1, 0],
        df,
        "p9999",
        "All selected layers: p99.99 tail, log-scale",
        "|activation|",
        layers=selected_layers,
        log=True,
    )

    plot_step_layer_heatmap(
        axes[1, 1],
        df,
        "p9999",
        "p99.99 across timestep/layer",
        layers=selected_layers,
    )

    plot_step_layer_heatmap(
        axes[1, 2],
        df,
        "massive_score",
        "max / p99.99 across timestep/layer",
        layers=selected_layers,
    )

    fig.suptitle(f"{args.name}: single-experiment ff_out outlier summary", fontsize=15)
    plt.tight_layout()
    plt.savefig(args.out, dpi=250, bbox_inches="tight")
    print("saved:", args.out, flush=True)


if __name__ == "__main__":
    main()
