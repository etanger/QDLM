import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "analysis/base_vs_exp13_readable"
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN_CSV = "analysis/base_vs_exp13/hidden_drift.csv"
LINEAR_CSV = "analysis/base_vs_exp13/linear_tail_stats.csv"

OUT_PNG = os.path.join(OUT_DIR, "fig2_readable_layer_summary.png")
OUT_TABLE = os.path.join(OUT_DIR, "fig2_layer_summary_table.csv")


def main():
    hidden = pd.read_csv(HIDDEN_CSV)
    linear = pd.read_csv(LINEAR_CSV)

    # hidden drift：按 layer 聚合
    h = (
        hidden.groupby("layer")
        .agg(
            relative_l2_mean=("relative_l2", "mean"),
            relative_l2_std=("relative_l2", "std"),
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
        )
        .reset_index()
    )

    # ff_out tail：不用 ratio，用 absolute delta
    ff = linear[linear["module"] == "ff_out_INPUT"].copy()
    ff["delta_p9999"] = ff["exp13_p9999"] - ff["base_p9999"]
    ff["delta_top001"] = ff["exp13_top001_mean"] - ff["base_top001_mean"]
    ff["delta_max"] = ff["exp13_max"] - ff["base_max"]

    f = (
        ff.groupby("layer")
        .agg(
            delta_p9999_mean=("delta_p9999", "mean"),
            delta_p9999_std=("delta_p9999", "std"),
            delta_top001_mean=("delta_top001", "mean"),
            delta_top001_std=("delta_top001", "std"),
            delta_max_mean=("delta_max", "mean"),
            delta_max_std=("delta_max", "std"),
            base_p9999_mean=("base_p9999", "mean"),
            exp13_p9999_mean=("exp13_p9999", "mean"),
        )
        .reset_index()
    )

    summary = h.merge(f, on="layer", how="inner")
    summary.to_csv(OUT_TABLE, index=False)
    print("Saved table:", OUT_TABLE)

    layers = summary["layer"].astype(str).tolist()
    x = np.arange(len(layers))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 图 1：hidden drift
    axes[0].bar(
        x,
        summary["relative_l2_mean"],
        yerr=summary["relative_l2_std"],
        capsize=4,
    )
    axes[0].set_ylabel("Relative L2")
    axes[0].set_title("Hidden State Drift: Baseline FP16 vs Exp13 Middle Quant")
    axes[0].grid(axis="y", alpha=0.3)

    for i, v in enumerate(summary["relative_l2_mean"]):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # 图 2：ff_out tail delta
    width = 0.35
    axes[1].bar(
        x - width / 2,
        summary["delta_p9999_mean"],
        width,
        yerr=summary["delta_p9999_std"],
        capsize=4,
        label="Δ p99.99",
    )
    axes[1].bar(
        x + width / 2,
        summary["delta_top001_mean"],
        width,
        yerr=summary["delta_top001_std"],
        capsize=4,
        label="Δ top0.1% mean",
    )

    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Exp13 - Baseline")
    axes[1].set_title("ff_out Input Tail Change")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers)
    axes[1].set_xlabel("Layer")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
