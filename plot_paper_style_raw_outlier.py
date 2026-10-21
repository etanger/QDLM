import os
import math
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


BASE_DIR = "activation_dump/baseline_fp16_limit5"
EXP_DIR = "activation_dump/exp13_middle_quant_limit5"
CSV_PATH = "analysis/base_vs_exp13/linear_tail_stats.csv"
OUT_DIR = "analysis/base_vs_exp13_readable"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "fig1_paper_style_raw_outliers.png")


# 论文式 4 个 panel：normal outlier -> massive outlier
PANELS = [
    ("layer1_q_proj", 1, "q_proj_INPUT", "Layer1_Attn_q_proj"),
    ("layer27_attn_out", 27, "attn_out_INPUT", "Layer27_Attn_out"),
    ("layer31_ff_proj", 31, "ff_proj_INPUT", "Layer31_FFN_ff_proj"),
    ("layer31_ff_out", 31, "ff_out_INPUT", "Layer31_FFN_ff_out"),
]


def choose_case():
    """
    自动选一个 massive outlier 最明显的 case：
    从 exp13 的 layer31 ff_out_INPUT 里选 exp13_max 最大的 sample/step。
    """
    df = pd.read_csv(CSV_PATH)

    sub = df[
        (df["layer"] == 31)
        & (df["module"] == "ff_out_INPUT")
    ].copy()

    row = sub.sort_values("exp13_max", ascending=False).iloc[0]
    sample = row["sample"]
    step = int(row["step"])

    print("Selected case:")
    print("  sample:", sample)
    print("  step:", step)
    print("  exp13_max:", row["exp13_max"])
    print("  exp13_p9999:", row["exp13_p9999"])
    print("  exp13_top001_mean:", row["exp13_top001_mean"])

    return sample, step


def load_tensor(root, sample, step, layer, module):
    path = os.path.join(root, sample, "linear_inputs_selected.pt")
    d = torch.load(path, map_location="cpu")

    key = f"step_{step}_layer_{layer}_{module}"
    if key not in d:
        raise KeyError(f"Cannot find key {key} in {path}")

    x = d[key].detach().float().abs()

    # [1, seq, channel] -> [seq, channel]
    if x.dim() == 3:
        x = x[0]

    return x.cpu().numpy(), key


def crop_around_max(A, max_tokens=220):
    """
    MMLU 5-shot prompt 很长，直接画 900+ tokens 会很难看。
    这里自动裁剪到最大 activation 附近的一段 token window。
    """
    T, C = A.shape
    max_pos = np.unravel_index(np.argmax(A), A.shape)
    t_max = max_pos[0]

    half = max_tokens // 2
    start = max(0, t_max - half)
    end = min(T, start + max_tokens)

    if end - start < max_tokens:
        start = max(0, end - max_tokens)

    return A[start:end, :], start, end, t_max


def downsample_channels(A, max_channels=600):
    """
    channel 太多 3D 会爆炸，所以只在 channel 维下采样。
    """
    T, C = A.shape
    stride = max(1, math.ceil(C / max_channels))
    return A[:, ::stride], stride


def plot_one(ax, A, title, massive=False):
    """
    论文风格：
    - 蓝色/浅色 surface 表示主体 activation landscape
    - 红色 bar 表示 outlier tail
    """
    A_crop, t_start, t_end, t_max = crop_around_max(A)
    A_ds, ch_stride = downsample_channels(A_crop)

    T, C = A_ds.shape

    token_axis = np.arange(t_start, t_end)
    channel_axis = np.arange(0, A.shape[1], ch_stride)[:C]

    X, Y = np.meshgrid(channel_axis, token_axis)

    # base surface 不直接用 max，否则 massive outlier 会把图压扁
    base_clip = np.percentile(A_ds, 99.5)
    base = np.minimum(A_ds, base_clip)

    ax.plot_surface(
        X,
        Y,
        base,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
        alpha=0.92,
        rstride=1,
        cstride=1,
    )

    # outlier 阈值：normal 用 p99.5，massive 用 p99.9
    if massive:
        thr = np.percentile(A_ds, 99.9)
        max_bars = 250
    else:
        thr = np.percentile(A_ds, 99.5)
        max_bars = 400

    mask = A_ds > thr
    ys, xs = np.where(mask)

    if len(xs) > 0:
        vals = A_ds[ys, xs]

        # 只画 top bars，避免红柱太密
        if len(vals) > max_bars:
            keep = np.argsort(vals)[-max_bars:]
            xs = xs[keep]
            ys = ys[keep]
            vals = vals[keep]

        x_real = channel_axis[xs]
        y_real = token_axis[ys]

        z0 = np.zeros_like(vals)
        dx = np.full_like(vals, ch_stride * 0.75, dtype=np.float32)
        dy = np.full_like(vals, 0.75, dtype=np.float32)

        ax.bar3d(
            x_real,
            y_real,
            z0,
            dx,
            dy,
            vals,
            color="red",
            alpha=0.85,
            shade=True,
        )

    max_v = float(A.max())
    p9999 = float(np.percentile(A, 99.99))
    mean_v = float(A.mean())

    ax.set_title(
        f"{title}\nmax={max_v:.1f}, p99.99={p9999:.1f}, mean={mean_v:.2f}",
        fontsize=9,
    )

    ax.set_xlabel("Channel", fontsize=8)
    ax.set_ylabel("Token", fontsize=8)
    ax.set_zlabel("|Act|", fontsize=8)

    ax.view_init(elev=25, azim=-60)
    ax.tick_params(axis="both", labelsize=7)

    # massive panel 要保留大 spike
    if massive:
        ax.text2D(
            0.72,
            0.88,
            "Massive",
            transform=ax.transAxes,
            color="red",
            fontsize=10,
            fontweight="bold",
        )


def main():
    sample, step = choose_case()

    fig = plt.figure(figsize=(22, 9))

    for row_idx, (root, row_name) in enumerate(
        [
            (BASE_DIR, "Baseline FP16"),
            (EXP_DIR, "Exp13 Middle Quant"),
        ]
    ):
        for col_idx, (_, layer, module, title) in enumerate(PANELS):
            ax_idx = row_idx * len(PANELS) + col_idx + 1
            ax = fig.add_subplot(2, len(PANELS), ax_idx, projection="3d")

            A, key = load_tensor(root, sample, step, layer, module)

            full_title = f"{row_name}\n{title}"
            massive = module == "ff_out_INPUT"

            plot_one(ax, A, full_title, massive=massive)

    fig.suptitle(
        f"Raw Linear Input Activation Outliers | sample={sample}, step={step}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
