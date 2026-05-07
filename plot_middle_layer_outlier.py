import os
import math
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


PANELS = [
    ("Layer8_FFN_ff_out", 8, "ff_out_INPUT"),
    ("Layer16_FFN_ff_out", 16, "ff_out_INPUT"),
    ("Layer23_FFN_ff_out", 23, "ff_out_INPUT"),
    ("Layer31_FFN_ff_out", 31, "ff_out_INPUT"),
]

def list_samples(root):
    return sorted(
        d for d in os.listdir(root)
        if d.startswith("sample_") and os.path.isdir(os.path.join(root, d))
    )


def load_linear(root, sample):
    path = os.path.join(root, sample, "linear_inputs_selected.pt")
    return torch.load(path, map_location="cpu")


def find_best_case(root):
    """
    在单个 experiment 内部找 Layer31 ff_out massive outlier 最强的 sample/step。
    不和别的 experiment 比。
    """
    best = None

    samples = list_samples(root)
    for si, sample in enumerate(samples):
        if si % 20 == 0:
            print(f"scanning {root}: {si}/{len(samples)} {sample}", flush=True)

        d = load_linear(root, sample)

        for key, x in d.items():
            if "layer_31_ff_out_INPUT" not in key:
                continue

            # key: step_16_layer_31_ff_out_INPUT
            step = int(key.split("_")[1])

            x = x.detach().float().abs()
            max_v = x.max().item()

            if best is None or max_v > best["max"]:
                best = {
                    "sample": sample,
                    "step": step,
                    "key": key,
                    "max": max_v,
                }

        del d

    print("Selected case:", best, flush=True)
    return best["sample"], best["step"]


def get_tensor(root, sample, step, layer, module):
    d = load_linear(root, sample)
    key = f"step_{step}_layer_{layer}_{module}"
    if key not in d:
        raise KeyError(f"{key} not found in {root}/{sample}/linear_inputs_selected.pt")

    x = d[key].detach().float().abs()
    if x.dim() == 3:
        x = x[0]  # [seq, channel]
    return x.numpy(), key


def crop_tokens(A, max_tokens=220):
    """
    5-shot prompt 太长，自动裁剪到最大 activation 附近。
    """
    T, C = A.shape
    t_max, c_max = np.unravel_index(np.argmax(A), A.shape)

    half = max_tokens // 2
    start = max(0, t_max - half)
    end = min(T, start + max_tokens)

    if end - start < max_tokens:
        start = max(0, end - max_tokens)

    return A[start:end], start, end


def downsample_channels(A, max_channels=650):
    T, C = A.shape
    stride = max(1, math.ceil(C / max_channels))
    return A[:, ::stride], stride


def plot_panel(ax, A, title, is_massive=False):
    A_crop, t_start, t_end = crop_tokens(A)
    A_ds, ch_stride = downsample_channels(A_crop)

    T, C = A_ds.shape
    token_axis = np.arange(t_start, t_end)
    channel_axis = np.arange(0, A.shape[1], ch_stride)[:C]

    X, Y = np.meshgrid(channel_axis, token_axis)

    # base surface 截断，不让极端值把整张图压黑
    base_clip = np.percentile(A_ds, 99.5)
    base = np.minimum(A_ds, base_clip)

    ax.plot_surface(
        X, Y, base,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
        alpha=0.90,
        rstride=1,
        cstride=1,
    )

    # 红色 outlier：normal 用 p99.5，massive 用 p99.9
    if is_massive:
        thr = np.percentile(A_ds, 99.9)
        max_bars = 200
    else:
        thr = np.percentile(A_ds, 99.5)
        max_bars = 350

    ys, xs = np.where(A_ds > thr)

    if len(xs) > 0:
        vals = A_ds[ys, xs]

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
            x_real, y_real, z0,
            dx, dy, vals,
            color="red",
            alpha=0.85,
            shade=True,
        )

    max_v = float(A.max())
    mean_v = float(A.mean())
    p9999 = float(np.percentile(A, 99.99))

    ax.set_title(
        f"{title}\nmax={max_v:.1f}, p99.99={p9999:.1f}, mean={mean_v:.2f}",
        fontsize=9,
    )

    ax.set_xlabel("Channel", fontsize=8)
    ax.set_ylabel("Token", fontsize=8)
    ax.set_zlabel("|Act|", fontsize=8)
    ax.view_init(elev=25, azim=-60)
    ax.tick_params(axis="both", labelsize=7)

    if is_massive:
        ax.text2D(
            0.70, 0.88, "Massive",
            transform=ax.transAxes,
            color="red",
            fontsize=10,
            fontweight="bold",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.sample is None or args.step is None:
        sample, step = find_best_case(args.root)
    else:
        sample, step = args.sample, args.step

    fig = plt.figure(figsize=(22, 5.5))

    for i, (title, layer, module) in enumerate(PANELS, start=1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        A, key = get_tensor(args.root, sample, step, layer, module)
        plot_panel(
            ax,
            A,
            f"{title}",
            is_massive=(module == "ff_out_INPUT"),
        )

    fig.suptitle(
        f"{args.name} | Raw Linear Input Activation Outliers | sample={sample}, step={step}",
        fontsize=15,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=250, bbox_inches="tight")
    print("saved:", args.out, flush=True)


if __name__ == "__main__":
    main()
