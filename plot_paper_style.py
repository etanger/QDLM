import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# 1. 你在这里改文件、标题、阈值
# ============================================================
PLOTS = [
    {
        "path": "activation_dumps_clean/Layer1_Attn_q_proj_INPUT.pt",
        "title": "(a1) LLaDA_8B_Layer1_Attn_q_proj",
        "outlier_thr": 2.5,      # 超过这个值画成红色 outlier
        "base_clip": 2.5,        # 蓝色底图 clip 到这里
        "zlim": 15,
        "channel_stride": 4,     # 通道太多时下采样，不然 3D 太慢
        "token_stride": 1,
        "annotate_max": False,
    },
    {
        "path": "activation_dumps_clean/Layer27_Attn_out_INPUT.pt",
        "title": "(b1) LLaDA_8B_Layer27_Attn_out_proj",
        "outlier_thr": 3.5,
        "base_clip": 3.5,
        "zlim": 10,
        "channel_stride": 4,
        "token_stride": 1,
        "annotate_max": False,
    },
    {
        "path": "activation_dumps_clean/Layer31_FFN_ff_proj_INPUT.pt",
        "title": "(c1) LLaDA_8B_Layer31_FFN_ff_proj",
        "outlier_thr": 8.0,
        "base_clip": 8.0,
        "zlim": 60,
        "channel_stride": 4,
        "token_stride": 1,
        "annotate_max": False,
    },
    {
        "path": "activation_dumps_clean/Layer31_FFN_ff_out_INPUT.pt",
        "title": "(d1) LLaDA_8B_Layer31_FFN_ff_out",
        "outlier_thr": 20.0,
        "base_clip": 20.0,
        "zlim": 120,             # massive outlier 时别设太小
        "channel_stride": 8,     # 12288 channels，建议 stride 大一点
        "token_stride": 1,
        "annotate_max": True,
    },
]


OUTPUT_PNG = "paper_style_activation.png"


# ============================================================
# 2. 读 tensor
# ============================================================
def load_activation(path):
    x = torch.load(path, map_location="cpu")

    if isinstance(x, dict):
        # 如果你保存成 dict，就尽量自动取一个 tensor
        for k in ["activation", "tensor", "data", "x"]:
            if k in x:
                x = x[k]
                break

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    x = x.detach().float().cpu()

    # 常见形状:
    # [1, T, C] -> [T, C]
    # [T, C]    -> [T, C]
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    elif x.dim() == 3:
        # 如果还有 batch，默认取第一个
        x = x[0]

    if x.dim() != 2:
        raise ValueError(f"{path} shape not supported: {tuple(x.shape)}")

    # 论文这类图一般看 magnitude
    x = x.abs()

    return x.numpy()  # [T, C]


# ============================================================
# 3. 单张图：蓝色 base + 红色 outlier
# ============================================================
def plot_one(ax, act, title,
             outlier_thr=None,
             base_clip=None,
             zlim=None,
             channel_stride=1,
             token_stride=1,
             annotate_max=False,
             max_red_bars=1500):
    """
    act: [T, C]
    """
    T0, C0 = act.shape

    # 下采样，加快绘图
    tok_idx = np.arange(0, T0, token_stride)
    ch_idx  = np.arange(0, C0, channel_stride)

    A = act[np.ix_(tok_idx, ch_idx)]   # [T, C]
    T, C = A.shape

    if outlier_thr is None:
        outlier_thr = np.quantile(A, 0.999)
    if base_clip is None:
        base_clip = outlier_thr

    # 蓝色底图：把大值截断到 base_clip
    base = np.minimum(A, base_clip)

    X, Y = np.meshgrid(ch_idx, tok_idx)

    # 画 base surface
    ax.plot_surface(
        X, Y, base,
        cmap="coolwarm_r",   # 你也可以改成 "Blues"
        linewidth=0,
        antialiased=False,
        alpha=0.95,
        rstride=1,
        cstride=1,
    )

    # 找 outlier
    mask = A > outlier_thr
    ys, xs = np.where(mask)

    if len(xs) > 0:
        vals = A[ys, xs]

        # 如果 outlier 太多，只画 top max_red_bars 个
        if len(vals) > max_red_bars:
            keep = np.argsort(vals)[-max_red_bars:]
            xs = xs[keep]
            ys = ys[keep]
            vals = vals[keep]

        x_real = ch_idx[xs]
        y_real = tok_idx[ys]
        z0 = np.full_like(vals, fill_value=outlier_thr, dtype=np.float32)
        dz = vals - outlier_thr

        dx = np.full_like(vals, fill_value=max(channel_stride * 0.8, 0.8), dtype=np.float32)
        dy = np.full_like(vals, fill_value=max(token_stride * 0.8, 0.8), dtype=np.float32)

        ax.bar3d(
            x_real, y_real, z0,
            dx, dy, dz,
            color="red",
            alpha=0.90,
            shade=True,
            zsort="average"
        )

    # 标最大值
    if annotate_max:
        iy, ix = np.unravel_index(np.argmax(A), A.shape)
        x_m = ch_idx[ix]
        y_m = tok_idx[iy]
        z_m = A[iy, ix]
        ax.text(
            x_m, y_m, z_m * 1.05,
            "Massive",
            color="red",
            fontsize=10,
            weight="bold"
        )

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Channel", labelpad=6)
    ax.set_ylabel("Token", labelpad=6)
    ax.set_zlabel("", labelpad=0)

    ax.view_init(elev=28, azim=-60)

    if zlim is not None:
        ax.set_zlim(0, zlim)

    # 白底更像论文图
    ax.xaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 1))

    ax.grid(True)


# ============================================================
# 4. 主函数：自动排版
# ============================================================
def main():
    n = len(PLOTS)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(5.8 * ncols, 4.8 * nrows), constrained_layout=True)

    for i, cfg in enumerate(PLOTS, start=1):
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            continue

        act = load_activation(path)
        ax = fig.add_subplot(nrows, ncols, i, projection="3d")

        plot_one(
            ax,
            act,
            title=cfg.get("title", os.path.basename(path)),
            outlier_thr=cfg.get("outlier_thr"),
            base_clip=cfg.get("base_clip"),
            zlim=cfg.get("zlim"),
            channel_stride=cfg.get("channel_stride", 1),
            token_stride=cfg.get("token_stride", 1),
            annotate_max=cfg.get("annotate_max", False),
        )

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[DONE] saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
