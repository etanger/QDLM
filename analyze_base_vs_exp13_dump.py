from tqdm import tqdm
import os
import re
import csv
import gc
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = "activation_dump/baseline_fp16_limit5"
EXP_DIR = "activation_dump/exp13_middle_quant_limit5"
OUT_DIR = "analysis/base_vs_exp13"
os.makedirs(OUT_DIR, exist_ok=True)

KEY_RE = re.compile(r"step_(\d+)_layer_(\d+)_(.+)")


def parse_key(key):
    m = KEY_RE.match(key)
    if not m:
        return None
    step = int(m.group(1))
    layer = int(m.group(2))
    name = m.group(3)
    return step, layer, name


def load_pt(path):
    return torch.load(path, map_location="cpu")


def cosine_and_l2(a, b):
    a = a.float().flatten()
    b = b.float().flatten()

    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel_l2 = (torch.norm(b - a) / (torch.norm(a) + 1e-8)).item()
    max_abs_diff = torch.max(torch.abs(b - a)).item()
    mean_abs_diff = torch.mean(torch.abs(b - a)).item()

    return cos, rel_l2, max_abs_diff, mean_abs_diff

def percentile_by_kthvalue(x, q):
    """
    Exact percentile using kthvalue instead of torch.quantile.
    q: 0.99 / 0.999 / 0.9999
    """
    x = x.contiguous()
    n = x.numel()
    k = int(math.ceil(q * n))
    k = max(1, min(k, n))
    return torch.kthvalue(x, k).values.item()


def tail_stats(x):
    x = x.abs().float().flatten().contiguous()

    max_v = x.max().item()
    mean_v = x.mean().item()

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
        "max_over_p9999": max_v / (p9999 + 1e-8),
        "max_over_mean": max_v / (mean_v + 1e-8),
    }


def list_samples(root):
    return sorted(
        d for d in os.listdir(root)
        if d.startswith("sample_") and os.path.isdir(os.path.join(root, d))
    )


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(rows, value_key, title, out_png):
    # aggregate mean by step/layer
    bucket = defaultdict(list)
    for r in rows:
        bucket[(int(r["step"]), int(r["layer"]))].append(float(r[value_key]))

    steps = sorted(set(k[0] for k in bucket.keys()))
    layers = sorted(set(k[1] for k in bucket.keys()))

    arr = np.full((len(steps), len(layers)), np.nan)
    for i, s in enumerate(steps):
        for j, l in enumerate(layers):
            vals = bucket.get((s, l), [])
            if vals:
                arr[i, j] = np.mean(vals)

    plt.figure(figsize=(9, 5))
    plt.imshow(arr.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xticks(range(len(steps)), steps, rotation=45)
    plt.yticks(range(len(layers)), layers)
    plt.xlabel("Timestep")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    base_samples = list_samples(BASE_DIR)
    exp_samples = list_samples(EXP_DIR)

    common = sorted(set(base_samples) & set(exp_samples))
    print("baseline samples:", len(base_samples))
    print("exp13 samples:", len(exp_samples))
    print("common samples:", len(common))

    hidden_rows = []
    linear_rows = []

    for idx, sample in enumerate(tqdm(common, desc="Processing samples")):
        print(f"\n[{idx}/{len(common)}] {sample}", flush=True)

        base_sample_dir = os.path.join(BASE_DIR, sample)
        exp_sample_dir = os.path.join(EXP_DIR, sample)

        # ============================
        # hidden drift
        # ============================
        base_hidden_path = os.path.join(base_sample_dir, "full_hidden_selected.pt")
        exp_hidden_path = os.path.join(exp_sample_dir, "full_hidden_selected.pt")

        if os.path.exists(base_hidden_path) and os.path.exists(exp_hidden_path):
            print(f"  loading hidden: {sample}", flush=True)
            bh = load_pt(base_hidden_path)
            eh = load_pt(exp_hidden_path)
            print(f"  hidden loaded: {len(bh)} base keys, {len(eh)} exp keys", flush=True)

            common_keys = sorted(set(bh.keys()) & set(eh.keys()))
            for key in tqdm(common_keys, desc=f"  hidden keys {sample}", leave=False):
                parsed = parse_key(key)
                if parsed is None:
                    continue
                step, layer, name = parsed

                cos, rel_l2, max_abs_diff, mean_abs_diff = cosine_and_l2(bh[key], eh[key])

                hidden_rows.append({
                    "sample": sample,
                    "step": step,
                    "layer": layer,
                    "name": name,
                    "cosine": cos,
                    "relative_l2": rel_l2,
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                })

            del bh, eh
            gc.collect()

        # ============================
        # linear input outlier stats
        # ============================
        base_lin_path = os.path.join(base_sample_dir, "linear_inputs_selected.pt")
        exp_lin_path = os.path.join(exp_sample_dir, "linear_inputs_selected.pt")

        if os.path.exists(base_lin_path) and os.path.exists(exp_lin_path):
            print(f"  loading linear inputs: {sample}", flush=True)
            bl = load_pt(base_lin_path)
            el = load_pt(exp_lin_path)
            print(f"  linear loaded: {len(bl)} base keys, {len(el)} exp keys", flush=True)

            common_keys = sorted(set(bl.keys()) & set(el.keys()))
            for key in tqdm(common_keys, desc=f"  linear keys {sample}", leave=False):
                parsed = parse_key(key)
                if parsed is None:
                    continue
                step, layer, module = parsed

                bs = tail_stats(bl[key])
                es = tail_stats(el[key])

                row = {
                    "sample": sample,
                    "step": step,
                    "layer": layer,
                    "module": module,
                }

                for k, v in bs.items():
                    row[f"base_{k}"] = v
                for k, v in es.items():
                    row[f"exp13_{k}"] = v

                row["delta_max"] = es["max"] - bs["max"]
                row["ratio_max"] = es["max"] / (bs["max"] + 1e-8)
                row["delta_p9999"] = es["p9999"] - bs["p9999"]
                row["ratio_p9999"] = es["p9999"] / (bs["p9999"] + 1e-8)
                row["delta_top001_mean"] = es["top001_mean"] - bs["top001_mean"]
                row["ratio_top001_mean"] = es["top001_mean"] / (bs["top001_mean"] + 1e-8)

                linear_rows.append(row)

            del bl, el
            gc.collect()

    # ============================
    # save CSV
    # ============================
    hidden_csv = os.path.join(OUT_DIR, "hidden_drift.csv")
    linear_csv = os.path.join(OUT_DIR, "linear_tail_stats.csv")

    write_csv(
        hidden_csv,
        hidden_rows,
        [
            "sample", "step", "layer", "name",
            "cosine", "relative_l2", "max_abs_diff", "mean_abs_diff"
        ],
    )

    linear_fields = [
        "sample", "step", "layer", "module",
        "base_mean", "base_max", "base_p99", "base_p999", "base_p9999",
        "base_top001_mean", "base_max_over_p9999", "base_max_over_mean",
        "exp13_mean", "exp13_max", "exp13_p99", "exp13_p999", "exp13_p9999",
        "exp13_top001_mean", "exp13_max_over_p9999", "exp13_max_over_mean",
        "delta_max", "ratio_max",
        "delta_p9999", "ratio_p9999",
        "delta_top001_mean", "ratio_top001_mean",
    ]

    write_csv(linear_csv, linear_rows, linear_fields)

    print("saved:", hidden_csv)
    print("saved:", linear_csv)

    # ============================
    # plots
    # ============================
    plot_heatmap(
        hidden_rows,
        "cosine",
        "Baseline vs Exp13 Hidden Cosine",
        os.path.join(OUT_DIR, "hidden_cosine_heatmap.png"),
    )

    plot_heatmap(
        hidden_rows,
        "relative_l2",
        "Baseline vs Exp13 Hidden Relative L2",
        os.path.join(OUT_DIR, "hidden_relative_l2_heatmap.png"),
    )

    # only ff_out input, usually most important
    ff_rows = [r for r in linear_rows if r["module"] == "ff_out_INPUT"]

    plot_heatmap(
        ff_rows,
        "ratio_p9999",
        "Exp13 / Baseline p9999 Ratio (ff_out input)",
        os.path.join(OUT_DIR, "ff_out_p9999_ratio_heatmap.png"),
    )

    plot_heatmap(
        ff_rows,
        "ratio_top001_mean",
        "Exp13 / Baseline Top0.1% Mean Ratio (ff_out input)",
        os.path.join(OUT_DIR, "ff_out_top001_ratio_heatmap.png"),
    )

    plot_heatmap(
        ff_rows,
        "ratio_max",
        "Exp13 / Baseline Max Ratio (ff_out input)",
        os.path.join(OUT_DIR, "ff_out_max_ratio_heatmap.png"),
    )

    print("plots saved under:", OUT_DIR)


if __name__ == "__main__":
    main()
