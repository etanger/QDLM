import json
import numpy as np
import matplotlib.pyplot as plt


def load_jsonl(path):
    with open(path, 'r') as f:
        data = json.loads(f.readlines()[-1])
    return data


def to_array(step_layer_list):
    return np.array(step_layer_list)  # [T, L]


def plot_heatmap(arr, title, save_path, vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(arr.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel("Timestep")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ========= LOAD =========
baseline = load_jsonl("step_logs/baseline1_mmlu_fp16_with_layer.jsonl")
exp12 = load_jsonl("step_logs/exp12_mmlu_2bit_layer_middle_half_fp16_with_layer.jsonl")

# ========= EXTRACT =========
b_var = to_array(baseline["step_layer_hidden_var"])
e_var = to_array(exp12["step_layer_hidden_var"])

b_max = to_array(baseline["step_layer_hidden_max"])
e_max = to_array(exp12["step_layer_hidden_max"])

# ========= VAR =========
plot_heatmap(b_var, "Baseline FP16 - Hidden Variance", "fig_baseline_var.png")
plot_heatmap(e_var, "Exp12 - Hidden Variance", "fig_exp12_var.png")

var_diff = e_var - b_var
plot_heatmap(var_diff, "Exp12 - Baseline (Variance Diff)", "fig_diff_var.png", vmin=-10, vmax=10)

# ========= MAX =========
plot_heatmap(b_max, "Baseline FP16 - Max Activation", "fig_baseline_max.png")
plot_heatmap(e_max, "Exp12 - Max Activation", "fig_exp12_max.png")

max_diff = e_max - b_max
plot_heatmap(max_diff, "Exp12 - Baseline (Max Diff)", "fig_diff_max.png", vmin=-50, vmax=50)

print("Saved:")
print("fig_baseline_var.png")
print("fig_exp12_var.png")
print("fig_diff_var.png")
print("fig_baseline_max.png")
print("fig_exp12_max.png")
print("fig_diff_max.png")
