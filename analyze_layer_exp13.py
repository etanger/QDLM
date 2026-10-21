import json
import numpy as np
import matplotlib.pyplot as plt


def load_jsonl(path):
    with open(path, "r") as f:
        return json.loads(f.readlines()[-1])


def to_array(x):
    return np.array(x)


def plot_heatmap(arr, title, save_path, vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(arr.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel("Timestep")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


baseline = load_jsonl("step_logs/baseline1_mmlu_fp16_with_layer.jsonl")
exp13 = load_jsonl("step_logs/exp13_mmlu_2bit_layer_only_middle_quant_with_layer.jsonl")

# norm
b_norm = to_array(baseline["step_layer_hidden_norm"])
e_norm = to_array(exp13["step_layer_hidden_norm"])
plot_heatmap(e_norm, "Exp13 Middle Quant - Hidden Norm", "fig_exp13_norm.png")
plot_heatmap(e_norm - b_norm, "Exp13 - Baseline (Norm Diff)", "fig_exp13_diff_norm.png")

# variance
b_var = to_array(baseline["step_layer_hidden_var"])
e_var = to_array(exp13["step_layer_hidden_var"])
plot_heatmap(e_var, "Exp13 Middle Quant - Hidden Variance", "fig_exp13_var.png")
plot_heatmap(e_var - b_var, "Exp13 - Baseline (Variance Diff)", "fig_exp13_diff_var.png", vmin=-10, vmax=10)

# max
b_max = to_array(baseline["step_layer_hidden_max"])
e_max = to_array(exp13["step_layer_hidden_max"])
plot_heatmap(e_max, "Exp13 Middle Quant - Max Activation", "fig_exp13_max.png")
plot_heatmap(e_max - b_max, "Exp13 - Baseline (Max Diff)", "fig_exp13_diff_max.png", vmin=-50, vmax=50)

print("Saved:")
print("fig_exp13_norm.png")
print("fig_exp13_diff_norm.png")
print("fig_exp13_var.png")
print("fig_exp13_diff_var.png")
print("fig_exp13_max.png")
print("fig_exp13_diff_max.png")
