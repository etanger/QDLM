import json
import numpy as np
import matplotlib.pyplot as plt

def load_jsonl(path):
    with open(path, 'r') as f:
        data = json.loads(f.readlines()[-1])
    return data

def to_array(step_layer_list):
    # [T, L]
    return np.array(step_layer_list)

def plot_heatmap(arr, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(arr.T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel("Timestep")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==== load ====
baseline = load_jsonl("step_logs/baseline1_mmlu_fp16_with_layer.jsonl")
exp12 = load_jsonl("step_logs/exp12_mmlu_2bit_layer_middle_half_fp16_with_layer.jsonl")

# ==== extract ====
b_norm = to_array(baseline["step_layer_hidden_norm"])
e_norm = to_array(exp12["step_layer_hidden_norm"])

# ==== plot ====
plot_heatmap(b_norm, "Baseline FP16 - Hidden Norm", "fig_baseline_norm.png")
plot_heatmap(e_norm, "Exp12 Middle FP16 - Hidden Norm", "fig_exp12_norm.png")

# ==== difference ====
diff = e_norm - b_norm
plot_heatmap(diff, "Exp12 - Baseline (Norm Diff)", "fig_diff_norm.png")

print("Saved figures:")
print("fig_baseline_norm.png")
print("fig_exp12_norm.png")
print("fig_diff_norm.png")

