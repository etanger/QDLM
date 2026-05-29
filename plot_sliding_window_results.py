import os
import matplotlib.pyplot as plt

os.makedirs("analysis/sliding_window", exist_ok=True)

# Current results
w12 = [
    ("8-19", 0.5471),
    ("10-21", 0.5344),
    ("12-23", 0.5493),
    ("14-25", 0.5614),
    ("16-27", 0.5928),
]

w10 = [
    ("8-17", 0.5772),
    ("10-19", 0.5583),
    ("12-21", 0.5496),
    ("14-23", 0.5721),
]

def centers(labels):
    out = []
    for s in labels:
        a, b = s.split("-")
        out.append((int(a) + int(b)) / 2)
    return out

w12_labels = [x[0] for x in w12]
w12_scores = [x[1] for x in w12]
w12_centers = centers(w12_labels)

w10_labels = [x[0] for x in w10]
w10_scores = [x[1] for x in w10]
w10_centers = centers(w10_labels)

plt.figure(figsize=(8, 5))

plt.plot(w12_centers, w12_scores, marker="o", linewidth=2, label="W12 quant window")
plt.plot(w10_centers, w10_scores, marker="s", linewidth=2, label="W10 quant window")

for x, y, label in zip(w12_centers, w12_scores, w12_labels):
    plt.text(x, y + 0.004, label, ha="center", fontsize=9)

for x, y, label in zip(w10_centers, w10_scores, w10_labels):
    plt.text(x, y - 0.012, label, ha="center", fontsize=9)

plt.axhline(0.519, linestyle="--", linewidth=1, label="Exp13 W16: 8-23")
plt.text(15.5, 0.522, "Exp13 W16 8-23", fontsize=9)

plt.xlabel("Window center layer")
plt.ylabel("MMLU exact match")
plt.title("Sliding Quantization Window Results")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

out = "analysis/sliding_window/sliding_window_accuracy.png"
plt.savefig(out, dpi=250, bbox_inches="tight")
print("saved:", out)
