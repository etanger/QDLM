import os
import torch
import matplotlib.pyplot as plt

IN_DIR = "activation_dumps_clean"
OUT_DIR = "figs/activation_clean"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_one(name, vmax=None):
    path = os.path.join(IN_DIR, f"{name}.pt")
    x = torch.load(path)[0].abs().numpy()  # [token, channel]

    plt.figure(figsize=(12, 5))
    plt.imshow(x, aspect="auto", origin="lower", vmax=vmax)
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Token")
    plt.title(name)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{name}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved", out, "max=", x.max())

names = [
    "Layer1_Attn_q_proj_INPUT",
    "Layer27_Attn_out_INPUT",
    "Layer31_FFN_ff_proj_INPUT",
    "Layer31_FFN_up_proj_INPUT",
    "Layer31_FFN_ff_out_INPUT",
]

for n in names:
    plot_one(n)

# optional clipped versions to see non-massive structure
for n in names:
    plot_one(n + "_clip20", vmax=20) if False else None

