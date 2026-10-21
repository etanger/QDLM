import os
import torch
import matplotlib.pyplot as plt
import numpy as np

IN_DIR = "activation_dumps_clean"
OUT_DIR = "figs/activation_clean"
os.makedirs(OUT_DIR, exist_ok=True)

names = [
    "Layer1_Attn_q_proj_INPUT",
    "Layer27_Attn_out_INPUT",
    "Layer31_FFN_ff_proj_INPUT",
    "Layer31_FFN_up_proj_INPUT",
    "Layer31_FFN_ff_out_INPUT",
]

for name in names:
    x = torch.load(os.path.join(IN_DIR, f"{name}.pt"))[0].abs().float().numpy()

    # clipped version
    plt.figure(figsize=(12, 5))
    plt.imshow(x, aspect="auto", origin="lower", vmin=0, vmax=20)
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Token")
    plt.title(name + "_clip20")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_clip20.png"), dpi=200)
    plt.close()

    # log version
    plt.figure(figsize=(12, 5))
    plt.imshow(np.log1p(x), aspect="auto", origin="lower")
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Token")
    plt.title(name + "_log1p")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_log1p.png"), dpi=200)
    plt.close()

print("saved clipped/log plots")

