import torch
import matplotlib.pyplot as plt

def plot(path, out):
    x = torch.load(path)[0].abs().numpy()  # [token, channel]
    plt.figure(figsize=(10, 4))
    plt.imshow(x, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Token")
    plt.title(out.replace(".png", ""))
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

plot("layer1_q_proj.pt", "Layer1_Attn_q_proj.png")
plot("layer27_attn_out.pt", "Layer27_Attn_out.png")
plot("layer31_ff_out.pt", "Layer31_FFN_ff_out.png")
