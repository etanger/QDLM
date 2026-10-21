import torch
import os

files = [
    "activation_dumps_clean/Layer1_Attn_q_proj_INPUT.pt",
    "activation_dumps_clean/Layer27_Attn_out_INPUT.pt",
    "activation_dumps_clean/Layer31_FFN_ff_proj_INPUT.pt",
    "activation_dumps_clean/Layer31_FFN_up_proj_INPUT.pt",
    "activation_dumps_clean/Layer31_FFN_ff_out_INPUT.pt",
]

for path in files:
    x = torch.load(path)[0].abs().float()  # [token, channel]
    flat = x.flatten()

    print("=" * 80)
    print(path)
    print("shape:", tuple(x.shape))
    print("max:", flat.max().item())
    print("mean:", flat.mean().item())
    print("p99:", torch.quantile(flat, 0.99).item())
    print("p999:", torch.quantile(flat, 0.999).item())
    print("p9999:", torch.quantile(flat, 0.9999).item())

    vals, idxs = torch.topk(flat, 10)
    print("top10:")
    for v, idx in zip(vals, idxs):
        token = idx.item() // x.shape[1]
        channel = idx.item() % x.shape[1]
        print(f"  value={v.item():.4f}, token={token}, channel={channel}")

