import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct"
OUT_DIR = "activation_dumps_clean"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

activations = {}

def make_hook(name):
    def hook(module, inputs, output):
        # IMPORTANT: save linear INPUT, not output
        x = inputs[0].detach().float().cpu()
        activations[name] = x
        print(f"[HOOK] {name}: {tuple(x.shape)} max={x.abs().max().item():.4f}")
    return hook

blocks = model.model.transformer.blocks
hooks = []

# match paper-style linear input activations
targets = [
    ("Layer1_Attn_q_proj_INPUT", blocks[1].q_proj),
    ("Layer27_Attn_out_INPUT", blocks[27].attn_out),
    ("Layer31_FFN_ff_proj_INPUT", blocks[31].ff_proj),
    ("Layer31_FFN_up_proj_INPUT", blocks[31].up_proj),
    ("Layer31_FFN_ff_out_INPUT", blocks[31].ff_out),
]

for name, module in targets:
    hooks.append(module.register_forward_hook(make_hook(name)))

prompt = (
    "Solve the following step by step: A train travels 60 miles per hour for 2.5 hours. "
    "Then it travels 40 miles per hour for 1.5 hours. What is the total distance? "
    "Explain all calculations carefully."
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
print("Input tokens:", inputs["input_ids"].shape)

with torch.no_grad():
    _ = model(**inputs)

for name, tensor in activations.items():
    path = os.path.join(OUT_DIR, f"{name}.pt")
    torch.save(tensor, path)
    print("Saved", path, tuple(tensor.shape))

for h in hooks:
    h.remove()

