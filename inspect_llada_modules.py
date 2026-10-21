from transformers import AutoModelForCausalLM
import torch

model_path = '/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
print(f"Loading model from {model_path}...")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='cpu',
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("\n===== TOP-LEVEL MODULES =====")
for name, module in model.named_children():
    print(name, "->", type(module))

print("\n===== MODULES WITH layer/block =====")
for name, module in model.named_modules():
    lname = name.lower()
    if "layer" in lname or "block" in lname:
        print(name, "->", type(module))

print("\n===== MODULES WITH attn/mlp/ffn =====")
for name, module in model.named_modules():
    lname = name.lower()
    if "attn" in lname or "mlp" in lname or "ffn" in lname:
        print(name, "->", type(module))

print("\n===== FIRST 300 MODULES =====")
for i, (name, module) in enumerate(model.named_modules()):
    print(name, "->", type(module))
    if i >= 299:
        break

