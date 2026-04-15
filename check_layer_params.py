from transformers import AutoModelForCausalLM
import torch, re

model_path = '/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='cpu',
    low_cpu_mem_usage=True, trust_remote_code=True
)

total_params = 0
layer_params = {}
for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    match = re.search(r'\.(\d+)\.', name)
    if match:
        layer_num = int(match.group(1))
        layer_params[layer_num] = layer_params.get(layer_num, 0) + num_params

print(f"\n{'Layer':<10} {'Parameters':>20} {'Percentage':>15}")
print("="*50)
for layer_num in sorted(layer_params.keys()):
    params = layer_params[layer_num]
    print(f"Layer {layer_num:<3} {params:>20,} {params/total_params*100:>14.2f}%")
print("="*50)
print(f"{'Total':<10} {total_params:>20,}")
