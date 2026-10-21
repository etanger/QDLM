import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct"

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

activations = {}

def make_hook(name):
    def hook(module, inputs, output):
        x = inputs[0].detach().float().cpu()
        activations[name] = x
    return hook

# 🔥 注册 hook（我们刚选好的）
hooks = []

hooks.append(model.model.transformer.blocks[1].q_proj.register_forward_hook(make_hook("layer1_q_proj")))

hooks.append(model.model.transformer.blocks[31].ff_proj.register_forward_hook(make_hook("layer31_ff_proj")))

hooks.append(model.model.transformer.blocks[31].up_proj.register_forward_hook(make_hook("layer31_up_proj")))

# 🔥 随便一个输入（先测试）
prompt = "John has 3 apples. He buys 5 more and then gives away 2. How many apples does he have now? Explain step by step."
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    _ = model(**inputs)

# 保存
for k, v in activations.items():
    torch.save(v, f"{k}.pt")
    print(k, v.shape)

# 清理 hook
for h in hooks:
    h.remove()

