import random

import numpy as np
import torch

from auto_gptq import (
    AutoGPTQForCausalLM,
    BaseQuantizeConfig,
)
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
)


MODEL_PATH = (
    "/workspace/hdd/datasets/zwang97/"
    "models/LLaDA-8B-Instruct"
)

TARGET_LAYER = 8
WBITS = 2
GROUP_SIZE = 128

print("=" * 72)
print("Exp103: Saliency Preservation Probe")
print(f"Target layer: {TARGET_LAYER}")
print(f"Weight bits: {WBITS}")
print(f"Group size: {GROUP_SIZE}")
print("=" * 72)


def get_wikitext2(
    nsamples,
    seed,
    seqlen,
    model_path,
):
    traindata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
    )

    trainenc = tokenizer(
        "\n\n".join(traindata["text"]),
        return_tensors="pt",
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    traindataset = []

    for _ in range(nsamples):
        start = random.randint(
            0,
            trainenc.input_ids.shape[1]
            - seqlen
            - 1,
        )

        end = start + seqlen

        inp = trainenc.input_ids[
            :,
            start:end,
        ]

        attention_mask = torch.ones_like(
            inp
        )

        traindataset.append({
            "input_ids": inp,
            "attention_mask": attention_mask,
        })

    return traindataset


# ============================================================
# 1. Build GPTQ model.
#
# _base.py currently controls the layer policy:
# only Layer 8 is quantized; all other layers remain FP16.
# ============================================================

quantize_config = BaseQuantizeConfig(
    bits=WBITS,
    group_size=GROUP_SIZE,
    desc_act=False,
)

print("Loading model for GPTQ...")

q_model = AutoGPTQForCausalLM.from_pretrained(
    MODEL_PATH,
    quantize_config,
    trust_remote_code=True,
)

print("Preparing GPTQ calibration data...")

calib_data = get_wikitext2(
    nsamples=128,
    seed=0,
    seqlen=2048,
    model_path=MODEL_PATH,
)

print("Quantizing target layer...")

q_model.quantize(
    calib_data
)

q_model = q_model.to(
    "cuda:1"
)

q_model.eval()

print("GPTQ model ready.")


# ============================================================
# 2. Load untouched FP16/BF16 reference model.
# ============================================================

print("Loading FP16 reference model...")

fp_model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(
    "cuda:0"
)

fp_model.eval()

print("FP16 reference model ready.")

print("=" * 72)
print("Model preparation complete.")
print("=" * 72)


# ============================================================
# 3. Locate target layer and the four GPTQ operators.
# ============================================================

MODULES = [
    "attn_out",
    "ff_proj",
    "up_proj",
    "ff_out",
]


def get_blocks(model):
    if (
        hasattr(model, "model")
        and hasattr(model.model, "transformer")
    ):
        return model.model.transformer.blocks

    if (
        hasattr(model, "model")
        and hasattr(model.model, "model")
        and hasattr(
            model.model.model,
            "transformer",
        )
    ):
        return model.model.model.transformer.blocks

    if hasattr(model, "transformer"):
        return model.transformer.blocks

    raise RuntimeError(
        "Cannot locate transformer blocks in "
        f"{type(model)}"
    )


q_blocks = get_blocks(q_model)
fp_blocks = get_blocks(fp_model)

print()
print("Number of q_model blocks:", len(q_blocks))
print("Number of fp_model blocks:", len(fp_blocks))

q_block = q_blocks[TARGET_LAYER]
fp_block = fp_blocks[TARGET_LAYER]

print()
print("=" * 72)
print(f"Layer {TARGET_LAYER} operator check")
print("=" * 72)

for module_name in MODULES:
    if not hasattr(q_block, module_name):
        raise RuntimeError(
            f"q_model Layer {TARGET_LAYER} "
            f"has no module: {module_name}"
        )

    if not hasattr(fp_block, module_name):
        raise RuntimeError(
            f"fp_model Layer {TARGET_LAYER} "
            f"has no module: {module_name}"
        )

    q_module = getattr(
        q_block,
        module_name,
    )

    fp_module = getattr(
        fp_block,
        module_name,
    )

    print()
    print(f"[{module_name}]")
    print(
        "  FP16 type:",
        type(fp_module),
    )
    print(
        "  GPTQ type:",
        type(q_module),
    )

    if hasattr(fp_module, "weight"):
        print(
            "  FP16 weight shape:",
            tuple(fp_module.weight.shape),
        )

    for attr in [
        "infeatures",
        "outfeatures",
        "in_features",
        "out_features",
        "bits",
        "group_size",
    ]:
        if hasattr(q_module, attr):
            print(
                f"  GPTQ {attr}:",
                getattr(q_module, attr),
            )

print()
print("=" * 72)
print("Operator lookup complete.")
print("=" * 72)
