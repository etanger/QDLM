'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import time
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
# from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
import json
from datetime import datetime

from transformers import AutoTokenizer, AutoModel
# from generate import generate

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,
             fp_model=None, q_model=None, quant_start_step: int = 0):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        fp_model: fp16 model
        q_model: quantimize model
        quant_start_step: when to switch the model
    '''

    step_losses = []
    step_confidences = []
    step_losses = []
    step_confidences = []
    step_layer_hidden_norm = []
    step_layer_hidden_var = []
    step_layer_hidden_max = []
    step_mask_ratio = []
    step_layer_hidden_p99 = []
    step_layer_hidden_p999 = []
    step_layer_hidden_p9999 = []
    step_layer_hidden_top001_mean = []

    dump_full_act = os.environ.get("LLADA_DUMP_FULL_ACT", "0") == "1"
    dump_root = os.environ.get("LLADA_DUMP_ROOT", "activation_dump")
    dump_tag = os.environ.get("LLADA_DUMP_TAG", "run")

    dump_selected_layers = {1, 8, 16, 23, 27, 31}
    dump_selected_steps = {0, 4, 16, 32, 63}
    dump_linear_modules = {"q_proj", "attn_out", "ff_proj", "up_proj", "ff_out"}

    dump_full_hidden = {}
    dump_linear_inputs = {}

    # ============================================================
    # Exp62: single-layer causal quantization analysis
    #
    # IMPORTANT:
    #   FP16 model drives the denoising trajectory.
    #   Quantized model is only used as a probe on the SAME x_t.
    # ============================================================
    pair_analysis = False

    pair_analysis_target_layer = 24

    # gen_length = 64, so useful denoising transitions are mainly 0-63.
    pair_analysis_steps = {0, 8, 16, 32, 48, 63}

    pair_analysis_file = (
        "step_logs/layer_causal/"
        "exp67_single_quant_layer24.jsonl"
    )


    pair_analysis_records = []

    if pair_analysis:
        os.makedirs(
            os.path.dirname(pair_analysis_file),
            exist_ok=True,
        )
    # ============================================================

    # ============================================================
    # Exp63: FP16 standardized depth perturbation analysis
    #
    # Goal:
    #   Measure intrinsic depth-wise numerical sensitivity
    #   BEFORE using any mixed-precision policy.
    #
    # FP16 always drives the denoising trajectory.
    # Perturbed forwards are probes only and never update x.
    # ============================================================
    perturb_analysis = False

    perturb_layers = [0, 4, 8, 12, 16, 20, 24, 28]
    perturb_steps = {0, 16, 32, 48, 63}

    perturb_alpha = 0.01

    # Two independent normalized Gaussian directions.
    perturb_noise_seeds = [1234, 5678]

    perturb_analysis_file = (
        "step_logs/depth_perturbation/"
        "exp63_fp16_standardized_alpha001_limit5.jsonl"
    )

    perturb_analysis_records = []

    if perturb_analysis:
        os.makedirs(
            os.path.dirname(perturb_analysis_file),
            exist_ok=True,
        )
    # ============================================================

    # ============================================================
    # Exp68: weight-activation decomposition probe
    #
    # Goal:
    #   Explain why the same 2-bit quantization produces
    #   different local distortion across network depth.
    #
    # FP16 model drives the trajectory.
    # q_model supplies the quantized linear operators only.
    # ============================================================
    c_analysis = False

    c_target_layer = 24

    c_analysis_steps = {0, 16, 32, 48, 63}

    c_analysis_modules = [
        "attn_out",
        "ff_proj",
        "up_proj",
        "ff_out",
    ]

    c_iso_seeds = [1234, 5678]

    c_analysis_file = (
        "step_logs/weight_activation_probe/"
        "exp72_cprobe_layer24_limit2.jsonl"
    )

    c_analysis_records = []

    if c_analysis:
        os.makedirs(
            os.path.dirname(c_analysis_file),
            exist_ok=True,
        )
    # ============================================================

    # ============================================================
    # Exp73: within-block operator causal analysis
    # ============================================================

    op_analysis = False

    op_target_layer = 8

    op_analysis_steps = {0, 16, 32, 48, 63}

    op_modules = [
        "attn_out",
        "ff_proj",
        "up_proj",
        "ff_out",
    ]

    op_analysis_file = (
        "step_logs/operator_causal/"
        "exp73_operator_causal_layer8_limit2.jsonl"
    )

    op_analysis_records = []

    if op_analysis:
        os.makedirs(
            os.path.dirname(op_analysis_file),
            exist_ok=True,
    )

    # ============================================================

    # ============================================================
    # Exp101: FP16 activation saliency map
    #
    # Goal:
    #   Measure how activation saliency / outlier structure
    #   changes across network depth.
    #
    # FP16 model drives the trajectory.
    # No activation tensor is dumped to disk.
    # ============================================================

    saliency_analysis = False

    saliency_layers = [
        0, 4, 8, 12, 16, 20, 24, 28
    ]

    saliency_steps = {
        0, 16, 32, 48, 63
    }

    saliency_modules = [
        "attn_out",
        "ff_proj",
        "up_proj",
        "ff_out",
    ]

    saliency_topk_save = 128

    saliency_analysis_file = (
        "step_logs/activation_saliency/"
        "exp101_activation_saliency_map_limit2.jsonl"
    )

    saliency_records = []

    if saliency_analysis:
        os.makedirs(
            os.path.dirname(
                saliency_analysis_file
            ),
            exist_ok=True,
        )

    # ============================================================

    # ============================================================
    # Exp103: real-trajectory salient-subspace preservation
    #
    # FP16 model drives the denoising trajectory.
    # W2 model is only probed on the same real x_t.
    # ============================================================

    sp_analysis = False

    sp_target_layer = int(
        os.environ.get(
            "EXP104_TARGET_LAYER",
            "8",
        )
    )

    sp_analysis_steps = {
        0, 16, 32, 48, 63
    }

    sp_modules = [
        "attn_out",
    ]

    sp_top_fraction = 0.01

    sp_bits = int(
        os.environ.get(
            "GPTQ_PROBE_BITS",
            "2",
        )
    )

    sp_analysis_file = (
        "step_logs/gptq_probe/"
        f"gptq_probe_layer{sp_target_layer}_"
        f"w{sp_bits}_limit2.jsonl"
    )
    sp_analysis_records = []

    # ============================================================
    # Channel-level activation / quantization-error relation.
    # ============================================================
    channel_relation_dir = (
        "step_logs/channel_relation"
    )

    os.makedirs(
        channel_relation_dir,
        exist_ok=True,
    )

    channel_error_file = (
        f"{channel_relation_dir}/"
        f"column_error_layer"
        f"{sp_target_layer}_"
        f"w{sp_bits}.pt"
    )

    inference_energy_file = (
        f"{channel_relation_dir}/"
        f"inference_energy_layer"
        f"{sp_target_layer}_"
        f"w{sp_bits}_limit2.jsonl"
    )

    # These accumulate the real attn_out activation energy
    # over the selected denoising steps for ONE example.
    channel_energy_sum = None
    channel_energy_count = 0
    # ============================================================



    if sp_analysis:
        os.makedirs(
            os.path.dirname(
                sp_analysis_file
            ),
            exist_ok=True,
        )

    # ============================================================

    # ============================================================
    # Exp104: causal salient-channel recovery curve
    #
    # Restore selected attn_out input-channel contributions
    # from W2 back to FP16, then measure target-block Q.
    # ============================================================

    recovery_analysis = False

    recovery_target_layer = int(
        os.environ.get(
            "EXP104_TARGET_LAYER",
            "8",
        )
    )

    recovery_analysis_steps = {
        0, 16, 32, 48, 63
    }

    recovery_fractions = [
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
    ]

    recovery_analysis_file = (
        "step_logs/saliency_recovery/"
        f"exp104_saliency_recovery_layer"
        f"{recovery_target_layer}_limit2.jsonl"
    )

    recovery_analysis_records = []

    if recovery_analysis:
        os.makedirs(
            os.path.dirname(
                recovery_analysis_file
            ),
            exist_ok=True,
        )

    # ============================================================


    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            start_time = time.time()  # 记录开始时间
            # ===== 内存监控 =====
            if i % 10 == 0:  # 每 10 步打印一次
                if torch.cuda.is_available():
                    gpu0_alloc = torch.cuda.memory_allocated(0) / 1024**3
                    gpu0_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    if torch.cuda.device_count() > 1:
                        gpu1_alloc = torch.cuda.memory_allocated(1) / 1024**3
                        gpu1_total = torch.cuda.get_device_properties(1).total_memory / 1024**3
                        print(f"[Step {i:3d}/{steps}] GPU0: {gpu0_alloc:5.2f}/{gpu0_total:.2f} GB | GPU1: {gpu1_alloc:5.2f}/{gpu1_total:.2f} GB")
                    else:
                        print(f"[Step {i:3d}/{steps}] GPU0: {gpu0_alloc:5.2f}/{gpu0_total:.2f} GB")
            # ====================

            mask_index = (x == mask_id)
            mask_ratio = (x == mask_id).float().mean().item()
            step_mask_ratio.append(mask_ratio)

            # ===== switch model =====
            # cur_model = model
            # if (fp_model is not None) and (q_model is not None):
            #     cur_model = q_model if i >= quant_start_step else fp_model
            #     model_name = "quant" if i >= quant_start_step else "fp16"
            # ==========================
            # ===== switch model =====
            cur_model = model

            if (
                (
                    pair_analysis or perturb_analysis or c_analysis or saliency_analysis or sp_analysis
                    or recovery_analysis
                )
                and (fp_model is not None)
                and (q_model is not None)
            ):
                # Causal-analysis mode:
                # FP16 always drives x_t -> x_{t+1}.
                # q_model is only probed on the same x_t.
                cur_model = fp_model
                model_name = "fp16_driver"

            elif (fp_model is not None) and (q_model is not None):
                cur_model = (
                    q_model
                    if i >= quant_start_step
                    else fp_model
                )
                model_name = (
                    "quant"
                    if i >= quant_start_step
                    else "fp16"
                )
            # ==========================
            # ===== 跨 GPU 数据传输 =====
            base_device = model.device
            x_input = x.to(cur_model.device)
            # =============================

            layer_outputs = []
            handles = []

            def make_hook(layer_idx):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output

                    layer_outputs.append(out.detach())

                    if dump_full_act and (i in dump_selected_steps) and (layer_idx in dump_selected_layers):
                        dump_full_hidden[f"step_{i}_layer_{layer_idx}_block_out"] = out.detach().float().cpu()

                    return output
                return hook

            def make_linear_hook(layer_idx, module_name):
                def linear_hook(module, inputs, output):
                    if dump_full_act and (i in dump_selected_steps) and (layer_idx in dump_selected_layers):
                        x_lin = inputs[0].detach().float().cpu()
                        dump_linear_inputs[f"step_{i}_layer_{layer_idx}_{module_name}_INPUT"] = x_lin
                    return None
                return linear_hook

            # ============================================================
            # Exp101 helper:
            # summarize channel-wise activation saliency.
            # ============================================================
            def summarize_saliency(x_act):
                if (
                    x_act is None
                    or (not torch.is_tensor(x_act))
                    or x_act.numel() == 0
                ):
                    return None

                x_float = x_act.detach().float()

                dim = int(x_float.shape[-1])

                # Flatten batch / token dimensions:
                # [..., D] -> [N, D]
                x_2d = x_float.reshape(-1, dim)

                # Channel energy:
                # E_j = mean_n x_{n,j}^2
                channel_energy = (
                    x_2d
                    .pow(2)
                    .mean(dim=0)
                )

                eps = 1e-12

                total_energy = channel_energy.sum()

                # Top 1% channel count.
                k_1pct = max(
                    1,
                    (dim + 99) // 100,
                )

                k_save = min(
                    dim,
                    max(
                        saliency_topk_save,
                        k_1pct,
                        5,
                    ),
                )

                top_values, top_indices = torch.topk(
                    channel_energy,
                    k=k_save,
                    largest=True,
                    sorted=True,
                )

                c1 = (
                    top_values[0]
                    / (total_energy + eps)
                ).item()

                c5 = (
                    top_values[:min(5, dim)].sum()
                    / (total_energy + eps)
                ).item()

                c1pct = (
                    top_values[:k_1pct].sum()
                    / (total_energy + eps)
                ).item()

                # Participation-ratio effective dimension.
                effective_dim = (
                    total_energy.pow(2)
                    / (
                        channel_energy.pow(2).sum()
                        + eps
                    )
                ).item()

                channel_rms = torch.sqrt(
                    channel_energy + eps
                )

                max_channel_rms = (
                    channel_rms.max()
                )

                median_channel_rms = (
                    channel_rms.median()
                )

                max_channel_rms_over_median = (
                    max_channel_rms
                    / (
                        median_channel_rms
                        + eps
                    )
                ).item()

                global_rms = torch.sqrt(
                    x_2d.pow(2).mean()
                    + eps
                )

                max_abs = x_2d.abs().max()

                max_abs_over_global_rms = (
                    max_abs
                    / (
                        global_rms
                        + eps
                    )
                ).item()

                top_energy_fraction = (
                    top_values
                    / (
                        total_energy
                        + eps
                    )
                )

                return {
                    "num_tokens": int(
                        x_2d.shape[0]
                    ),
                    "dim": dim,

                    "total_energy": float(
                        total_energy.item()
                    ),

                    "c1": float(c1),
                    "c5": float(c5),
                    "c1pct": float(c1pct),

                    "effective_dim": float(
                        effective_dim
                    ),

                    "max_channel_rms_over_median": float(
                        max_channel_rms_over_median
                    ),

                    "max_abs_over_global_rms": float(
                        max_abs_over_global_rms
                    ),

                    "top_indices": (
                        top_indices
                        .detach()
                        .cpu()
                        .tolist()
                    ),

                    "top_energy_fraction": (
                        top_energy_fraction
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                }
            # ============================================================

            # ============================================================
            # Exp101 hook:
            # collect saliency statistics from linear-module inputs.
            # ============================================================
            def make_saliency_hook(
                layer_idx,
                module_name,
            ):
                def saliency_hook(
                    module,
                    inputs,
                    output,
                ):
                    if len(inputs) == 0:
                        return None

                    x_act = inputs[0]

                    if (
                        (not torch.is_tensor(x_act))
                        or x_act.numel() == 0
                    ):
                        return None

                    # Statistics over all tokens.
                    all_stats = summarize_saliency(
                        x_act
                    )

                    # Statistics over currently masked tokens only.
                    masked_stats = None

                    if x_act.dim() == 3:
                        local_mask = mask_index.to(
                            x_act.device
                        )

                        # Expected:
                        # x_act:     [B, S, D]
                        # local_mask:[B, S]
                        if (
                            tuple(local_mask.shape)
                            ==
                            tuple(x_act.shape[:2])
                        ):
                            masked_act = x_act[
                                local_mask
                            ]

                            if masked_act.numel() > 0:
                                masked_stats = summarize_saliency(
                                    masked_act
                                )

                    saliency_records.append({
                        "step": int(i),
                        "layer": int(layer_idx),
                        "module": module_name,
                        "mask_ratio": float(
                            mask_ratio
                        ),
                        "all": all_stats,
                        "masked": masked_stats,
                    })

                    return None

                return saliency_hook
            # ============================================================


            if hasattr(cur_model, "model") and hasattr(cur_model.model, "transformer"):
                blocks = cur_model.model.transformer.blocks
            elif hasattr(cur_model, "model") and hasattr(cur_model.model, "model") and hasattr(cur_model.model.model, "transformer"):
                blocks = cur_model.model.model.transformer.blocks
            else:
                raise AttributeError(f"Cannot find transformer blocks in cur_model of type {type(cur_model)}")

            # ============================================================
            # Exp68: capture REAL FP16 inputs to selected linear modules
            # at the target layer.
            # ============================================================
            c_real_inputs = {}

            if (
                c_analysis
                and (i in c_analysis_steps)
            ):
                if c_target_layer >= len(blocks):
                    raise RuntimeError(
                        f"Invalid c_target_layer={c_target_layer}; "
                        f"num_layers={len(blocks)}"
                    )

                c_fp_block = blocks[c_target_layer]

                def make_c_input_hook(module_name):
                    def hook(module, inputs, output):
                        if (
                            module_name not in c_real_inputs
                            and len(inputs) > 0
                        ):
                            c_real_inputs[module_name] = (
                                inputs[0].detach()
                            )
                        return output
                    return hook

                for module_name in c_analysis_modules:
                    if hasattr(c_fp_block, module_name):
                        handles.append(
                            getattr(
                                c_fp_block,
                                module_name
                            ).register_forward_hook(
                                make_c_input_hook(
                                    module_name
                                )
                            )
                        )
            # ============================================================

            # ============================================================
            # Exp103: capture REAL FP16 inputs to selected operators.
            #
            # These are the actual activations encountered on the
            # FP16-driven denoising trajectory.
            # ============================================================
            sp_real_inputs = {}

            if (
                sp_analysis
                and (i in sp_analysis_steps)
            ):
                if sp_target_layer >= len(blocks):
                    raise RuntimeError(
                        f"Invalid sp_target_layer={sp_target_layer}; "
                        f"num_layers={len(blocks)}"
                    )

                sp_fp_block = blocks[
                    sp_target_layer
                ]

                def make_sp_input_hook(
                    module_name
                ):
                    def hook(
                        module,
                        inputs,
                        output,
                    ):
                        if (
                            module_name
                            not in sp_real_inputs
                            and len(inputs) > 0
                        ):
                            sp_real_inputs[
                                module_name
                            ] = (
                                inputs[0].detach()
                            )

                        return output

                    return hook

                for module_name in sp_modules:
                    if hasattr(
                        sp_fp_block,
                        module_name,
                    ):
                        handles.append(
                            getattr(
                                sp_fp_block,
                                module_name,
                            ).register_forward_hook(
                                make_sp_input_hook(
                                    module_name
                                )
                            )
                        )
            # ============================================================

            # ============================================================
            # Exp104: capture the REAL FP16 input to attn_out
            # at the target layer.
            # ============================================================
            recovery_real_inputs = {}

            if (
                recovery_analysis
                and (i in recovery_analysis_steps)
            ):
                if recovery_target_layer >= len(blocks):
                    raise RuntimeError(
                        f"Invalid recovery_target_layer="
                        f"{recovery_target_layer}; "
                        f"num_layers={len(blocks)}"
                    )

                recovery_fp_block = blocks[
                    recovery_target_layer
                ]

                def recovery_input_hook(
                    module,
                    inputs,
                    output,
                ):
                    if (
                        "attn_out"
                        not in recovery_real_inputs
                        and len(inputs) > 0
                    ):
                        recovery_real_inputs[
                            "attn_out"
                        ] = inputs[0].detach()

                    return output

                handles.append(
                    recovery_fp_block.attn_out
                    .register_forward_hook(
                        recovery_input_hook
                    )
                )
            # ============================================================


            for layer_idx, block in enumerate(blocks):
                handles.append(block.register_forward_hook(make_hook(layer_idx)))

                if (
                    saliency_analysis
                    and (i in saliency_steps)
                    and (layer_idx in saliency_layers)
                ):
                    for module_name in saliency_modules:
                        if hasattr(block, module_name):
                            handles.append(
                                getattr(
                                    block,
                                    module_name,
                                ).register_forward_hook(
                                    make_saliency_hook(
                                        layer_idx,
                                        module_name,
                                    )
                                )
                            )

                if dump_full_act and (i in dump_selected_steps) and (layer_idx in dump_selected_layers):
                    for module_name in dump_linear_modules:
                        if hasattr(block, module_name):
                            handles.append(
                                getattr(block, module_name).register_forward_hook(
                                    make_linear_hook(layer_idx, module_name)
                                )
                            )


            if cfg_scale > 0.:
                un_x = x_input.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x_input, un_x], dim=0)

                logits = cur_model(x_).logits # use current model

                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:

                logits = cur_model(x_input).logits # use current model

            # ============================================================
            # Exp103: real-trajectory salient-subspace preservation
            # ============================================================
            if (
                sp_analysis
                and (i in sp_analysis_steps)
                and (fp_model is not None)
                and (q_model is not None)
            ):
                if cfg_scale != 0.0:
                    raise RuntimeError(
                        "sp_analysis currently requires cfg_scale == 0"
                    )

                # --------------------------------------------------------
                # Locate quantized transformer blocks.
                # --------------------------------------------------------
                if (
                    hasattr(q_model, "model")
                    and hasattr(q_model.model, "transformer")
                ):
                    sp_q_blocks = (
                        q_model.model.transformer.blocks
                    )

                elif (
                    hasattr(q_model, "model")
                    and hasattr(q_model.model, "model")
                    and hasattr(
                        q_model.model.model,
                        "transformer",
                    )
                ):
                    sp_q_blocks = (
                        q_model.model.model
                        .transformer.blocks
                    )

                else:
                    raise AttributeError(
                        "Cannot find transformer blocks "
                        "in q_model of type "
                        f"{type(q_model)}"
                    )

                sp_fp_block = blocks[
                    sp_target_layer
                ]

                sp_q_block = sp_q_blocks[
                    sp_target_layer
                ]

                def sp_get_module_device(
                    module,
                    fallback,
                ):
                    for param in module.parameters():
                        return param.device

                    for buf in module.buffers():
                        return buf.device

                    return fallback

                # --------------------------------------------------------
                # Probe each quantized operator using the REAL FP16
                # activation captured on this denoising step.
                # --------------------------------------------------------
                for module_name in sp_modules:

                    if module_name not in sp_real_inputs:
                        continue

                    if not hasattr(
                        sp_fp_block,
                        module_name,
                    ):
                        continue

                    if not hasattr(
                        sp_q_block,
                        module_name,
                    ):
                        continue

                    fp_module = getattr(
                        sp_fp_block,
                        module_name,
                    )

                    q_module = getattr(
                        sp_q_block,
                        module_name,
                    )

                    x_real = sp_real_inputs[
                        module_name
                    ]

                    # ----------------------------------------------------
                    # Restrict to currently masked tokens.
                    # ----------------------------------------------------
                    module_mask = mask_index.to(
                        x_real.device
                    )

                    x_full = x_real[
                        module_mask
                    ]

                    if x_full.numel() == 0:
                        continue

                    # ----------------------------------------------------
                    # Channel saliency:
                    #
                    # E_j = mean_n x[n,j]^2
                    #
                    # Top 1% is determined independently for the
                    # current real activation.
                    # ----------------------------------------------------
                    x_stats = (
                        x_full
                        .detach()
                        .float()
                    )

                    channel_energy = (
                        x_stats
                        .pow(2)
                        .mean(dim=0)
                    )


                    # ============================================
                    # Accumulate the FULL per-channel activation
                    # energy vector over selected denoising steps.
                    # ============================================
                    if module_name == "attn_out":
                        current_energy = (
                            channel_energy
                            .detach()
                            .double()
                            .cpu()
                        )

                        if channel_energy_sum is None:
                            channel_energy_sum = (
                                torch.zeros_like(
                                    current_energy
                                )
                            )

                        channel_energy_sum += (
                            current_energy
                        )

                        channel_energy_count += 1
                    # ============================================

                    hidden_dim = int(
                        channel_energy.numel()
                    )

                    # ============================================
                    # Channel-level quantization-error probe.
                    #
                    # Run only once per Layer / bit-width.
                    #
                    # For a linear operator:
                    #     y = x W^T
                    #
                    # Feeding basis vector e_j isolates input
                    # channel j, so the centered output is W[:,j].
                    # ============================================
                    if (
                        module_name == "attn_out"
                        and not os.path.exists(
                            channel_error_file
                        )
                    ):
                        fp_probe_device = (
                            sp_get_module_device(
                                fp_module,
                                x_full.device,
                            )
                        )

                        q_probe_device = (
                            sp_get_module_device(
                                q_module,
                                q_model.device,
                            )
                        )

                        probe_dtype = (
                            x_full.dtype
                        )

                        column_abs_error = (
                            torch.empty(
                                hidden_dim,
                                dtype=torch.float32,
                            )
                        )

                        column_fp_norm = (
                            torch.empty(
                                hidden_dim,
                                dtype=torch.float32,
                            )
                        )

                        chunk_size = 128

                        with torch.no_grad():

                            fp_zero_input = (
                                torch.zeros(
                                    1,
                                    hidden_dim,
                                    device=(
                                        fp_probe_device
                                    ),
                                    dtype=probe_dtype,
                                )
                            )

                            q_zero_input = (
                                torch.zeros(
                                    1,
                                    hidden_dim,
                                    device=(
                                        q_probe_device
                                    ),
                                    dtype=probe_dtype,
                                )
                            )

                            fp_zero_output = (
                                fp_module(
                                    fp_zero_input
                                )
                                .detach()
                                .float()
                                .cpu()
                            )

                            q_zero_output = (
                                q_module(
                                    q_zero_input
                                )
                                .detach()
                                .float()
                                .cpu()
                            )

                            for start in range(
                                0,
                                hidden_dim,
                                chunk_size,
                            ):
                                end = min(
                                    start
                                    + chunk_size,
                                    hidden_dim,
                                )

                                batch = (
                                    end - start
                                )

                                fp_basis = (
                                    torch.zeros(
                                        batch,
                                        hidden_dim,
                                        device=(
                                            fp_probe_device
                                        ),
                                        dtype=probe_dtype,
                                    )
                                )

                                q_basis = (
                                    torch.zeros(
                                        batch,
                                        hidden_dim,
                                        device=(
                                            q_probe_device
                                        ),
                                        dtype=probe_dtype,
                                    )
                                )

                                fp_rows = (
                                    torch.arange(
                                        batch,
                                        device=(
                                            fp_probe_device
                                        ),
                                    )
                                )

                                fp_cols = (
                                    torch.arange(
                                        start,
                                        end,
                                        device=(
                                            fp_probe_device
                                        ),
                                    )
                                )

                                q_rows = (
                                    torch.arange(
                                        batch,
                                        device=(
                                            q_probe_device
                                        ),
                                    )
                                )

                                q_cols = (
                                    torch.arange(
                                        start,
                                        end,
                                        device=(
                                            q_probe_device
                                        ),
                                    )
                                )

                                fp_basis[
                                    fp_rows,
                                    fp_cols,
                                ] = 1

                                q_basis[
                                    q_rows,
                                    q_cols,
                                ] = 1

                                fp_columns = (
                                    fp_module(
                                        fp_basis
                                    )
                                    .detach()
                                    .float()
                                    .cpu()
                                    - fp_zero_output
                                )

                                q_columns = (
                                    q_module(
                                        q_basis
                                    )
                                    .detach()
                                    .float()
                                    .cpu()
                                    - q_zero_output
                                )

                                column_diff = (
                                    q_columns
                                    - fp_columns
                                )

                                column_abs_error[
                                    start:end
                                ] = (
                                    torch.linalg.vector_norm(
                                        column_diff,
                                        dim=1,
                                    )
                                )

                                column_fp_norm[
                                    start:end
                                ] = (
                                    torch.linalg.vector_norm(
                                        fp_columns,
                                        dim=1,
                                    )
                                )

                        column_rel_error = (
                            column_abs_error
                            /
                            (
                                column_fp_norm
                                + 1e-12
                            )
                        )

                        torch.save(
                            {
                                "layer": int(
                                    sp_target_layer
                                ),
                                "bits": int(
                                    sp_bits
                                ),
                                "module": (
                                    "attn_out"
                                ),
                                "hidden_dim": int(
                                    hidden_dim
                                ),
                                "column_abs_error": (
                                    column_abs_error
                                ),
                                "column_fp_norm": (
                                    column_fp_norm
                                ),
                                "column_rel_error": (
                                    column_rel_error
                                ),
                            },
                            channel_error_file,
                        )

                        print(
                            "[CHANNEL RELATION] "
                            "saved column error:",
                            channel_error_file,
                        )
                    # ============================================

                    k_salient = max(
                        1,
                        (hidden_dim + 99) // 100,
                    )

                    salient_indices = torch.topk(
                        channel_energy,
                        k=k_salient,
                        largest=True,
                    ).indices

                    channel_mask = torch.zeros(
                        hidden_dim,
                        device=x_full.device,
                        dtype=torch.bool,
                    )

                    channel_mask[
                        salient_indices.to(
                            channel_mask.device
                        )
                    ] = True

                    # ----------------------------------------------------
                    # Decompose:
                    #
                    # X_full = X_salient + X_rest
                    # ----------------------------------------------------
                    x_salient = torch.zeros_like(
                        x_full
                    )

                    x_salient[
                        ...,
                        channel_mask,
                    ] = x_full[
                        ...,
                        channel_mask,
                    ]

                    x_rest = (
                        x_full
                        - x_salient
                    )

                    # ----------------------------------------------------
                    # Matched-size random 1% control.
                    # Deterministic per step / layer / module.
                    # ----------------------------------------------------
                    module_seed_offsets = {
                        "attn_out": 23,
                        "ff_proj": 37,
                        "up_proj": 53,
                        "ff_out": 71,
                    }

                    random_seed = (
                        103000
                        + 1009 * int(
                            sp_target_layer
                        )
                        + 9176 * int(i)
                        + module_seed_offsets[
                            module_name
                        ]
                    )

                    generator = torch.Generator(
                        device=x_full.device
                    )

                    generator.manual_seed(
                        random_seed
                    )

                    random_indices = torch.randperm(
                        hidden_dim,
                        generator=generator,
                        device=x_full.device,
                    )[:k_salient]

                    random_mask = torch.zeros(
                        hidden_dim,
                        device=x_full.device,
                        dtype=torch.bool,
                    )

                    random_mask[
                        random_indices
                    ] = True

                    x_random = torch.zeros_like(
                        x_full
                    )

                    x_random[
                        ...,
                        random_mask,
                    ] = x_full[
                        ...,
                        random_mask,
                    ]

                    # ----------------------------------------------------
                    # Devices.
                    # ----------------------------------------------------
                    fp_device = (
                        sp_get_module_device(
                            fp_module,
                            x_full.device,
                        )
                    )

                    q_device = (
                        sp_get_module_device(
                            q_module,
                            q_model.device,
                        )
                    )

                    # ----------------------------------------------------
                    # Helper:
                    #
                    # Use centered operator output
                    #
                    #   g(x) = f(x) - f(0)
                    #
                    # so bias cannot break:
                    #
                    #   g(X_full)
                    #     = g(X_salient) + g(X_rest)
                    # ----------------------------------------------------
                    def run_sp_operator(
                        module,
                        x_value,
                        device,
                    ):
                        x_value = x_value.to(
                            device
                        )

                        zero_value = (
                            torch.zeros_like(
                                x_value
                            )
                        )

                        y_value = module(
                            x_value
                        )

                        y_zero = module(
                            zero_value
                        )

                        return (
                            y_value
                            - y_zero
                        ).detach().float().cpu()

                    # FP16
                    fp_full = run_sp_operator(
                        fp_module,
                        x_full,
                        fp_device,
                    )

                    fp_salient = run_sp_operator(
                        fp_module,
                        x_salient,
                        fp_device,
                    )

                    fp_rest = run_sp_operator(
                        fp_module,
                        x_rest,
                        fp_device,
                    )

                    fp_random = run_sp_operator(
                        fp_module,
                        x_random,
                        fp_device,
                    )

                    # W2
                    q_full = run_sp_operator(
                        q_module,
                        x_full,
                        q_device,
                    )

                    q_salient = run_sp_operator(
                        q_module,
                        x_salient,
                        q_device,
                    )

                    q_rest = run_sp_operator(
                        q_module,
                        x_rest,
                        q_device,
                    )

                    q_random = run_sp_operator(
                        q_module,
                        x_random,
                        q_device,
                    )

                    # ----------------------------------------------------
                    # Quantization-error vectors.
                    # ----------------------------------------------------
                    e_full = (
                        q_full
                        - fp_full
                    )

                    e_salient = (
                        q_salient
                        - fp_salient
                    )

                    e_rest = (
                        q_rest
                        - fp_rest
                    )

                    e_random = (
                        q_random
                        - fp_random
                    )

                    eps = 1e-12

                    d_full = (
                        e_full.norm()
                        /
                        (
                            fp_full.norm()
                            + eps
                        )
                    ).item()

                    d_salient = (
                        e_salient.norm()
                        /
                        (
                            fp_salient.norm()
                            + eps
                        )
                    ).item()

                    d_rest = (
                        e_rest.norm()
                        /
                        (
                            fp_rest.norm()
                            + eps
                        )
                    ).item()

                    d_random = (
                        e_random.norm()
                        /
                        (
                            fp_random.norm()
                            + eps
                        )
                    ).item()

                    # ----------------------------------------------------
                    # How much activation energy lies in top 1%.
                    # ----------------------------------------------------
                    salient_energy_fraction = (
                        channel_energy[
                            salient_indices
                        ].sum()
                        /
                        (
                            channel_energy.sum()
                            + eps
                        )
                    ).item()

                    # ----------------------------------------------------
                    # Error decomposition diagnostics.
                    # ----------------------------------------------------
                    salient_error_over_full = (
                        e_salient.norm()
                        /
                        (
                            e_full.norm()
                            + eps
                        )
                    ).item()

                    rest_error_over_full = (
                        e_rest.norm()
                        /
                        (
                            e_full.norm()
                            + eps
                        )
                    ).item()

                    error_cosine = (
                        torch.sum(
                            e_salient
                            * e_rest
                        )
                        /
                        (
                            e_salient.norm()
                            * e_rest.norm()
                            + eps
                        )
                    ).item()

                    decomposition_residual = (
                        (
                            e_full
                            - e_salient
                            - e_rest
                        ).norm()
                        /
                        (
                            e_full.norm()
                            + eps
                        )
                    ).item()

                    # ----------------------------------------------------
                    # FP16 signal contribution diagnostics.
                    # ----------------------------------------------------
                    salient_output_over_full = (
                        fp_salient.norm()
                        /
                        (
                            fp_full.norm()
                            + eps
                        )
                    ).item()

                    random_output_over_full = (
                        fp_random.norm()
                        /
                        (
                            fp_full.norm()
                            + eps
                        )
                    ).item()

                    # ----------------------------------------------------
                    # Save one scalar record.
                    # ----------------------------------------------------
                    sp_analysis_records.append({
                        "step": int(i),
                        "layer": int(
                            sp_target_layer
                        ),
                        "module": module_name,

                        "num_masked_tokens": int(
                            x_full.shape[0]
                        ),

                        "hidden_dim": int(
                            hidden_dim
                        ),

                        "k_salient": int(
                            k_salient
                        ),

                        "salient_indices": (
                            salient_indices
                            .detach()
                            .cpu()
                            .tolist()
                        ),

                        "random_indices": (
                            random_indices
                            .detach()
                            .cpu()
                            .tolist()
                        ),

                        "salient_energy_fraction": float(
                            salient_energy_fraction
                        ),

                        "d_full": float(
                            d_full
                        ),

                        "d_salient": float(
                            d_salient
                        ),

                        "d_rest": float(
                            d_rest
                        ),

                        "d_random": float(
                            d_random
                        ),

                        "salient_over_random": float(
                            d_salient
                            /
                            (
                                d_random
                                + eps
                            )
                        ),

                        "salient_error_over_full": float(
                            salient_error_over_full
                        ),

                        "rest_error_over_full": float(
                            rest_error_over_full
                        ),

                        "error_cosine": float(
                            error_cosine
                        ),

                        "decomposition_residual": float(
                            decomposition_residual
                        ),

                        "salient_output_over_full": float(
                            salient_output_over_full
                        ),

                        "random_output_over_full": float(
                            random_output_over_full
                        ),
                    })

                    del fp_full
                    del fp_salient
                    del fp_rest
                    del fp_random
                    del q_full
                    del q_salient
                    del q_rest
                    del q_random
                    del e_full
                    del e_salient
                    del e_rest
                    del e_random

            # ============================================================

            # ============================================================
            # Exp104: causal salient-channel recovery curve
            # ============================================================
            if (
                recovery_analysis
                and (i in recovery_analysis_steps)
                and (fp_model is not None)
                and (q_model is not None)
            ):
                if cfg_scale != 0.0:
                    raise RuntimeError(
                        "recovery_analysis currently requires cfg_scale == 0"
                    )

                target_layer = recovery_target_layer

                if target_layer >= len(layer_outputs):
                    raise RuntimeError(
                        f"Target layer {target_layer} not found in "
                        f"layer_outputs; len={len(layer_outputs)}"
                    )

                if "attn_out" not in recovery_real_inputs:
                    raise RuntimeError(
                        "Exp104 did not capture FP16 attn_out input"
                    )

                # FP16 target-block output from the normal
                # FP16-driven forward above.
                fp_hidden = layer_outputs[
                    target_layer
                ]

                # --------------------------------------------------------
                # Locate q_model transformer blocks.
                # --------------------------------------------------------
                if (
                    hasattr(q_model, "model")
                    and hasattr(
                        q_model.model,
                        "transformer",
                    )
                ):
                    recovery_q_blocks = (
                        q_model.model.transformer.blocks
                    )

                elif (
                    hasattr(q_model, "model")
                    and hasattr(
                        q_model.model,
                        "model",
                    )
                    and hasattr(
                        q_model.model.model,
                        "transformer",
                    )
                ):
                    recovery_q_blocks = (
                        q_model.model.model
                        .transformer.blocks
                    )

                else:
                    raise AttributeError(
                        "Cannot find transformer blocks "
                        "in q_model of type "
                        f"{type(q_model)}"
                    )

                fp_block = blocks[
                    target_layer
                ]

                q_block = recovery_q_blocks[
                    target_layer
                ]

                fp_attn = fp_block.attn_out
                q_attn = q_block.attn_out

                # --------------------------------------------------------
                # Determine salient input channels from the REAL
                # FP16 attn_out activation at this step.
                # --------------------------------------------------------
                x_real = recovery_real_inputs[
                    "attn_out"
                ]

                module_mask = mask_index.to(
                    x_real.device
                )

                x_masked = x_real[
                    module_mask
                ]

                if x_masked.numel() == 0:
                    raise RuntimeError(
                        "Exp104 found no masked tokens"
                    )

                x_stats = (
                    x_masked
                    .detach()
                    .float()
                )

                channel_energy = (
                    x_stats
                    .pow(2)
                    .mean(dim=0)
                )

                total_energy = (
                    channel_energy.sum()
                    + 1e-12
                )

                hidden_dim = int(
                    channel_energy.numel()
                )

                salient_order = torch.argsort(
                    channel_energy,
                    descending=True,
                )

                # One random permutation per example / step.
                # All random-k controls are nested, just like
                # the top-k salient sets.
                prompt_seed_offset = (
                    int(
                        prompt.detach()
                        .long()
                        .sum()
                        .item()
                    )
                    % 1000003
                )

                random_seed = (
                    104000
                    + 1009 * int(target_layer)
                    + 9176 * int(i)
                    + prompt_seed_offset
                )

                generator = torch.Generator(
                    device=channel_energy.device
                )

                generator.manual_seed(
                    random_seed
                )

                random_order = torch.randperm(
                    hidden_dim,
                    generator=generator,
                    device=channel_energy.device,
                )

                # --------------------------------------------------------
                # Device helper.
                # --------------------------------------------------------
                def recovery_module_device(
                    module,
                    fallback,
                ):
                    for p in module.parameters():
                        return p.device

                    for b in module.buffers():
                        return b.device

                    return fallback

                fp_attn_device = (
                    recovery_module_device(
                        fp_attn,
                        x_real.device,
                    )
                )

                # --------------------------------------------------------
                # Selected-channel correction:
                #
                # y_corr =
                #     q(x)
                #     + [fp(x_S) - fp(0)]
                #     - [q(x_S) - q(0)]
                #
                # Therefore only the selected input-channel
                # contribution is restored to FP16.
                # --------------------------------------------------------
                def make_recovery_hook(
                    selected_indices=None,
                    full_restore=False,
                ):
                    def hook(
                        q_module,
                        inputs,
                        output,
                    ):
                        if len(inputs) == 0:
                            raise RuntimeError(
                                "Exp104 attn_out hook "
                                "received no input"
                            )

                        x_in = inputs[0]

                        if full_restore:
                            x_fp = x_in.to(
                                fp_attn_device
                            )

                            y_fp = fp_attn(
                                x_fp
                            )

                            return y_fp.to(
                                device=output.device,
                                dtype=output.dtype,
                            )

                        idx = selected_indices.to(
                            x_in.device
                        )

                        x_selected = (
                            torch.zeros_like(
                                x_in
                            )
                        )

                        x_selected[
                            ...,
                            idx,
                        ] = x_in[
                            ...,
                            idx,
                        ]

                        x_zero = torch.zeros_like(
                            x_in
                        )

                        # Direct .forward() avoids recursively
                        # triggering this same forward hook.
                        q_selected = q_module.forward(
                            x_selected
                        )

                        q_zero = q_module.forward(
                            x_zero
                        )

                        x_selected_fp = (
                            x_selected.to(
                                fp_attn_device
                            )
                        )

                        x_zero_fp = x_zero.to(
                            fp_attn_device
                        )

                        fp_selected = fp_attn(
                            x_selected_fp
                        )

                        fp_zero = fp_attn(
                            x_zero_fp
                        )

                        correction = (
                            (
                                fp_selected
                                - fp_zero
                            )
                            .to(
                                device=output.device,
                                dtype=output.dtype,
                            )
                            -
                            (
                                q_selected
                                - q_zero
                            )
                            .to(
                                device=output.device,
                                dtype=output.dtype,
                            )
                        )

                        return (
                            output
                            + correction
                        )

                    return hook

                # --------------------------------------------------------
                # Run one counterfactual q_model forward and capture
                # the target-block output.
                # --------------------------------------------------------
                def run_recovery_variant(
                    selected_indices=None,
                    full_restore=False,
                ):
                    captured = []
                    recovery_handle = None

                    def capture_block_output(
                        module,
                        inputs,
                        output,
                    ):
                        if isinstance(
                            output,
                            tuple,
                        ):
                            out = output[0]
                        else:
                            out = output

                        captured.append(
                            out.detach()
                        )

                        return output

                    if (
                        selected_indices is not None
                        or full_restore
                    ):
                        recovery_handle = (
                            q_attn.register_forward_hook(
                                make_recovery_hook(
                                    selected_indices=(
                                        selected_indices
                                    ),
                                    full_restore=(
                                        full_restore
                                    ),
                                )
                            )
                        )

                    block_handle = (
                        q_block.register_forward_hook(
                            capture_block_output
                        )
                    )

                    try:
                        q_x_input = x.to(
                            q_model.device
                        )

                        q_out = q_model(
                            q_x_input
                        )

                        _ = q_out.logits

                    finally:
                        block_handle.remove()

                        if (
                            recovery_handle
                            is not None
                        ):
                            recovery_handle.remove()

                    if len(captured) != 1:
                        raise RuntimeError(
                            "Exp104 expected one "
                            "target-block output, got "
                            f"{len(captured)}"
                        )

                    result = captured[0]

                    del q_out

                    return result

                # --------------------------------------------------------
                # Block-Q helper.
                # Same masked-token relative L2 definition as Exp73.
                # --------------------------------------------------------
                fp_cpu = (
                    fp_hidden
                    .detach()
                    .float()
                    .cpu()
                )

                mask_cpu = (
                    mask_index
                    .detach()
                    .cpu()
                )

                fp_masked = fp_cpu[
                    mask_cpu
                ]

                def compute_recovery_q(
                    hidden,
                ):
                    h_cpu = (
                        hidden
                        .detach()
                        .float()
                        .cpu()
                    )

                    h_masked = h_cpu[
                        mask_cpu
                    ]

                    diff_masked = (
                        h_masked
                        - fp_masked
                    )

                    return (
                        diff_masked.norm()
                        /
                        (
                            fp_masked.norm()
                            + 1e-12
                        )
                    ).item()

                # --------------------------------------------------------
                # 0% baseline:
                # target layer remains normal W2.
                # --------------------------------------------------------
                baseline_hidden = (
                    run_recovery_variant()
                )

                q_baseline = (
                    compute_recovery_q(
                        baseline_hidden
                    )
                )

                # --------------------------------------------------------
                # 100% attn_out FP16 restore:
                # upper bound for what attn_out correction can recover.
                #
                # ff_proj / up_proj / ff_out remain W2.
                # --------------------------------------------------------
                full_hidden = (
                    run_recovery_variant(
                        full_restore=True
                    )
                )

                q_full_attn_restore = (
                    compute_recovery_q(
                        full_hidden
                    )
                )

                full_attn_gain = (
                    q_baseline
                    - q_full_attn_restore
                )

                # --------------------------------------------------------
                # Recovery curve:
                # 0.5%, 1%, 2%, 5%, 10%
                #
                # Compare top-k salient vs matched-size random.
                # --------------------------------------------------------
                for fraction in recovery_fractions:

                    k_channels = max(
                        1,
                        int(
                            hidden_dim
                            * float(fraction)
                            + 0.999999
                        ),
                    )

                    k_channels = min(
                        k_channels,
                        hidden_dim,
                    )

                    salient_indices = (
                        salient_order[
                            :k_channels
                        ]
                    )

                    random_indices = (
                        random_order[
                            :k_channels
                        ]
                    )

                    salient_hidden = (
                        run_recovery_variant(
                            selected_indices=(
                                salient_indices
                            )
                        )
                    )

                    random_hidden = (
                        run_recovery_variant(
                            selected_indices=(
                                random_indices
                            )
                        )
                    )

                    q_salient = (
                        compute_recovery_q(
                            salient_hidden
                        )
                    )

                    q_random = (
                        compute_recovery_q(
                            random_hidden
                        )
                    )

                    delta_q_salient = (
                        q_baseline
                        - q_salient
                    )

                    delta_q_random = (
                        q_baseline
                        - q_random
                    )

                    salient_advantage = (
                        q_random
                        - q_salient
                    )

                    salient_energy_fraction = (
                        channel_energy[
                            salient_indices
                        ].sum()
                        /
                        total_energy
                    ).item()

                    random_energy_fraction = (
                        channel_energy[
                            random_indices
                        ].sum()
                        /
                        total_energy
                    ).item()

                    if full_attn_gain > 1e-12:
                        normalized_salient_recovery = (
                            delta_q_salient
                            /
                            full_attn_gain
                        )

                        normalized_random_recovery = (
                            delta_q_random
                            /
                            full_attn_gain
                        )

                    else:
                        normalized_salient_recovery = None
                        normalized_random_recovery = None

                    recovery_analysis_records.append({
                        "step": int(i),

                        "layer": int(
                            target_layer
                        ),

                        "fraction": float(
                            fraction
                        ),

                        "k_channels": int(
                            k_channels
                        ),

                        "hidden_dim": int(
                            hidden_dim
                        ),

                        "q_baseline": float(
                            q_baseline
                        ),

                        "q_full_attn_restore": float(
                            q_full_attn_restore
                        ),

                        "full_attn_gain": float(
                            full_attn_gain
                        ),

                        "q_salient": float(
                            q_salient
                        ),

                        "q_random": float(
                            q_random
                        ),

                        "delta_q_salient": float(
                            delta_q_salient
                        ),

                        "delta_q_random": float(
                            delta_q_random
                        ),

                        "salient_advantage": float(
                            salient_advantage
                        ),

                        "salient_energy_fraction": float(
                            salient_energy_fraction
                        ),

                        "random_energy_fraction": float(
                            random_energy_fraction
                        ),

                        "normalized_salient_recovery": (
                            None
                            if normalized_salient_recovery
                            is None
                            else float(
                                normalized_salient_recovery
                            )
                        ),

                        "normalized_random_recovery": (
                            None
                            if normalized_random_recovery
                            is None
                            else float(
                                normalized_random_recovery
                            )
                        ),
                    })

                    del salient_hidden
                    del random_hidden

                del baseline_hidden
                del full_hidden

                torch.cuda.empty_cache()

            # ============================================================


            # ============================================================
            # Exp73: within-block operator causal analysis
            #
            # q_model:
            #   target layer has four W2 operators.
            #
            # Counterfactual variants:
            #   - attn_out only W2
            #   - ff_proj only W2
            #   - up_proj only W2
            #   - ff_out only W2
            #   - all four W2
            #
            # Other W2 operators are replaced by their FP16 counterpart
            # during the q_model forward.
            # ============================================================
            if (
                op_analysis
                and (i in op_analysis_steps)
                and (fp_model is not None)
                and (q_model is not None)
            ):
                if cfg_scale != 0.0:
                    raise RuntimeError(
                        "op_analysis currently requires cfg_scale == 0"
                    )

                target_layer = op_target_layer

                # FP16 target-block output was already captured
                # during the baseline FP16 forward.
                if target_layer >= len(layer_outputs):
                    raise RuntimeError(
                        f"Target layer {target_layer} not found in "
                        f"layer_outputs; len={len(layer_outputs)}"
                    )

                fp_hidden = layer_outputs[target_layer]

                # --------------------------------------------------------
                # Locate q_model transformer blocks.
                # --------------------------------------------------------
                if (
                    hasattr(q_model, "model")
                    and hasattr(q_model.model, "transformer")
                ):
                    q_blocks = q_model.model.transformer.blocks

                elif (
                    hasattr(q_model, "model")
                    and hasattr(q_model.model, "model")
                    and hasattr(
                        q_model.model.model,
                        "transformer",
                    )
                ):
                    q_blocks = (
                        q_model.model.model.transformer.blocks
                    )

                else:
                    raise AttributeError(
                        "Cannot find transformer blocks in q_model "
                        f"of type {type(q_model)}"
                    )

                fp_block = blocks[target_layer]
                q_block = q_blocks[target_layer]

                # --------------------------------------------------------
                # Get the device used by one FP16 operator.
                # --------------------------------------------------------
                def op_module_device(module):
                    for p in module.parameters():
                        return p.device

                    for b in module.buffers():
                        return b.device

                    return None

                # --------------------------------------------------------
                # Forward hook:
                #
                # The q_model operator executes normally, but its output
                # is replaced by the output of the FP16 counterpart using
                # the SAME current counterfactual input.
                # --------------------------------------------------------
                def make_fp_override_hook(fp_module):
                    def hook(q_module, inputs, output):
                        if len(inputs) == 0:
                            raise RuntimeError(
                                "Operator override received no input"
                            )

                        x_in = inputs[0]

                        fp_device = op_module_device(
                            fp_module
                        )

                        if fp_device is None:
                            fp_device = x_in.device

                        x_fp = x_in.to(
                            fp_device
                        )

                        y_fp = fp_module(
                            x_fp
                        )

                        if not torch.is_tensor(output):
                            raise RuntimeError(
                                "Expected tensor output from "
                                "quantized operator, got "
                                f"{type(output)}"
                            )

                        y_fp = y_fp.to(
                            device=output.device,
                            dtype=output.dtype,
                        )

                        return y_fp

                    return hook

                # --------------------------------------------------------
                # Run one counterfactual variant.
                # --------------------------------------------------------
                def run_op_variant(active_module):
                    override_handles = []
                    captured = []

                    def capture_block_output(
                        module,
                        inputs,
                        output,
                    ):
                        if isinstance(output, tuple):
                            out = output[0]
                        else:
                            out = output

                        captured.append(
                            out.detach()
                        )

                        return output

                    # active_module stays W2.
                    # The other three operators are restored to FP16.
                    if active_module != "all_four":
                        for module_name in op_modules:
                            if module_name == active_module:
                                continue

                            if not hasattr(
                                q_block,
                                module_name,
                            ):
                                raise AttributeError(
                                    f"q_block has no {module_name}"
                                )

                            if not hasattr(
                                fp_block,
                                module_name,
                            ):
                                raise AttributeError(
                                    f"fp_block has no {module_name}"
                                )

                            q_module = getattr(
                                q_block,
                                module_name,
                            )

                            fp_module = getattr(
                                fp_block,
                                module_name,
                            )

                            handle = (
                                q_module.register_forward_hook(
                                    make_fp_override_hook(
                                        fp_module
                                    )
                                )
                            )

                            override_handles.append(
                                handle
                            )

                    block_handle = (
                        q_block.register_forward_hook(
                            capture_block_output
                        )
                    )

                    try:
                        q_x_input = x.to(
                            q_model.device
                        )

                        q_out = q_model(
                            q_x_input
                        )

                        _ = q_out.logits

                    finally:
                        block_handle.remove()

                        for handle in override_handles:
                            handle.remove()

                    if len(captured) != 1:
                        raise RuntimeError(
                            f"Variant {active_module}: expected "
                            "one target-block output, got "
                            f"{len(captured)}"
                        )

                    result = captured[0]

                    del q_out

                    return result

                # --------------------------------------------------------
                # Run four single-operator variants + normal all-W2.
                # --------------------------------------------------------
                variant_names = [
                    "attn_out",
                    "ff_proj",
                    "up_proj",
                    "ff_out",
                    "all_four",
                ]

                variant_hidden = {}

                for variant_name in variant_names:
                    variant_hidden[
                        variant_name
                    ] = run_op_variant(
                        variant_name
                    )

                # --------------------------------------------------------
                # FP16 block output and current masked positions.
                # --------------------------------------------------------
                fp_cpu = (
                    fp_hidden
                    .detach()
                    .float()
                    .cpu()
                )

                mask_cpu = (
                    mask_index
                    .detach()
                    .cpu()
                )

                fp_masked = fp_cpu[
                    mask_cpu
                ]

                variant_metrics = {}
                error_vectors = {}

                # --------------------------------------------------------
                # Block-output distortion for each variant.
                # --------------------------------------------------------
                for variant_name in variant_names:
                    h_cpu = (
                        variant_hidden[
                            variant_name
                        ]
                        .detach()
                        .float()
                        .cpu()
                    )

                    diff_all = (
                        h_cpu - fp_cpu
                    )

                    h_masked = h_cpu[
                        mask_cpu
                    ]

                    diff_masked = (
                        h_masked
                        - fp_masked
                    )

                    rel_all = (
                        diff_all.norm()
                        /
                        (
                            fp_cpu.norm()
                            + 1e-12
                        )
                    ).item()

                    rel_masked = (
                        diff_masked.norm()
                        /
                        (
                            fp_masked.norm()
                            + 1e-12
                        )
                    ).item()

                    variant_metrics[
                        variant_name
                    ] = {
                        "relative_l2_all": float(
                            rel_all
                        ),
                        "relative_l2_masked": float(
                            rel_masked
                        ),
                    }

                    error_vectors[
                        variant_name
                    ] = (
                        diff_masked.reshape(-1)
                    )

                # --------------------------------------------------------
                # Pairwise error-vector cosine.
                # --------------------------------------------------------
                single_names = [
                    "attn_out",
                    "ff_proj",
                    "up_proj",
                    "ff_out",
                ]

                pairwise_cosine = {}

                for a_idx in range(
                    len(single_names)
                ):
                    for b_idx in range(
                        a_idx + 1,
                        len(single_names),
                    ):
                        a = single_names[a_idx]
                        b = single_names[b_idx]

                        ea = error_vectors[a]
                        eb = error_vectors[b]

                        cosine = (
                            torch.dot(
                                ea,
                                eb,
                            )
                            /
                            (
                                ea.norm()
                                * eb.norm()
                                + 1e-12
                            )
                        ).item()

                        pairwise_cosine[
                            f"{a}__{b}"
                        ] = float(
                            cosine
                        )

                # --------------------------------------------------------
                # Interaction:
                #
                # compare actual all-W2 block error against the vector sum
                # of four single-operator causal errors.
                # --------------------------------------------------------
                e_sum = (
                    error_vectors["attn_out"]
                    + error_vectors["ff_proj"]
                    + error_vectors["up_proj"]
                    + error_vectors["ff_out"]
                )

                e_all = error_vectors[
                    "all_four"
                ]

                fp_norm = (
                    fp_masked
                    .reshape(-1)
                    .norm()
                )

                sum_single_relative_l2 = (
                    e_sum.norm()
                    /
                    (
                        fp_norm
                        + 1e-12
                    )
                ).item()

                all_relative_l2 = (
                    e_all.norm()
                    /
                    (
                        fp_norm
                        + 1e-12
                    )
                ).item()

                interaction_residual = (
                    e_all - e_sum
                )

                interaction_relative_l2 = (
                    interaction_residual.norm()
                    /
                    (
                        fp_norm
                        + 1e-12
                    )
                ).item()

                interaction_over_all = (
                    interaction_residual.norm()
                    /
                    (
                        e_all.norm()
                        + 1e-12
                    )
                ).item()

                sum_vs_all_cosine = (
                    torch.dot(
                        e_sum,
                        e_all,
                    )
                    /
                    (
                        e_sum.norm()
                        * e_all.norm()
                        + 1e-12
                    )
                ).item()

                op_analysis_records.append({
                    "step": int(i),
                    "layer": int(
                        target_layer
                    ),
                    "mask_ratio": float(
                        mask_ratio
                    ),
                    "variants": variant_metrics,
                    "pairwise_cosine": pairwise_cosine,
                    "sum_single_relative_l2_masked": float(
                        sum_single_relative_l2
                    ),
                    "all_four_relative_l2_masked": float(
                        all_relative_l2
                    ),
                    "interaction_relative_l2_masked": float(
                        interaction_relative_l2
                    ),
                    "interaction_over_all": float(
                        interaction_over_all
                    ),
                    "sum_vs_all_cosine": float(
                        sum_vs_all_cosine
                    ),
                })

                del variant_hidden
                del error_vectors
                del fp_cpu
                del fp_masked
                del e_sum
                del e_all
                del interaction_residual

                torch.cuda.empty_cache()
            # ============================================================

            # ============================================================
            # Exp62 pair analysis:
            #
            # Current "logits" + "layer_outputs" are FP16 because
            # FP16 drives the trajectory.
            #
            # Run q_model on exactly the same x_t and compare:
            #
            #   1) local hidden-state error at target layer
            #   2) final masked-token logit error
            #   3) masked-token KL
            #   4) masked top-1 flip rate
            #   5) frontier token / position change
            # ============================================================
            if (
                pair_analysis
                and (i in pair_analysis_steps)
                and (fp_model is not None)
                and (q_model is not None)
            ):
                if cfg_scale != 0.0:
                    raise RuntimeError(
                        "pair_analysis currently requires cfg_scale == 0"
                    )

                target_layer = pair_analysis_target_layer

                # --------------------------------------------------------
                # FP16 target-layer output is already captured by
                # existing hooks in layer_outputs.
                # --------------------------------------------------------
                if target_layer >= len(layer_outputs):
                    raise RuntimeError(
                        f"Target layer {target_layer} not found in "
                        f"FP16 layer_outputs; len={len(layer_outputs)}"
                    )

                fp_hidden = layer_outputs[target_layer]

                # --------------------------------------------------------
                # Locate q_model transformer blocks.
                # --------------------------------------------------------
                if (
                    hasattr(q_model, "model")
                    and hasattr(q_model.model, "transformer")
                ):
                    q_blocks = q_model.model.transformer.blocks

                elif (
                    hasattr(q_model, "model")
                    and hasattr(q_model.model, "model")
                    and hasattr(q_model.model.model, "transformer")
                ):
                    q_blocks = q_model.model.model.transformer.blocks

                else:
                    raise AttributeError(
                        "Cannot find transformer blocks in q_model "
                        f"of type {type(q_model)}"
                    )

                q_target_output = []

                def q_target_hook(module, inputs, output):
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output

                    q_target_output.append(out.detach())
                    return output

                q_handle = q_blocks[
                    target_layer
                ].register_forward_hook(q_target_hook)

                # --------------------------------------------------------
                # Run single-layer-quantized model on SAME x_t.
                # q_model does NOT update x.
                # --------------------------------------------------------
                q_x_input = x.to(q_model.device)

                q_logits = q_model(q_x_input).logits

                q_handle.remove()

                if len(q_target_output) != 1:
                    raise RuntimeError(
                        f"Expected one q target-layer output, "
                        f"got {len(q_target_output)}"
                    )

                q_hidden = q_target_output[0]

                # --------------------------------------------------------
                # A. Local hidden-state quantization error
                #
                # Compute both:
                #   - all tokens
                #   - currently masked tokens only
                # --------------------------------------------------------
                fp_hidden_cpu = (
                    fp_hidden
                    .detach()
                    .float()
                    .cpu()
                )

                q_hidden_cpu = (
                    q_hidden
                    .detach()
                    .float()
                    .cpu()
                )

                local_relative_l2_all = (
                    (q_hidden_cpu - fp_hidden_cpu).norm()
                    /
                    (fp_hidden_cpu.norm() + 1e-12)
                ).item()

                fp_mask_hidden = (
                    fp_hidden[
                        mask_index.to(fp_hidden.device)
                    ]
                    .detach()
                    .float()
                    .cpu()
                )

                q_mask_hidden = (
                    q_hidden[
                        mask_index.to(q_hidden.device)
                    ]
                    .detach()
                    .float()
                    .cpu()
                )

                if fp_mask_hidden.numel() > 0:
                    local_relative_l2_masked = (
                        (q_mask_hidden - fp_mask_hidden).norm()
                        /
                        (fp_mask_hidden.norm() + 1e-12)
                    ).item()

                    local_cosine_masked = (
                        F.cosine_similarity(
                            fp_mask_hidden,
                            q_mask_hidden,
                            dim=-1,
                        )
                        .mean()
                        .item()
                    )
                else:
                    local_relative_l2_masked = 0.0
                    local_cosine_masked = 1.0

                # --------------------------------------------------------
                # B. Final output difference on currently masked tokens
                # --------------------------------------------------------
                fp_mask = mask_index.to(logits.device)
                q_mask = mask_index.to(q_logits.device)

                fp_masked_logits = (
                    logits[fp_mask]
                    .detach()
                    .float()
                    .cpu()
                )

                q_masked_logits = (
                    q_logits[q_mask]
                    .detach()
                    .float()
                    .cpu()
                )

                if fp_masked_logits.shape != q_masked_logits.shape:
                    raise RuntimeError(
                        "FP16 / quant masked logits shape mismatch: "
                        f"{fp_masked_logits.shape} vs "
                        f"{q_masked_logits.shape}"
                    )

                if fp_masked_logits.numel() > 0:
                    masked_logit_relative_l2 = (
                        (q_masked_logits - fp_masked_logits).norm()
                        /
                        (fp_masked_logits.norm() + 1e-12)
                    ).item()

                    fp_log_prob = F.log_softmax(
                        fp_masked_logits,
                        dim=-1,
                    )

                    q_log_prob = F.log_softmax(
                        q_masked_logits,
                        dim=-1,
                    )

                    fp_prob = fp_log_prob.exp()
                    q_prob = q_log_prob.exp()

                    # KL(FP16 || Quant)
                    masked_kl = F.kl_div(
                        q_log_prob,
                        fp_prob,
                        reduction="batchmean",
                    ).item()

                    fp_top1 = fp_masked_logits.argmax(dim=-1)
                    q_top1 = q_masked_logits.argmax(dim=-1)

                    masked_top1_flip_rate = (
                        fp_top1 != q_top1
                    ).float().mean().item()

                    fp_conf = fp_prob.max(dim=-1).values
                    q_conf = q_prob.max(dim=-1).values

                    # ----------------------------------------------------
                    # C. DLM-specific frontier analysis
                    #
                    # At this step, which token position would each model
                    # choose to commit, and would the token identity differ?
                    # ----------------------------------------------------
                    k_transfer = int(
                        num_transfer_tokens[0, i].item()
                    )

                    if k_transfer > 0:
                        fp_frontier = torch.topk(
                            fp_conf,
                            k=k_transfer,
                        ).indices

                        q_frontier = torch.topk(
                            q_conf,
                            k=k_transfer,
                        ).indices

                        fp_frontier_set = set(
                            fp_frontier.tolist()
                        )
                        q_frontier_set = set(
                            q_frontier.tolist()
                        )

                        frontier_position_overlap = (
                            len(
                                fp_frontier_set
                                & q_frontier_set
                            )
                            / float(k_transfer)
                        )

                        frontier_token_flip_rate = (
                            fp_top1[fp_frontier]
                            !=
                            q_top1[fp_frontier]
                        ).float().mean().item()

                    else:
                        frontier_position_overlap = 1.0
                        frontier_token_flip_rate = 0.0

                else:
                    masked_logit_relative_l2 = 0.0
                    masked_kl = 0.0
                    masked_top1_flip_rate = 0.0
                    frontier_position_overlap = 1.0
                    frontier_token_flip_rate = 0.0

                pair_analysis_records.append({
                    "step": int(i),
                    "target_layer": int(target_layer),
                    "mask_ratio": float(mask_ratio),

                    "local_relative_l2_all": float(
                        local_relative_l2_all
                    ),
                    "local_relative_l2_masked": float(
                        local_relative_l2_masked
                    ),
                    "local_cosine_masked": float(
                        local_cosine_masked
                    ),

                    "masked_logit_relative_l2": float(
                        masked_logit_relative_l2
                    ),
                    "masked_kl_fp_to_quant": float(
                        masked_kl
                    ),
                    "masked_top1_flip_rate": float(
                        masked_top1_flip_rate
                    ),

                    "frontier_position_overlap": float(
                        frontier_position_overlap
                    ),
                    "frontier_token_flip_rate": float(
                        frontier_token_flip_rate
                    ),
                })

                del q_logits
                del q_hidden
                del q_target_output
                del fp_hidden_cpu
                del q_hidden_cpu
                del fp_mask_hidden
                del q_mask_hidden
                del fp_masked_logits
                del q_masked_logits

                torch.cuda.empty_cache()
            # ============================================================

            layer_norms = []
            layer_vars = []
            layer_maxs = []
            layer_p99s = []
            layer_p999s = []
            layer_p9999s = []
            layer_top001_means = []

            for h in layer_outputs:
                h = h.float()
                abs_h = h.abs().flatten()

                layer_norms.append(h.norm(dim=-1).mean().item())
                layer_vars.append(h.var().item())
                layer_maxs.append(abs_h.max().item())
                layer_p99s.append(torch.quantile(abs_h, 0.99).item())
                layer_p999s.append(torch.quantile(abs_h, 0.999).item())
                layer_p9999s.append(torch.quantile(abs_h, 0.9999).item())

                k = max(1, int(abs_h.numel() * 0.001))
                layer_top001_means.append(torch.topk(abs_h, k).values.mean().item())

            step_layer_hidden_norm.append(layer_norms)
            step_layer_hidden_var.append(layer_vars)
            step_layer_hidden_max.append(layer_maxs)
            step_layer_hidden_p99.append(layer_p99s)
            step_layer_hidden_p999.append(layer_p999s)
            step_layer_hidden_p9999.append(layer_p9999s)
            step_layer_hidden_top001_mean.append(layer_top001_means)

            for h in handles:
                h.remove()

            # ============================================================
            # Exp63: standardized FP16 perturbation probes
            #
            # Baseline FP16 hooks have already been removed.
            # Each probe:
            #   1. uses exactly the same x_t
            #   2. injects 1% per-token relative hidden perturbation
            #   3. measures downstream output change
            #   4. never updates the denoising trajectory
            # ============================================================
            if (
                perturb_analysis
                and (i in perturb_steps)
                and (fp_model is not None)
            ):
                if cfg_scale != 0.0:
                    raise RuntimeError(
                        "perturb_analysis currently requires cfg_scale == 0"
                    )

                # --------------------------------------------------------
                # Baseline masked-token prediction from FP16.
                # --------------------------------------------------------
                baseline_mask = mask_index.to(logits.device)

                baseline_masked_logits = (
                    logits[baseline_mask]
                    .detach()
                    .float()
                )

                if baseline_masked_logits.numel() == 0:
                    raise RuntimeError(
                        f"No masked tokens at perturbation step {i}"
                    )

                baseline_log_prob = F.log_softmax(
                    baseline_masked_logits,
                    dim=-1,
                )

                baseline_prob = baseline_log_prob.exp()

                baseline_top1 = baseline_masked_logits.argmax(
                    dim=-1
                )

                baseline_conf = baseline_prob.max(
                    dim=-1
                ).values

                # Number of tokens FP16 would commit at this step.
                k_transfer = int(
                    num_transfer_tokens[0, i].item()
                )

                if k_transfer > 0:
                    baseline_frontier = torch.topk(
                        baseline_conf,
                        k=k_transfer,
                    ).indices
                else:
                    baseline_frontier = None

                # Prompt-dependent deterministic seed offset.
                prompt_seed_offset = int(
                    prompt.detach().long().sum().item()
                ) % 1000003

                for target_layer in perturb_layers:

                    if target_layer >= len(blocks):
                        raise RuntimeError(
                            f"Invalid perturbation layer "
                            f"{target_layer}; num layers={len(blocks)}"
                        )

                    for noise_seed_base in perturb_noise_seeds:

                        actual_seed = (
                            int(noise_seed_base)
                            + 1009 * int(target_layer)
                            + 9176 * int(i)
                            + prompt_seed_offset
                        )

                        perturb_state = {}

                        def standardized_perturb_hook(
                            module,
                            inputs,
                            output,
                            _seed=actual_seed,
                        ):
                            if isinstance(output, tuple):
                                out = output[0]
                            else:
                                out = output

                            out_fp32 = out.float()

                            # --------------------------------------------
                            # Deterministic isotropic Gaussian direction.
                            # --------------------------------------------
                            generator = torch.Generator(
                                device=out.device
                            )
                            generator.manual_seed(_seed)

                            noise = torch.randn(
                                out_fp32.shape,
                                device=out.device,
                                dtype=torch.float32,
                                generator=generator,
                            )

                            # Per-token normalization.
                            noise_norm = noise.norm(
                                dim=-1,
                                keepdim=True,
                            ).clamp_min(1e-12)

                            hidden_norm = out_fp32.norm(
                                dim=-1,
                                keepdim=True,
                            )

                            delta = (
                                perturb_alpha
                                * hidden_norm
                                * noise
                                / noise_norm
                            )

                            perturbed_fp32 = out_fp32 + delta

                            # Match the model's original activation dtype.
                            perturbed = perturbed_fp32.to(
                                dtype=out.dtype
                            )

                            # Measure ACTUAL perturbation after dtype cast.
                            actual_delta = (
                                perturbed.float()
                                - out_fp32
                            )

                            per_token_relative = (
                                actual_delta.norm(dim=-1)
                                /
                                (
                                    out_fp32.norm(dim=-1)
                                    + 1e-12
                                )
                            )

                            achieved_relative_mean = (
                                per_token_relative
                                .mean()
                                .item()
                            )

                            achieved_relative_global = (
                                actual_delta.norm()
                                /
                                (
                                    out_fp32.norm()
                                    + 1e-12
                                )
                            ).item()

                            perturb_state[
                                "achieved_relative_mean"
                            ] = achieved_relative_mean

                            perturb_state[
                                "achieved_relative_global"
                            ] = achieved_relative_global

                            if isinstance(output, tuple):
                                return (
                                    perturbed,
                                    *output[1:],
                                )

                            return perturbed

                        perturb_handle = blocks[
                            target_layer
                        ].register_forward_hook(
                            standardized_perturb_hook
                        )

                        # --------------------------------------------
                        # Same FP16 model, SAME x_t.
                        # Only hidden state at target layer is altered.
                        # --------------------------------------------
                        perturbed_logits = fp_model(
                            x_input
                        ).logits

                        perturb_handle.remove()

                        if (
                            "achieved_relative_global"
                            not in perturb_state
                        ):
                            raise RuntimeError(
                                "Perturbation hook did not execute"
                            )

                        perturbed_mask = mask_index.to(
                            perturbed_logits.device
                        )

                        perturbed_masked_logits = (
                            perturbed_logits[
                                perturbed_mask
                            ]
                            .float()
                        )

                        # --------------------------------------------
                        # Output relative L2.
                        # --------------------------------------------
                        masked_logit_relative_l2 = (
                            (
                                perturbed_masked_logits
                                - baseline_masked_logits.to(
                                    perturbed_masked_logits.device
                                )
                            ).norm()
                            /
                            (
                                baseline_masked_logits.to(
                                    perturbed_masked_logits.device
                                ).norm()
                                + 1e-12
                            )
                        ).item()

                        achieved_input_relative = float(
                            perturb_state[
                                "achieved_relative_global"
                            ]
                        )

                        transfer_gain = (
                            masked_logit_relative_l2
                            /
                            (
                                achieved_input_relative
                                + 1e-12
                            )
                        )

                        # --------------------------------------------
                        # KL(FP16 || Perturbed).
                        # --------------------------------------------
                        perturbed_log_prob = F.log_softmax(
                            perturbed_masked_logits,
                            dim=-1,
                        )

                        baseline_prob_device = (
                            baseline_prob.to(
                                perturbed_log_prob.device
                            )
                        )

                        masked_kl = F.kl_div(
                            perturbed_log_prob,
                            baseline_prob_device,
                            reduction="batchmean",
                        ).item()

                        # --------------------------------------------
                        # Masked-token top-1 changes.
                        # --------------------------------------------
                        perturbed_top1 = (
                            perturbed_masked_logits.argmax(
                                dim=-1
                            )
                        )

                        baseline_top1_device = (
                            baseline_top1.to(
                                perturbed_top1.device
                            )
                        )

                        masked_top1_flip_rate = (
                            baseline_top1_device
                            != perturbed_top1
                        ).float().mean().item()

                        # --------------------------------------------
                        # Frontier effect.
                        # --------------------------------------------
                        perturbed_prob = (
                            perturbed_log_prob.exp()
                        )

                        perturbed_conf = perturbed_prob.max(
                            dim=-1
                        ).values

                        if k_transfer > 0:
                            perturbed_frontier = torch.topk(
                                perturbed_conf,
                                k=k_transfer,
                            ).indices

                            baseline_frontier_device = (
                                baseline_frontier.to(
                                    perturbed_frontier.device
                                )
                            )

                            baseline_frontier_set = set(
                                baseline_frontier_device
                                .detach()
                                .cpu()
                                .tolist()
                            )

                            perturbed_frontier_set = set(
                                perturbed_frontier
                                .detach()
                                .cpu()
                                .tolist()
                            )

                            frontier_position_overlap = (
                                len(
                                    baseline_frontier_set
                                    &
                                    perturbed_frontier_set
                                )
                                /
                                float(k_transfer)
                            )

                            frontier_token_flip_rate = (
                                baseline_top1_device[
                                    baseline_frontier_device
                                ]
                                !=
                                perturbed_top1[
                                    baseline_frontier_device
                                ]
                            ).float().mean().item()

                        else:
                            frontier_position_overlap = 1.0
                            frontier_token_flip_rate = 0.0

                        perturb_analysis_records.append({
                            "step": int(i),
                            "target_layer": int(
                                target_layer
                            ),
                            "alpha": float(
                                perturb_alpha
                            ),
                            "noise_seed_base": int(
                                noise_seed_base
                            ),
                            "actual_seed": int(
                                actual_seed
                            ),
                            "mask_ratio": float(
                                mask_ratio
                            ),

                            "achieved_relative_mean": float(
                                perturb_state[
                                    "achieved_relative_mean"
                                ]
                            ),
                            "achieved_relative_global": float(
                                achieved_input_relative
                            ),

                            "masked_logit_relative_l2": float(
                                masked_logit_relative_l2
                            ),
                            "transfer_gain": float(
                                transfer_gain
                            ),
                            "masked_kl_fp_to_perturbed": float(
                                masked_kl
                            ),
                            "masked_top1_flip_rate": float(
                                masked_top1_flip_rate
                            ),
                            "frontier_position_overlap": float(
                                frontier_position_overlap
                            ),
                            "frontier_token_flip_rate": float(
                                frontier_token_flip_rate
                            ),
                        })

                        del perturbed_logits
                        del perturbed_masked_logits
                        del perturbed_log_prob
                        del perturbed_prob

                del baseline_masked_logits
                del baseline_log_prob
                del baseline_prob

                torch.cuda.empty_cache()
            # ============================================================

            # ============================================================
            # Exp68: weight-activation decomposition
            #
            # For each selected module:
            #
            #   D_real = ||Q(X_real) - FP(X_real)|| / ||X_real||
            #
            #   D_iso  = same quantity on isotropic inputs whose
            #            per-token norms equal those of X_real
            #
            #   R_act  = D_real / D_iso
            #
            # D_iso approximates weight/operator-side difficulty.
            # R_act measures how strongly real DLM activations excite
            # the quantization-error operator.
            # ============================================================
            if (
                c_analysis
                and (i in c_analysis_steps)
                and (fp_model is not None)
                and (q_model is not None)
            ):
                if cfg_scale != 0.0:
                    raise RuntimeError(
                        "c_analysis currently requires cfg_scale == 0"
                    )

                # --------------------------------------------------------
                # Locate quantized transformer blocks.
                # --------------------------------------------------------
                if (
                    hasattr(q_model, "model")
                    and hasattr(q_model.model, "transformer")
                ):
                    c_q_blocks = q_model.model.transformer.blocks

                elif (
                    hasattr(q_model, "model")
                    and hasattr(q_model.model, "model")
                    and hasattr(
                        q_model.model.model,
                        "transformer"
                    )
                ):
                    c_q_blocks = (
                        q_model.model.model.transformer.blocks
                    )

                else:
                    raise AttributeError(
                        "Cannot find transformer blocks in q_model "
                        f"of type {type(q_model)}"
                    )

                c_fp_block = blocks[c_target_layer]
                c_q_block = c_q_blocks[c_target_layer]

                # Stable module-specific seed offsets.
                module_seed_offsets = {
                    "q_proj": 11,
                    "attn_out": 23,
                    "ff_proj": 37,
                    "up_proj": 53,
                    "ff_out": 71,
                }

                prompt_seed_offset = (
                    int(
                        prompt.detach()
                        .long()
                        .sum()
                        .item()
                    )
                    % 1000003
                )

                def get_module_device(module, fallback):
                    for param in module.parameters():
                        return param.device

                    for buf in module.buffers():
                        return buf.device

                    return fallback

                for module_name in c_analysis_modules:

                    if module_name not in c_real_inputs:
                        continue

                    if not hasattr(c_fp_block, module_name):
                        continue

                    if not hasattr(c_q_block, module_name):
                        continue

                    fp_module = getattr(
                        c_fp_block,
                        module_name,
                    )

                    q_module = getattr(
                        c_q_block,
                        module_name,
                    )

                    x_real = c_real_inputs[
                        module_name
                    ]

                    # ----------------------------------------------------
                    # Restrict analysis to currently masked tokens.
                    #
                    # x_real shape is expected to be:
                    # [batch, seq, hidden]
                    #
                    # After boolean indexing:
                    # [num_masked_tokens, hidden]
                    # ----------------------------------------------------
                    module_mask = mask_index.to(
                        x_real.device
                    )

                    x_real_masked = x_real[
                        module_mask
                    ]

                    if x_real_masked.numel() == 0:
                        continue

                    # Work in FP32 for statistics.
                    x_stats = (
                        x_real_masked
                        .detach()
                        .float()
                    )

                    # ----------------------------------------------------
                    # Activation statistics.
                    # ----------------------------------------------------
                    input_rms = torch.sqrt(
                        torch.mean(x_stats * x_stats)
                    ).item()

                    abs_flat = x_stats.abs().flatten()

                    p99 = torch.quantile(
                        abs_flat,
                        0.99,
                    ).item()

                    p999 = torch.quantile(
                        abs_flat,
                        0.999,
                    ).item()

                    input_max = abs_flat.max().item()

                    p99_over_rms = (
                        p99 / (input_rms + 1e-12)
                    )

                    p999_over_rms = (
                        p999 / (input_rms + 1e-12)
                    )

                    max_over_rms = (
                        input_max
                        / (input_rms + 1e-12)
                    )

                    # ----------------------------------------------------
                    # Determine module devices.
                    # ----------------------------------------------------
                    fp_device = get_module_device(
                        fp_module,
                        x_real_masked.device,
                    )

                    q_device = get_module_device(
                        q_module,
                        q_model.device,
                    )

                    x_fp = x_real_masked.to(
                        fp_device
                    )

                    x_q = x_real_masked.to(
                        q_device
                    )

                    # ----------------------------------------------------
                    # REAL activation operator distortion.
                    # ----------------------------------------------------
                    y_fp_real = fp_module(x_fp)

                    y_q_real = q_module(x_q)

                    y_fp_real_cpu = (
                        y_fp_real
                        .detach()
                        .float()
                        .cpu()
                    )

                    y_q_real_cpu = (
                        y_q_real
                        .detach()
                        .float()
                        .cpu()
                    )

                    x_real_cpu = (
                        x_real_masked
                        .detach()
                        .float()
                        .cpu()
                    )

                    real_diff = (
                        y_q_real_cpu
                        - y_fp_real_cpu
                    )

                    real_error_per_input = (
                        real_diff.norm()
                        /
                        (
                            x_real_cpu.norm()
                            + 1e-12
                        )
                    ).item()

                    real_output_relative_l2 = (
                        real_diff.norm()
                        /
                        (
                            y_fp_real_cpu.norm()
                            + 1e-12
                        )
                    ).item()

                    # ----------------------------------------------------
                    # ISOTROPIC controls.
                    #
                    # Preserve each token's L2 norm exactly,
                    # randomize only direction.
                    # ----------------------------------------------------
                    iso_error_per_input_values = []
                    iso_output_relative_values = []

                    token_norm = x_stats.norm(
                        dim=-1,
                        keepdim=True,
                    )

                    for iso_seed_base in c_iso_seeds:

                        actual_seed = (
                            int(iso_seed_base)
                            + 1009 * int(c_target_layer)
                            + 9176 * int(i)
                            + module_seed_offsets[
                                module_name
                            ]
                            + prompt_seed_offset
                        )

                        generator = torch.Generator(
                            device=x_stats.device
                        )

                        generator.manual_seed(
                            actual_seed
                        )

                        noise = torch.randn(
                            x_stats.shape,
                            device=x_stats.device,
                            dtype=torch.float32,
                            generator=generator,
                        )

                        noise_norm = noise.norm(
                            dim=-1,
                            keepdim=True,
                        ).clamp_min(1e-12)

                        x_iso_fp32 = (
                            token_norm
                            * noise
                            / noise_norm
                        )

                        # Match original activation dtype.
                        x_iso = x_iso_fp32.to(
                            dtype=x_real_masked.dtype
                        )

                        # Sanity check: same per-token norms.
                        real_norm_mean = (
                            x_stats.norm(
                                dim=-1
                            )
                            .mean()
                            .item()
                        )

                        iso_norm_mean = (
                            x_iso.float()
                            .norm(
                                dim=-1
                            )
                            .mean()
                            .item()
                        )

                        x_iso_fp = x_iso.to(
                            fp_device
                        )

                        x_iso_q = x_iso.to(
                            q_device
                        )

                        y_fp_iso = fp_module(
                            x_iso_fp
                        )

                        y_q_iso = q_module(
                            x_iso_q
                        )

                        y_fp_iso_cpu = (
                            y_fp_iso
                            .detach()
                            .float()
                            .cpu()
                        )

                        y_q_iso_cpu = (
                            y_q_iso
                            .detach()
                            .float()
                            .cpu()
                        )

                        x_iso_cpu = (
                            x_iso
                            .detach()
                            .float()
                            .cpu()
                        )

                        iso_diff = (
                            y_q_iso_cpu
                            - y_fp_iso_cpu
                        )

                        iso_error_per_input = (
                            iso_diff.norm()
                            /
                            (
                                x_iso_cpu.norm()
                                + 1e-12
                            )
                        ).item()

                        iso_output_relative_l2 = (
                            iso_diff.norm()
                            /
                            (
                                y_fp_iso_cpu.norm()
                                + 1e-12
                            )
                        ).item()

                        iso_error_per_input_values.append(
                            iso_error_per_input
                        )

                        iso_output_relative_values.append(
                            iso_output_relative_l2
                        )

                        del y_fp_iso
                        del y_q_iso
                        del y_fp_iso_cpu
                        del y_q_iso_cpu
                        del iso_diff
                        del x_iso_fp
                        del x_iso_q
                        del x_iso_cpu

                    iso_error_per_input_mean = float(
                        np.mean(
                            iso_error_per_input_values
                        )
                    )

                    iso_error_per_input_std = float(
                        np.std(
                            iso_error_per_input_values
                        )
                    )

                    iso_output_relative_mean = float(
                        np.mean(
                            iso_output_relative_values
                        )
                    )

                    activation_excitation_ratio = (
                        real_error_per_input
                        /
                        (
                            iso_error_per_input_mean
                            + 1e-12
                        )
                    )

                    c_analysis_records.append({
                        "step": int(i),
                        "layer": int(
                            c_target_layer
                        ),
                        "module": module_name,
                        "mask_ratio": float(
                            mask_ratio
                        ),

                        "input_rms": float(
                            input_rms
                        ),
                        "p99_over_rms": float(
                            p99_over_rms
                        ),
                        "p999_over_rms": float(
                            p999_over_rms
                        ),
                        "max_over_rms": float(
                            max_over_rms
                        ),

                        "real_error_per_input": float(
                            real_error_per_input
                        ),
                        "real_output_relative_l2": float(
                            real_output_relative_l2
                        ),

                        "iso_error_per_input_values": [
                            float(v)
                            for v in iso_error_per_input_values
                        ],
                        "iso_error_per_input_mean": float(
                            iso_error_per_input_mean
                        ),
                        "iso_error_per_input_std": float(
                            iso_error_per_input_std
                        ),

                        "iso_output_relative_values": [
                            float(v)
                            for v in iso_output_relative_values
                        ],
                        "iso_output_relative_mean": float(
                            iso_output_relative_mean
                        ),

                        "activation_excitation_ratio": float(
                            activation_excitation_ratio
                        ),

                        "real_token_norm_mean": float(
                            real_norm_mean
                        ),
                        "iso_token_norm_mean": float(
                            iso_norm_mean
                        ),
                    })

                    del y_fp_real
                    del y_q_real
                    del y_fp_real_cpu
                    del y_q_real_cpu
                    del real_diff
                    del x_real_cpu
                    del x_stats
                    del x_fp
                    del x_q

                del c_real_inputs

                torch.cuda.empty_cache()
            # ============================================================

            # 结果移回原设备
            logits = logits.to(base_device)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l


            # === Step Loss (masked CE) ===
            with torch.no_grad():
                step_mask = (x == mask_id)
                if step_mask.any():
                    step_loss = F.cross_entropy(
                    logits[step_mask],
                    x0[step_mask], 
                    reduction='mean'
                    ).item()
                else:
                    step_loss = 0.0
            step_losses.append(step_loss)
            
            #=========================
            
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # ===== store confidence=====
            valid_conf = confidence[confidence != -np.inf]
            avg_conf = valid_conf.mean().item() if valid_conf.numel() > 0 else 0.0
            step_confidences.append(avg_conf)
            #=========================

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            step_time = time.time() - start_time

            gpu0_alloc = torch.cuda.memory_allocated(0) / 1024**3
            gpu1_alloc = torch.cuda.memory_allocated(1) / 1024**3

            # 修改打印，添加时间
            if i % 10 == 0:
                print(f"[Step {i:3d}/{steps}] GPU0: {gpu0_alloc:5.2f}/{gpu0_total:5.2f} GB | GPU1: {gpu1_alloc:5.2f}/{gpu1_total:5.2f} GB | Time: {step_time:.2f}s | Model: {model_name if (fp_model is not None and q_model is not None) else 'Single'}")

    # ============================================================
    # Save Exp62 pair-analysis metrics.
    #
    # One JSONL line = one MMLU example.
    # Only scalar statistics are saved.
    # ============================================================
    if pair_analysis and pair_analysis_records:
        pair_record = {
            "timestamp": datetime.now().isoformat(),
            "target_layer": int(
                pair_analysis_target_layer
            ),
            "selected_steps": sorted(
                list(pair_analysis_steps)
            ),
            "records": pair_analysis_records,
        }

        with open(
            pair_analysis_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(pair_record)
                + "\n"
            )
    # ============================================================

    # ============================================================
    # Save Exp63 standardized perturbation analysis.
    # One JSONL line corresponds to one MMLU example.
    # ============================================================
    if perturb_analysis and perturb_analysis_records:
        perturb_record = {
            "timestamp": datetime.now().isoformat(),
            "perturb_layers": perturb_layers,
            "perturb_steps": sorted(
                list(perturb_steps)
            ),
            "perturb_alpha": float(
                perturb_alpha
            ),
            "perturb_noise_seeds": (
                perturb_noise_seeds
            ),
            "records": perturb_analysis_records,
        }

        with open(
            perturb_analysis_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(perturb_record)
                + "\n"
            )
    # ============================================================

    # ============================================================
    # Save Exp68 weight-activation decomposition.
    #
    # One JSONL line = one evaluation example.
    # For Exp68:
    # 5 timesteps x 5 modules = 25 records/sample.
    # ============================================================
    if c_analysis and c_analysis_records:
        c_record = {
            "timestamp": datetime.now().isoformat(),
            "target_layer": int(
                c_target_layer
            ),
            "analysis_steps": sorted(
                list(c_analysis_steps)
            ),
            "modules": c_analysis_modules,
            "iso_seeds": c_iso_seeds,
            "records": c_analysis_records,
        }

        with open(
            c_analysis_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(c_record)
                + "\n"
            )
    # ============================================================

    # ============================================================
    # Save Exp104 salient-channel recovery analysis.
    # One JSONL line = one evaluation example.
    # ============================================================
    if recovery_analysis and recovery_analysis_records:
        recovery_record = {
            "timestamp": datetime.now().isoformat(),
            "analysis": (
                "exp104_salient_channel_recovery"
            ),
            "target_layer": int(
                recovery_target_layer
            ),
            "analysis_steps": sorted(
                list(recovery_analysis_steps)
            ),
            "fractions": recovery_fractions,
            "prompt_length": int(
                prompt.shape[1]
            ),
            "num_records": int(
                len(recovery_analysis_records)
            ),
            "records": recovery_analysis_records,
        }

        with open(
            recovery_analysis_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(recovery_record)
                + "\n"
            )
    # ============================================================


    # ============================================================
    # Save Exp103 salient-subspace preservation analysis.
    # One JSONL line = one evaluation example.
    # ============================================================
    if sp_analysis and sp_analysis_records:
        sp_record = {
            "timestamp": datetime.now().isoformat(),
            "analysis": (
                "exp103_salient_subspace_preservation"
            ),
            "target_layer": int(
                sp_target_layer
            ),
            "analysis_steps": sorted(
                list(sp_analysis_steps)
            ),
            "modules": sp_modules,
            "top_fraction": float(
                sp_top_fraction
            ),
            "prompt_length": int(
                prompt.shape[1]
            ),
            "num_records": int(
                len(sp_analysis_records)
            ),
            "records": sp_analysis_records,
        }

        with open(
            sp_analysis_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(sp_record)
                + "\n"
            )
    # ============================================================

    # ============================================================
    # Save the mean real-denoising activation energy vector
    # for this evaluation example.
    #
    # Average over the selected denoising steps.
    # ============================================================
    if (
        sp_analysis
        and channel_energy_sum is not None
        and channel_energy_count > 0
    ):
        mean_channel_energy = (
            channel_energy_sum
            /
            float(
                channel_energy_count
            )
        )

        energy_record = {
            "layer": int(
                sp_target_layer
            ),
            "bits": int(
                sp_bits
            ),
            "module": "attn_out",
            "num_measurements": int(
                channel_energy_count
            ),
            "prompt_length": int(
                prompt.shape[1]
            ),
            "channel_energy": (
                mean_channel_energy
                .float()
                .tolist()
            ),
        }

        with open(
            inference_energy_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    energy_record
                )
                + "\n"
            )
    # ============================================================

    # ============================================================
    # Save Exp73 operator causal analysis.
    # One JSONL line = one evaluation example.
    # ============================================================
    if op_analysis and op_analysis_records:
        op_record = {
            "timestamp": datetime.now().isoformat(),
            "target_layer": int(
                op_target_layer
            ),
            "analysis_steps": sorted(
                list(op_analysis_steps)
            ),
            "modules": op_modules,
            "records": op_analysis_records,
        }

        with open(
            op_analysis_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(op_record)
                + "\n"
            )
    # ============================================================

    # ============================================================
    # Save Exp101 activation saliency analysis.
    # One JSONL line = one evaluation example.
    # ============================================================
    if saliency_analysis and saliency_records:
        saliency_record = {
            "timestamp": datetime.now().isoformat(),

            "analysis": "exp101_activation_saliency",

            "layers": sorted(
                list(saliency_layers)
            ),

            "analysis_steps": sorted(
                list(saliency_steps)
            ),

            "modules": saliency_modules,

            "prompt_length": int(
                prompt.shape[1]
            ),

            "num_records": int(
                len(saliency_records)
            ),

            "records": saliency_records,
        }

        with open(
            saliency_analysis_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(saliency_record)
                + "\n"
            )
    # ============================================================

    # ===== save per-step logs =====
    log_dir = "step_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "denoise_step_metrics.jsonl")
    record = {
        "timestamp": datetime.now().isoformat(),
        "steps": len(step_losses),
        "quant_start_step": quant_start_step,
        "step_losses": step_losses,
        "step_confidences": step_confidences,
        "step_layer_hidden_norm": step_layer_hidden_norm,
        "step_layer_hidden_var": step_layer_hidden_var,
        "step_layer_hidden_max": step_layer_hidden_max,
        "step_layer_hidden_p99": step_layer_hidden_p99,
        "step_layer_hidden_p999": step_layer_hidden_p999,
        "step_layer_hidden_p9999": step_layer_hidden_p9999,
        "step_layer_hidden_top001_mean": step_layer_hidden_top001_mean,
        "step_mask_ratio": step_mask_ratio,
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    # =================================

    if dump_full_act:
        dump_dir = Path(dump_root) / dump_tag
        dump_dir.mkdir(parents=True, exist_ok=True)

        sample_id = len(list(dump_dir.glob("sample_*")))
        sample_dir = dump_dir / f"sample_{sample_id:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # torch.save(dump_full_hidden, sample_dir / "full_hidden_selected.pt")
        # torch.save(dump_linear_inputs, sample_dir / "linear_inputs_selected.pt")

        meta = {
            "sample_id": sample_id,
            "tag": dump_tag,
            "selected_layers": sorted(list(dump_selected_layers)),
            "selected_steps": sorted(list(dump_selected_steps)),
            "linear_modules": sorted(list(dump_linear_modules)),
            "steps": len(step_losses),
        }
        with open(sample_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return x




def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        model=None,
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        
        model_fp=None,
        model_q=None,
        quant_start_step: int = 0,

        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        if model is None or isinstance(model, str):
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        else:
            self.model = model
        self.model.eval()
        
        self.model_fp = model_fp
        self.model_q = model_q
        self.quant_start_step = int(quant_start_step)
        if self.model_fp is not None:
            self.model_fp.eval()
        if self.model_q is not None:
            self.model_q.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            if self.model_fp is not None:
                self.model_fp = self.accelerator.prepare(self.model_fp)
            if self.model_q is not None:
                self.model_q = self.accelerator.prepare(self.model_q)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif model is None or isinstance(model, str):

            # 不要移动 model_fp 和 model_q，保持它们在原来的 GPU 上
            self.model = self.model.to(device)
            #if self.model_fp is not None:
            #    self.model_fp = self.model_fp.to(device)
            #if self.model_q is not None:
            #    self.model_q = self.model_q.to(device)
        else:
            self.device = next(model.parameters()).device
            #if self.model_fp is not None:
            #    self.model_fp = self.model_fp.to(self.device)
            #if self.model_q is not None:
            #    self.model_q = self.model_q.to(self.device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking

        
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index).to(self.device)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index).to(self.device)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]
 
            try:
                generated_answer = generate(self.model, prompt, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                            temperature=0, cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_id,
                                            fp_model=self.model_fp,q_model=self.model_q,quant_start_step=self.quant_start_step)
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"Out of memory error while generating. Please try reducing the batch size or using a smaller model. Batch size: {self.batch_size}")
                generated_answer = torch.tensor([], device=self.device).unsqueeze(0)
                assert self.batch_size == 1, 'Out of memory error should only happen for one request'
            
            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
        return out
    
    def generate(self, *args, **kwargs):
        return generate(self.model, *args, **kwargs)


# if __name__ == "__main__":
#     set_seed(1234)
#     cli_evaluate()
    
