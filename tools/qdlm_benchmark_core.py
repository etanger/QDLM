"""Clean LLaDA decoding and paired timing aggregation for the QDLM benchmark."""

import statistics
from collections import defaultdict

import torch


def transfer_schedule(mask_count, steps):
    if mask_count < 0 or steps <= 0:
        raise ValueError("Invalid transfer schedule")
    base, remainder = divmod(mask_count, steps)
    return [base + int(i < remainder) for i in range(steps)]


@torch.no_grad()
def generate_clean(model, prompt, *, steps=128, gen_length=64,
                   block_length=64, mask_id=126336):
    """Match llada.py's temperature=0, cfg=0, low_confidence decoding.

    Preserve every forward, including zero-transfer steps. All tensors stay on
    the prompt's device. No analysis hooks, statistics, or file writes occur.
    """
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("This benchmark preserves request batch size 1")
    if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    blocks = gen_length // block_length
    if steps <= 0 or steps % blocks:
        raise ValueError("steps must be divisible by the number of blocks")
    steps_per_block = steps // blocks
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=prompt.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    for block in range(blocks):
        start = prompt.shape[1] + block * block_length
        end = start + block_length
        # Count the actual masks, as in the existing decoder. A prior step can
        # select an out-of-block position if all confidences are -inf.
        schedule = transfer_schedule(int((x[:, start:end] == mask_id).sum()),
                                     steps_per_block)
        for count in schedule:
            mask = x == mask_id
            logits = model(x).logits
            x0 = torch.argmax(logits, dim=-1)
            p = torch.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, end:] = -float("inf")
            x0 = torch.where(mask, x0, x)
            confidence = torch.where(mask, x0_p, -float("inf"))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            _, indices = torch.topk(confidence[0], k=count)
            transfer[0, indices] = True
            x[transfer] = x0[transfer]
    return x


def select_latency_cases(cases, per_task):
    """Choose evenly spaced cases from each subject's fixed evaluation slice."""
    if per_task <= 0:
        raise ValueError("latency-per-task must be positive")
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["task"]].append(case)
    selected = []
    for task in sorted(grouped):
        pool = grouped[task]
        count = min(per_task, len(pool))
        for i in range(count):
            selected.append(pool[(2 * i + 1) * len(pool) // (2 * count)])
    return selected


def summarize_latency(rows):
    """Compare FP16 and W2/W3 on the same prompts and the same physical GPU."""
    samples = defaultdict(list)
    for row in rows:
        if row["phase"] == "latency":
            samples[(row["case_id"], row["device"], row["variant"])].append(row)
    if not samples:
        raise ValueError("No latency measurements")
    pairs = sorted({(key[0], key[1]) for key in samples})
    by_device = {}
    for device in sorted({key[1] for key in pairs}):
        device_pairs = [key for key in pairs if key[1] == device]
        result = {"paired_prompts": len(device_pairs)}
        for metric in ("generation_seconds", "request_seconds"):
            totals = {}
            for variant in ("fp16", "sandwich_w23"):
                total = 0.0
                for case_id, _ in device_pairs:
                    key = (case_id, device, variant)
                    if key not in samples:
                        raise ValueError(f"Missing paired measurement: {key}")
                    other = "sandwich_w23" if variant == "fp16" else "fp16"
                    if len(samples[key]) != len(samples[(case_id, device, other)]):
                        raise ValueError("Unequal paired repetition counts")
                    total += statistics.median(row[metric] for row in samples[key])
                totals[variant] = total
            result[metric] = {
                "sum_of_prompt_medians": totals,
                "speedup": totals["fp16"] / totals["sandwich_w23"],
            }
        by_device[device] = result
    combined = {}
    for metric in ("generation_seconds", "request_seconds"):
        fp = sum(v[metric]["sum_of_prompt_medians"]["fp16"] for v in by_device.values())
        q = sum(v[metric]["sum_of_prompt_medians"]["sandwich_w23"] for v in by_device.values())
        combined[metric] = {"speedup": fp / q, "fp16_seconds": fp,
                            "sandwich_w23_seconds": q}
    return {"aggregation": "ratio of sums of per-prompt medians, paired by GPU",
            "by_device": by_device, "combined": combined}
