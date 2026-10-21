#!/usr/bin/env python3
"""Benchmark the existing sandwich GPTQ method with matched LLaDA decoding."""

import argparse
import ast
import contextlib
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "21905a0ef44a189deb1888c9b48290832db94e00"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def command_output(args):
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                              timeout=15, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Unavailable: {exc}"


def request_key(request):
    payload = [request.task_name, request.doc_id, request.idx, request.args]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["latency", "accuracy"], default="latency")
    parser.add_argument("--tasks", default="mmlu_generative")
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--latency-per-task", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--config", default="configs/qdlm_sandwich_w23.json")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()
    if min(args.limit, args.repeats, args.warmup, args.latency_per_task) <= 0:
        parser.error("limit, repeats, warmup, and latency-per-task must be positive")
    if args.gen_length <= 0 or args.block_length <= 0 or args.gen_length % args.block_length:
        parser.error("gen-length must be divisible by block-length")
    if args.steps <= 0 or args.steps % (args.gen_length // args.block_length):
        parser.error("steps must be divisible by the number of blocks")
    return args


def main():
    args = parse_args()
    os.chdir(ROOT)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(ROOT / "AutoGPTQ"), str(ROOT / "lm-evaluation-harness")]
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    import auto_gptq
    import lm_eval
    from qdlm_benchmark_core import generate_clean, select_latency_cases, summarize_latency

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Expose exactly two GPUs with CUDA_VISIBLE_DEVICES")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Launch with python, not torchrun or accelerate launch")
    for module, expected in [(auto_gptq, ROOT / "AutoGPTQ"),
                             (lm_eval, ROOT / "lm-evaluation-harness")]:
        if not Path(module.__file__).resolve().is_relative_to(expected):
            raise RuntimeError(f"Wrong source import: {module.__file__}")
    logging.basicConfig(level=logging.INFO)
    os.environ["QDLM_SANDWICH_CONFIG"] = str(Path(args.config).resolve())
    os.environ["QDLM_SANDWICH_PLAN_OUT"] = str(out / "sandwich_plan.json")
    os.environ["GPTQ_DUMP_CALIB_H"] = "0"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    eval_kwargs = dict(tasks=args.tasks.split(","), num_fewshot=args.num_fewshot,
                       limit=args.limit, random_seed=0, numpy_random_seed=1234,
                       torch_random_seed=1234, fewshot_random_seed=1234,
                       log_samples=False, confirm_run_unsafe_code=True)

    class RequestsCaptured(Exception):
        pass

    class RequestLM(LM):
        def __init__(self, answers=None):
            super().__init__()
            self.answers = answers
            self.requests = None

        def loglikelihood(self, requests):
            raise RuntimeError("Only generate_until tasks are supported")

        def loglikelihood_rolling(self, requests):
            raise RuntimeError("Only generate_until tasks are supported")

        def generate_until(self, requests):
            if self.answers is None:
                self.requests = list(requests)
                raise RequestsCaptured()
            keys = [request_key(r) for r in requests]
            if len(keys) != len(set(keys)) or set(keys) != set(self.answers):
                raise RuntimeError("Evaluation requests changed between capture and replay")
            return [self.answers[key] for key in keys]

    # Let this checkout's evaluator build the exact standard prompts. Stop
    # before inference or scoring; never report dummy responses as accuracy.
    capture = RequestLM()
    try:
        evaluator.simple_evaluate(capture, **eval_kwargs)
    except RequestsCaptured:
        pass
    if not capture.requests:
        raise RuntimeError("No generation requests were captured")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases = []
    for request in capture.requests:
        if request.repeats != 1:
            raise RuntimeError("Repeated task requests require a separate benchmark protocol")
        prompt, gen_kwargs = request.args
        until = gen_kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]
        ids = tokenizer(prompt)["input_ids"]
        cases.append(dict(case_id=request_key(request), task=request.task_name,
                          doc_id=request.doc_id, prompt=prompt, until=until,
                          input_ids=ids))
    if len({c["case_id"] for c in cases}) != len(cases):
        raise RuntimeError("Duplicate evaluation requests")
    latency_cases = select_latency_cases(cases, args.latency_per_task)
    latency_keys = {c["case_id"] for c in latency_cases}
    with (out / "prompts.jsonl").open("w") as handle:
        for case in cases:
            handle.write(json.dumps(dict(case, latency_sample=case["case_id"] in latency_keys)) + "\n")
    del capture

    source_files = ["AutoGPTQ/quantize.py", "AutoGPTQ/auto_gptq/modeling/_base.py",
                    "AutoGPTQ/auto_gptq/modeling/_utils.py",
                    "AutoGPTQ/auto_gptq/modeling/llada.py",
                    "AutoGPTQ/auto_gptq/mixed_precision/sandwich.py",
                    "AutoGPTQ/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py",
                    "AutoGPTQ/autogptq_extension/cuda_256/autogptq_cuda_kernel_256.cu",
                    "AutoGPTQ/autogptq_extension/cuda_64/autogptq_cuda_kernel_64.cu",
                    "lm-evaluation-harness/lm_eval/models/llada.py",
                    "tools/benchmark_sandwich.py", "tools/qdlm_benchmark_core.py"]
    metadata = dict(args=vars(args), reviewed_source_revision=SOURCE_REVISION,
                    actual_revision=command_output(["git", "rev-parse", "HEAD"]),
                    source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                   for p in source_files},
                    source_local_diff=command_output(["git", "diff", "HEAD", "--", *source_files]),
                    torch=torch.__version__, cuda=torch.version.cuda,
                    visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    gpus=[str(torch.cuda.get_device_properties(i)) for i in range(2)],
                    nvidia_smi_before=command_output(["nvidia-smi"]),
                    standard_request_count=len(cases),
                    requests_per_task=dict(Counter(c["task"] for c in cases)),
                    latency_request_count=len(latency_cases),
                    request_batch_size=1, baseline_dtype="float16",
                    quant_activation_dtype="float16", autocast=False,
                    temperature=0, cfg_scale=0, remasking="low_confidence",
                    quant_start_step=0, all_forward_calls_preserved=True,
                    initial_zero_transfer_steps=max(0, args.steps - args.gen_length),
                    inference_excludes=["model loading", "calibration", "packing",
                                        "warmup", "profiling", "file writes"],
                    generation_timing="GPU-resident prompt to synchronized final tokens",
                    request_timing="tokenization, input copy, generation, output copy, stop processing, decode",
                    accuracy_computed=args.mode == "accuracy")
    write_json(out / "metadata.json", metadata)
    print(f"Standard requests: {len(cases)} | latency requests: {len(latency_cases)} | mode: {args.mode}", flush=True)

    # Keep the calibration and packing recipe from AutoGPTQ/quantize.py.
    started = time.perf_counter()
    qmodel = AutoGPTQForCausalLM.from_pretrained(
        args.model, BaseQuantizeConfig(bits=2, group_size=128, desc_act=False),
        trust_remote_code=True, torch_dtype=torch.float16)
    try:
        calibration_tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=False, trust_remote_code=True)
    except Exception:
        calibration_tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, trust_remote_code=True)
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    encoded = calibration_tokenizer("\n\n".join(train["text"]), return_tensors="pt").input_ids
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    examples = []
    for _ in range(128):
        start = random.randint(0, encoded.shape[1] - 2048 - 1)
        ids = encoded[:, start:start + 2048]
        examples.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    qmodel.quantize(examples)
    qmodel.to("cuda:1").eval()
    metadata["quant_model_preparation_seconds"] = time.perf_counter() - started
    metadata["packed_validation"] = qmodel._sandwich_packed_summary
    if metadata["packed_validation"]["status"] != "pass":
        raise RuntimeError("Packed sandwich validation failed")
    del encoded, train, examples, calibration_tokenizer
    gc.collect()
    for device in range(2):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    fpmodel = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                       torch_dtype=torch.float16).to("cuda:0").eval()
    models = {"fp16": fpmodel, "sandwich_w23": qmodel}
    quant_modules = [(n, m) for n, m in qmodel.named_modules() if hasattr(m, "qweight")]
    if not quant_modules or {m.bits for _, m in quant_modules} != {2, 3}:
        raise RuntimeError("Expected packed W2 and W3 modules")
    metadata["quant_modules"] = [dict(name=n, bits=m.bits, backend=type(m).__module__,
                                      infeatures=m.infeatures, outfeatures=m.outfeatures,
                                      kernel_switch_threshold=getattr(m, "kernel_switch_threshold", None))
                                 for n, m in quant_modules]
    metadata["loaded_cuda_extensions"] = {}
    for _, module in quant_modules:
        extension = getattr(module, "autogptq_cuda", None)
        if extension is not None:
            path = Path(extension.__file__).resolve()
            if str(path) not in metadata["loaded_cuda_extensions"]:
                metadata["loaded_cuda_extensions"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["unquantized_linear_modules"] = [n for n, m in qmodel.named_modules()
                                               if isinstance(m, torch.nn.Linear)]
    metadata["quantized_weight_average_bits"] = (
        sum(m.bits * m.infeatures * m.outfeatures for _, m in quant_modules)
        / sum(m.infeatures * m.outfeatures for _, m in quant_modules))
    metadata["quantized_weight_fraction"] = (
        sum(m.infeatures * m.outfeatures for _, m in quant_modules)
        / sum(p.numel() for p in fpmodel.parameters()))
    metadata["parameter_dtypes"] = {
        name: dict(Counter(str(p.dtype) for p in model.parameters()))
        for name, model in models.items()}
    write_json(out / "metadata.json", metadata)
    decode_kwargs = dict(steps=args.steps, gen_length=args.gen_length,
                         block_length=args.block_length, mask_id=args.mask_id)

    # Verify the actual GPU model's tokens against the existing decoder before
    # claiming that removing diagnostics preserves this experiment's behavior.
    from lm_eval.models.llada import generate as generate_reference
    source_tree = ast.parse((ROOT / "lm-evaluation-harness/lm_eval/models/llada.py").read_text())
    flag_names = {"pair_analysis", "perturb_analysis", "c_analysis",
                  "saliency_analysis", "sp_analysis", "recovery_analysis"}
    assignments = {}
    for node in ast.walk(source_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in flag_names:
                    assignments.setdefault(target.id, []).append(node.value)
    if set(assignments) != flag_names or any(
            len(values) != 1 or not isinstance(values[0], ast.Constant)
            or values[0].value is not False for values in assignments.values()):
        raise RuntimeError("Reference parity check requires all six analysis flags to be False")
    os.environ["LLADA_DUMP_FULL_ACT"] = "0"
    parity_dir = out / "reference_check"
    parity_dir.mkdir()
    metadata["reference_parity"] = {}
    for variant, device in {"fp16": "cuda:0", "sandwich_w23": "cuda:1"}.items():
        print(f"Checking original versus clean decoding: {variant}", flush=True)
        prompt = torch.tensor([latency_cases[0]["input_ids"]], device=device)
        with torch.cuda.device(device), torch.no_grad(), torch.autocast("cuda", enabled=False):
            try:
                os.chdir(parity_dir)
                with (parity_dir / f"{variant}.log").open("w") as log, contextlib.redirect_stdout(log):
                    reference = generate_reference(models[variant], prompt, **decode_kwargs).cpu()
            finally:
                os.chdir(ROOT)
            clean = generate_clean(models[variant], prompt, **decode_kwargs).cpu()
        matches = torch.equal(reference, clean)
        metadata["reference_parity"][variant] = matches
        write_json(out / "metadata.json", metadata)
        if not matches:
            write_json(parity_dir / f"{variant}_mismatch.json",
                       {"reference": reference.tolist(), "clean": clean.tolist()})
            raise RuntimeError(f"Decoder parity failed for {variant}; benchmark stopped")
        del reference, clean, prompt
    print("Reference decoder parity passed for both models.", flush=True)
    rows = []
    answers = {name: {} for name in models}
    output_tokens = {name: {} for name in models}
    output_mismatches = []

    def measure(case, variant, device, phase, trial, assignment):
        model = models[variant]
        with torch.cuda.device(device), torch.no_grad(), torch.autocast("cuda", enabled=False):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            request_start = time.perf_counter()
            ids = tokenizer(case["prompt"])["input_ids"]
            if ids != case["input_ids"]:
                raise RuntimeError("Prompt tokenization changed")
            prompt = torch.tensor([ids], dtype=torch.long, device=device)
            torch.cuda.synchronize(device)
            generation_start = time.perf_counter()
            result = generate_clean(model, prompt, **decode_kwargs)
            torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - generation_start
            tokens = result[0, len(ids):].cpu().tolist()
            answer = tokenizer.decode(tokens, skip_special_tokens=False)
            for stop in case["until"]:
                if stop in answer:
                    answer = answer.split(stop)[0]
            answer = tokenizer.decode(tokenizer(answer)["input_ids"], skip_special_tokens=True)
            request_seconds = time.perf_counter() - request_start
            row = dict(case_id=case["case_id"], task=case["task"], doc_id=case["doc_id"],
                       variant=variant, device=str(device), assignment=assignment,
                       phase=phase, trial=trial, prompt_tokens=len(ids),
                       generated_slots=args.gen_length, forward_calls=args.steps,
                       generation_seconds=generation_seconds, request_seconds=request_seconds,
                       amortized_step_seconds=generation_seconds / args.steps,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        return row, answer, tokens

    def record(case, variant, device, phase, trial, assignment):
        row, answer, tokens = measure(case, variant, device, phase, trial, assignment)
        key = case["case_id"]
        if key in output_tokens[variant] and output_tokens[variant][key] != tokens:
            output_mismatches.append({"case_id": key, "variant": variant,
                                      "assignment": assignment, "trial": trial})
        else:
            output_tokens[variant].setdefault(key, tokens)
        answers[variant].setdefault(key, answer)
        rows.append(row)
        with (out / "timings.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        with (out / "outputs.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(row, answer=answer, generated_token_ids=tokens)) + "\n")
        print(f"{phase} {assignment} {variant} {case['task']} doc={case['doc_id']} "
              f"repeat={trial} gen={row['generation_seconds']:.3f}s "
              f"request={row['request_seconds']:.3f}s", flush=True)

    for assignment, placement in [("A", {"fp16": "cuda:0", "sandwich_w23": "cuda:1"}),
                                  ("B", {"fp16": "cuda:1", "sandwich_w23": "cuda:0"})]:
        if assignment == "B":
            # Stage through CPU so a swap never needs both models on one GPU.
            qmodel.to("cpu")
            fpmodel.to("cuda:1")
            qmodel.to("cuda:0")
            for device in range(2):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        for variant, device in placement.items():
            resident = {str(t.device) for t in list(models[variant].parameters())
                        + list(models[variant].buffers())}
            if resident != {device}:
                raise RuntimeError(f"Unexpected device placement for {variant}: {resident}")
            for _ in range(args.warmup):
                measure(latency_cases[0], variant, device, "warmup", -1, assignment)
        for index, case in enumerate(latency_cases):
            for trial in range(args.repeats):
                order = list(models)
                if (index + trial + (assignment == "B")) % 2:
                    order.reverse()
                for variant in order:
                    record(case, variant, placement[variant], "latency", trial, assignment)
        if assignment == "A" and args.mode == "accuracy":
            # Reuse the subset predictions and score all 80 examples per task.
            for index, case in enumerate(cases):
                if case["case_id"] in latency_keys:
                    continue
                order = list(models)
                if index % 2:
                    order.reverse()
                for variant in order:
                    record(case, variant, placement[variant], "accuracy", 0, assignment)

    summary = {"mode": args.mode, "latency": summarize_latency(rows),
               "latency_prompts": len(latency_cases), "standard_prompts": len(cases),
               "output_mismatch_count": len(output_mismatches),
               "output_mismatches": output_mismatches}
    write_json(out / "summary.json", summary)
    if args.mode == "accuracy":
        for variant in models:
            result = evaluator.simple_evaluate(RequestLM(answers[variant]), **eval_kwargs)
            write_json(out / f"accuracy_{variant}.json", result)
            summary[f"accuracy_{variant}"] = result["results"]
    else:
        summary["accuracy_note"] = "Latency subset only; no accuracy metric is computed."
    write_json(out / "summary.json", summary)

    if not args.skip_profile:
        # Profile a real, median-length masked input separately from all timers.
        case = sorted(latency_cases, key=lambda c: len(c["input_ids"]))[len(latency_cases) // 2]
        for variant, device in {"fp16": "cuda:1", "sandwich_w23": "cuda:0"}.items():
            with torch.cuda.device(device), torch.no_grad(), torch.autocast("cuda", enabled=False):
                x = torch.full((1, len(case["input_ids"]) + args.gen_length),
                               args.mask_id, dtype=torch.long, device=device)
                x[0, :len(case["input_ids"])]=torch.tensor(case["input_ids"], device=device)
                routes, handles = [], []
                def route_hook(name):
                    def hook(module, inputs):
                        tensor = inputs[0]
                        m = tensor.numel() // tensor.shape[-1]
                        backend = type(module).__module__
                        route = "backend-specific; inspect profiler"
                        if backend.endswith("qlinear_cuda_old"):
                            threshold = module.kernel_switch_threshold
                            packed = (tensor.device.type == "cuda"
                                      and module.autogptq_cuda_available is True
                                      and (threshold is False or m < threshold))
                            route = "packed_cuda" if packed else "unpack_then_dense_matmul"
                        routes.append(dict(name=name, bits=module.bits, m=m,
                                           input_shape=list(tensor.shape), route=route,
                                           backend=backend, input_dtype=str(tensor.dtype)))
                    return hook
                try:
                    if variant == "sandwich_w23":
                        handles = [m.register_forward_pre_hook(route_hook(n)) for n, m in quant_modules]
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=True) as profiler:
                        result = models[variant](x)
                        torch.cuda.synchronize(device)
                        del result
                finally:
                    for handle in handles:
                        handle.remove()
                (out / f"profile_{variant}.txt").write_text(
                    profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=60))
                profiler.export_chrome_trace(str(out / f"profile_{variant}.json"))
                if routes:
                    write_json(out / "backend_routes.json", dict(
                        case_id=case["case_id"], records=routes,
                        route_counts=dict(Counter(r["route"] for r in routes))))
    metadata["nvidia_smi_after"] = command_output(["nvidia-smi"])
    write_json(out / "metadata.json", metadata)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"Results saved to {out}", flush=True)


if __name__ == "__main__":
    main()
