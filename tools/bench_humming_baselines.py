"""Uniform W2/W3 controls, matched Humming latency, and isolated model memory.

Only physical GPUs 2 and 3 are allowed. Existing W2/W3 mixed weights and the
4560 saved requests are reused. Uniform controls recalibrate the same modules.
No existing quantizer, decoder, environment, or mixed checkpoint is modified.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import bench_humming_samples as sample
from bench_humming_full import read_jsonl, validate_cases, write_json

ROOT, HDD = sample.ROOT, sample.HDD
ASSIGNMENTS = {2: "uniform_w2", 3: "uniform_w3"}
MIXED = "sandwich_w23"


class UniformPlan:
    """Implement the existing quantizer plan interface without edge constraints."""
    def __init__(self, bits, layer_weights):
        if bits not in (2, 3) or len(layer_weights) != 32 or any(n <= 0 for n in layer_weights):
            raise ValueError("Expected W2/W3 and 32 positive block parameter counts")
        self.method = "uniform_gptq_control"
        self.stage = f"uniform_w{bits}"
        self.layer_weights = tuple(layer_weights)
        self.current_bits = (bits,) * 32
        self.current_layer_bit_map = dict(enumerate(self.current_bits))
        self.current_average_bits = float(bits)
        self.final_preview_bits = self.current_bits
        self.final_preview_average_bits = float(bits)
        for b in (1, 2, 3):
            layers = tuple(range(32)) if b == bits else ()
            setattr(self, f"current_w{b}_layers", layers)
            setattr(self, f"final_w{b}_layers", layers)

    def to_dict(self, packed_summary=None):
        return {"method": self.method, "stage": self.stage,
                "current_bits": self.current_bits, "layer_weights": self.layer_weights,
                "current_average_bits": self.current_average_bits,
                "scope": "Same selected linears as the mixed checkpoint, not every model parameter",
                "packed_validation": packed_summary}


def uniform_identity(model, bits):
    identity = sample.cache_identity(model)
    identity["uniform_control"] = {"bits": bits, "format": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return identity


def validate_uniform_specs(specs, bits, reference):
    sample.check_specs(reference)
    fields = ("name", "infeatures", "outfeatures", "group_size", "has_bias")
    layout = lambda entries: sorted(tuple(s[k] for k in fields) for s in entries)
    if len(specs) != 128 or Counter(s["bits"] for s in specs) != {bits: 128}:
        raise ValueError("Uniform control must contain exactly 128 linears at the requested bit width")
    if layout(specs) != layout(reference):
        raise ValueError("Uniform and mixed quantized module coverage differs")


def select_latency(cases):
    pool = sorted((c for c in cases if c.get("latency_sample")),
                  key=lambda c: (len(c["input_ids"]), c["case_id"]))
    if len(pool) != 57 or len({c["task"] for c in pool}) != 57:
        raise ValueError("Expected the original one-per-subject latency pool")
    chosen = [pool[round((0.05 + i * 0.9 / 8) * (len(pool) - 1))] for i in range(9)]
    if len({c["case_id"] for c in chosen}) != 9:
        raise ValueError("Latency selection contains duplicates")
    return chosen


def validate_outputs(cases, rows, variant):
    expected = {c["case_id"] for c in cases}
    seen = set()
    for row in rows:
        key = row["case_id"]
        if key not in expected or key in seen or row["variant"] != variant:
            raise ValueError("Unexpected, duplicate, or mislabeled prediction")
        if len(row["tokens"]) != 64 or any(type(t) is not int or t < 0 for t in row["tokens"]):
            raise ValueError("Prediction does not contain 64 valid generated token IDs")
        seen.add(key)
    if seen != expected:
        raise ValueError(f"Incomplete predictions: {len(seen)}/{len(expected)}")


def summarize_pairs(cases, rows, variant, gpu, repeats=3):
    expected = {(c["case_id"], v, r) for c in cases
                for v in ("fp16", variant) for r in range(repeats)}
    values = {}
    for row in rows:
        key = (row["case_id"], row["variant"], row["repeat"])
        if key not in expected or key in values or row["physical_gpu"] != gpu:
            raise ValueError("Duplicate, unexpected, or wrong-GPU latency observation")
        if any(not math.isfinite(row[m]) or row[m] <= 0
               for m in ("generation_seconds", "request_seconds")):
            raise ValueError("Invalid latency")
        if row["generation_seconds"] > row["request_seconds"] + 1e-6:
            raise ValueError("Generation cannot take longer than its containing request")
        values[key] = row
    if set(values) != expected:
        raise ValueError("Incomplete paired latency observations")
    result = {"variant": variant, "physical_gpu": gpu, "paired_prompts": len(cases),
              "repeats": repeats, "aggregation": "ratio of sums of per-prompt medians"}
    for metric in ("generation_seconds", "request_seconds"):
        totals = {v: sum(statistics.median(values[c["case_id"], v, r][metric]
                         for r in range(repeats)) for c in cases) for v in ("fp16", variant)}
        result[metric] = {"sum_of_prompt_medians": totals,
                          "speedup": totals["fp16"] / totals[variant]}
    return result


def manifest(args):
    return json.loads((args.output / "metadata.json").read_text())


def check_runtime(args):
    torch = sample.configure_torch()
    from importlib.metadata import distribution
    if int(os.environ["QDLM_PHYSICAL_GPU"]) != args.gpu:
        raise ValueError("Physical GPU assignment changed")
    dist = distribution("humming-kernels")
    revision = json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id")
    if str(torch.__version__) != "2.7.1+cu128" or torch.version.cuda != "12.8" or revision != sample.HUMMING_REV:
        raise ValueError("Use the existing tested Humming environment and revision")
    return torch


def payload_for(args, variant, torch):
    meta = manifest(args)
    cache = Path(meta["caches"][variant])
    payload = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    expected = (sample.cache_identity(args.model) if variant == MIXED
                else uniform_identity(args.model, int(variant[-1])))
    if payload["identity"] != expected:
        raise ValueError(f"Stale or incompatible cache: {cache}")
    if variant == MIXED:
        sample.check_specs(payload["specs"])
    else:
        reference = torch.load(meta["caches"][MIXED], map_location="cpu", weights_only=True, mmap=True)
        validate_uniform_specs(payload["specs"], int(variant[-1]), reference["specs"])
    if payload["packed_validation"]["status"] != "pass":
        raise ValueError("Packed validation failed")
    return payload


def prepare(args):
    import torch
    from auto_gptq import mixed_precision
    bits = int(args.variant[-1])
    meta = manifest(args)
    reference = payload_for(args, MIXED, torch)["specs"]
    identity = uniform_identity(args.model, bits)
    out = args.output / args.variant
    out.mkdir(exist_ok=True)
    job = SimpleNamespace(model=args.model, output=out, cache=Path(meta["caches"][args.variant]))
    # Only the plan is replaced during calibration. Packing, module selection,
    # calibration data, GPTQ fitting, and validation remain the tested path.
    with ExitStack() as stack:
        stack.enter_context(patch.object(mixed_precision, "build_sandwich_plan",
            lambda *, config, layer_weights: UniformPlan(bits, layer_weights)))
        stack.enter_context(patch.object(sample, "check_specs",
            lambda specs: validate_uniform_specs_unpatched(specs, bits, reference)))
        stack.enter_context(patch.object(sample, "cache_identity", lambda model: identity))
        sample.prepare_cache(job)
    payload_for(args, args.variant, torch)
    write_json(out / "calibration_complete.json", {"variant": args.variant,
        "cache": str(job.cache), "identity": identity, "status": "pass"})


# Retain the original mixed checker when prepare temporarily replaces it.
_mixed_specs_check = sample.check_specs


def validate_uniform_specs_unpatched(specs, bits, reference):
    with patch.object(sample, "check_specs", _mixed_specs_check):
        validate_uniform_specs(specs, bits, reference)


def load_quantized(args, variant, torch, out):
    from transformers import AutoModel
    payload = payload_for(args, variant, torch)
    specs, state = payload["specs"], payload["state"]
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    prefixes = tuple(s["name"] + "." for s in specs)
    unquant = {k: v for k, v in state.items() if not k.startswith(prefixes)}
    missing, unexpected = model.load_state_dict(unquant, strict=False)
    expected = {s["name"] + "." + k for s in specs
                for k in (["weight", "bias"] if s["has_bias"] else ["weight"])}
    if set(missing) != expected or unexpected:
        raise ValueError("Unexpected unquantized state")
    reports = []
    for i, spec in enumerate(specs):
        sample.assert_idle_for_us()
        attrs = {k: state[spec["name"] + "." + k] for k in ("qweight", "qzeros", "scales", "g_idx")}
        attrs["bias"] = state[spec["name"] + ".bias"] if spec["has_bias"] else None
        layer, report = sample.convert_linear(torch, SimpleNamespace(**spec, **attrs), spec["name"])
        parent, _, leaf = spec["name"].rpartition(".")
        setattr(model.get_submodule(parent), leaf, layer)
        reports.append(report)
        if (i + 1) % 32 == 0:
            print(f"GPU {args.gpu} {variant}: converted {i + 1}/128", flush=True)
    write_json(out / "conversion.json", reports)
    write_json(out / "storage.json", {**payload["packed_validation"],
        "cache_file_bytes": Path(manifest(args)["caches"][variant]).stat().st_size,
        "humming_state_tensor_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
        "note": "Tensor storage and serialized GPTQ cache are not peak runtime memory"})
    del payload, state, unquant, attrs, layer
    gc.collect()
    return model.cuda().eval()


def load_fp16(args, torch):
    from transformers import AutoModel
    return AutoModel.from_pretrained(args.model, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda().eval()


def generate(torch, model, tokenizer, case):
    from qdlm_benchmark_core import generate_clean
    torch.cuda.synchronize()
    start = time.perf_counter()
    ids = tokenizer(case["prompt"], return_tensors="pt")["input_ids"].cuda()
    torch.cuda.synchronize()
    gen_start = time.perf_counter()
    output = generate_clean(model, ids, steps=128, gen_length=64, block_length=64)
    torch.cuda.synchronize()
    gen_time = time.perf_counter() - gen_start
    tokens = output[0, ids.shape[1]:].cpu().tolist()
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    for stop in case["until"]:
        if stop:
            response = response.split(stop)[0]
    torch.cuda.synchronize()
    return {"tokens": tokens, "response": response, "generation_seconds": gen_time,
            "request_seconds": time.perf_counter() - start}


def profile_and_memory(args, torch, model, tokenizer, cases, label, out):
    # This function runs while exactly one model occupies the GPU.
    rows = []
    with torch.inference_mode():
        for case in (cases[0], cases[4], cases[8]):
            sample.assert_idle_for_us()
            generate(torch, model, tokenizer, case)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            generate(torch, model, tokenizer, case)
            sample.assert_idle_for_us()
            rows.append({"case_id": case["case_id"], "m": len(case["input_ids"]) + 64,
                "resident_allocated_bytes": resident,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        case = cases[4]
        ids = torch.tensor([case["input_ids"] + [126336] * 64], device="cuda")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            logits = model(ids).logits
            torch.cuda.synchronize()
        del logits, ids
    (out / f"profile_{label}.txt").write_text(
        prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=40))
    write_json(out / f"memory_{label}.json", {"variant": label, "physical_gpu": args.gpu,
        "single_model_resident": True, "rows": rows,
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
        "model_state_tensor_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
        "note": "PyTorch allocator measurements on three selected prompts, not phone memory or total process RSS",
        "profile_note": "Separate one-forward profile at the median selected M, outside timing"})


def measure(args):
    torch = check_runtime(args)
    from transformers import AutoTokenizer
    cases = read_jsonl(args.output / "latency_prompts.jsonl")
    out = args.output / f"diagnostics/gpu{args.gpu}/{args.variant}"
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for c in cases:
        if tokenizer(c["prompt"])["input_ids"] != c["input_ids"]:
            raise ValueError("Saved tokenization changed")
    fp = load_fp16(args, torch)
    profile_and_memory(args, torch, fp, tokenizer, cases, "fp16", out)
    del fp
    gc.collect()
    torch.cuda.empty_cache()
    baseline_bytes = torch.cuda.memory_allocated()
    if baseline_bytes > 256 * 1024**2:
        raise RuntimeError("FP16 model still occupies GPU memory before isolated quantized measurement")
    quant = load_quantized(args, args.variant, torch, out)
    profile_and_memory(args, torch, quant, tokenizer, cases, args.variant, out)
    fp = load_fp16(args, torch)
    models = {"fp16": fp, args.variant: quant}
    rows = []
    with torch.inference_mode(), (out / "timings.jsonl").open("x") as handle:
        for i, case in enumerate(cases):
            for v in models:
                sample.assert_idle_for_us()
                generate(torch, models[v], tokenizer, case)
            for repeat in range(3):
                order = list(models) if (i + repeat + args.gpu) % 2 == 0 else list(models)[::-1]
                for v in order:
                    sample.assert_idle_for_us()
                    row = {"case_id": case["case_id"], "variant": v, "repeat": repeat,
                           "physical_gpu": args.gpu, **generate(torch, models[v], tokenizer, case)}
                    sample.assert_idle_for_us()
                    rows.append(row)
                    print(json.dumps(row), file=handle, flush=True)
            print(f"GPU {args.gpu} {args.variant}: latency {i + 1}/9", flush=True)
    summary = summarize_pairs(cases, rows, args.variant, args.gpu)
    write_json(out / "summary.json", summary)
    print(f"GPU {args.gpu} {args.variant}: request speedup {summary['request_seconds']['speedup']:.4f}x", flush=True)


def accuracy(args):
    torch = check_runtime(args)
    from transformers import AutoTokenizer
    cases = read_jsonl(args.output / "prompts.jsonl")
    validate_cases(cases)
    out = args.output / args.variant
    out.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for c in cases:
        if tokenizer(c["prompt"])["input_ids"] != c["input_ids"]:
            raise ValueError("Saved tokenization changed")
    model = load_quantized(args, args.variant, torch, out)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), (out / "outputs.jsonl").open("x") as handle:
        for i, case in enumerate(cases):
            sample.assert_idle_for_us()
            row = {"case_id": case["case_id"], "variant": args.variant,
                   "physical_gpu": args.gpu, **generate(torch, model, tokenizer, case)}
            sample.assert_idle_for_us()
            print(json.dumps(row), file=handle, flush=True)
            progress = {"variant": args.variant, "completed": i + 1, "total": len(cases),
                        "elapsed_seconds": time.perf_counter() - started}
            write_json(out / "progress.json", progress)
            if i == 0 or (i + 1) % 40 == 0:
                print(f"GPU {args.gpu} {args.variant}: accuracy {i + 1}/4560", flush=True)
    validate_outputs(cases, read_jsonl(out / "outputs.jsonl"), args.variant)
    write_json(out / "generation_complete.json", {"predictions": 4560,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "note": "Single-model full accuracy peak includes first-use compilation; accuracy-loop times are not benchmark latency"})


def score(args):
    sys.path.insert(0, str(ROOT / "lm-evaluation-harness"))
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from transformers import AutoTokenizer
    from benchmark_sandwich import request_key
    cases = read_jsonl(args.output / "prompts.jsonl")
    validate_cases(cases)
    by_id = {c["case_id"]: c for c in cases}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for variant in ASSIGNMENTS.values():
        out = args.output / variant
        rows = read_jsonl(out / "outputs.jsonl")
        validate_outputs(cases, rows, variant)
        answers = {}
        for row in rows:
            answer = tokenizer.decode(row["tokens"], skip_special_tokens=False)
            for stop in by_id[row["case_id"]]["until"]:
                if stop:
                    answer = answer.split(stop)[0]
            answers[row["case_id"]] = tokenizer.decode(tokenizer(answer)["input_ids"], skip_special_tokens=True)

        class Replay(LM):
            def __init__(self):
                super().__init__()
                self.seen = set()

            def loglikelihood(self, requests):
                raise NotImplementedError

            def loglikelihood_rolling(self, requests):
                raise NotImplementedError

            def generate_until(self, requests):
                keys = [request_key(r) for r in requests]
                if len(set(keys)) != len(keys) or self.seen.intersection(keys) or not set(keys).issubset(answers):
                    raise ValueError("Evaluation request identity mismatch")
                self.seen.update(keys)
                return [answers[k] for k in keys]

        replay = Replay()
        result = evaluator.simple_evaluate(replay, tasks=["mmlu_generative"], num_fewshot=5,
            limit=80, random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
            fewshot_random_seed=1234, log_samples=False, confirm_run_unsafe_code=True)
        if replay.seen != set(answers):
            raise ValueError("Accuracy replay missed predictions")
        write_json(out / "accuracy.json", result)
        acc = result["results"]["mmlu_generative"]["exact_match,get_response"]
        print(f"{variant} MMLU accuracy: {acc:.2%} (4560 requests)", flush=True)


def summarize_run(args):
    meta = manifest(args)
    cases = read_jsonl(args.output / "prompts.jsonl")
    latency_cases = read_jsonl(args.output / "latency_prompts.jsonl")
    old = json.loads((args.source / "summary.json").read_text())
    results = {}
    for gpu, variant in ASSIGNMENTS.items():
        validate_outputs(cases, read_jsonl(args.output / variant / "outputs.jsonl"), variant)
        accuracy_result = json.loads((args.output / variant / "accuracy.json").read_text())
        diagnostic = {}
        for target in (variant, MIXED):
            directory = args.output / f"diagnostics/gpu{gpu}/{target}"
            diagnostic[target] = summarize_pairs(latency_cases, read_jsonl(directory / "timings.jsonl"), target, gpu)
            diagnostic[target]["memory"] = {v: json.loads((directory / f"memory_{v}.json").read_text())
                                           for v in ("fp16", target)}
        results[variant] = {"accuracy": accuracy_result["results"]["mmlu_generative"]["exact_match,get_response"],
            "storage": json.loads((args.output / variant / "storage.json").read_text()),
            "diagnostics": diagnostic}
    summary = {"status": "pass", "accuracy_prompts_per_uniform_variant": 4560,
        "latency_prompts": len(latency_cases), "latency_repeats": 3, "results": results,
        "existing_mixed_run": str(args.source), "existing_accuracy": old["mmlu_accuracy"],
        "existing_mixed_full_request_speedup": old["latency"]["combined"]["request_seconds"]["speedup"],
        "interpretation": "Compare each uniform control with mixed on the same GPU using the new nine-prompt diagnostics. Do not compare absolute GPU2 and GPU3 latency or treat the nine-prompt estimates as full-benchmark speedups.",
        "mobile_note": "L40S CUDA evidence only. Mobile runtime, memory and energy remain unmeasured."}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "complete.json", {"status": "pass", "accuracy_prompts_per_variant": 4560,
        "variants": list(ASSIGNMENTS.values()), "latency_prompts": 9, "latency_repeats": 3})
    print(json.dumps({v: {"accuracy": r["accuracy"], "request_speedup_9_prompts":
        r["diagnostics"][v]["request_seconds"]["speedup"]} for v, r in results.items()}, indent=2), flush=True)
    print(f"Completion verified. Results: {args.output}", flush=True)


def launch(args):
    from bench_flute_shape_ab import gpu_state, try_lock
    import fcntl
    old_meta = json.loads((args.source / "metadata.json").read_text())
    old_done = json.loads((args.source / "complete.json").read_text())
    if old_done.get("status") != "pass" or old_done.get("paired_prompts") != 4560:
        raise ValueError("The source full run must have passed all completion checks")
    if old_meta["model"] != args.model or old_meta["setup"] != {
            "tasks": "mmlu_generative", "num_fewshot": 5, "limit": 80,
            "steps": 128, "gen_length": 64, "block_length": 64}:
        raise ValueError("Source model or evaluation setup changed")
    for name, digest in old_meta["code_sha256"].items():
        if hashlib.sha256((ROOT / "tools" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source code changed since the completed run: {name}")
    if not Path(old_meta["cache"]).is_file():
        raise FileNotFoundError("The original mixed cache is missing")
    cases = read_jsonl(args.source / "prompts.jsonl")
    validate_cases(cases)
    selected = select_latency(cases)
    python = ROOT / ".venv/qdlm-humming/bin/python"
    if not python.is_file():
        raise FileNotFoundError("The tested Humming environment is missing")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("prompts.jsonl", cases), ("latency_prompts.jsonl", selected)):
        with (args.output / name).open("x") as handle:
            for row in rows:
                print(json.dumps(row), file=handle)
    caches = {MIXED: old_meta["cache"]}
    for variant in ASSIGNMENTS.values():
        digest = hashlib.sha256(json.dumps(uniform_identity(args.model, int(variant[-1])), sort_keys=True).encode()).hexdigest()[:20]
        caches[variant] = str(HDD / "cache/gptq" / f"{variant}_{digest}.pt")
    write_json(args.output / "metadata.json", {"model": args.model, "source": str(args.source),
        "setup": old_meta["setup"], "caches": caches, "gpu_assignments": ASSIGNMENTS,
        "quant_backend": "humming", "humming_revision": sample.HUMMING_REV,
        "source_prompts_sha256": hashlib.sha256((args.source / "prompts.jsonl").read_bytes()).hexdigest(),
        "latency_selection": "Nine fixed length quantiles from the original 57 latency requests",
        "calibration": "Same Wikitext2 128 x 2048, seed 0, group size 128, desc_act=False and selected linears",
        "accuracy_timing_note": "Accuracy-loop times include first use and must not be used as speed benchmarks"})
    children = []

    def stop_children():
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # Disable inherited research instrumentation in fresh subprocesses.
    for key in list(env):
        if key.startswith(("QDLM_", "GPTQ_DUMP_", "LLADA_")):
            env.pop(key)
    env.update(HUMMING_DISABLE_PARALLEL_BUILD="1", HUMMING_USE_TORCH_STABLE_API="0",
        HUMMING_CACHE_DIR=str(HDD / "cache/humming"), HUMMING_TMP_DIR=str(HDD / "cache/humming_tmp"),
        TORCH_EXTENSIONS_DIR=str(HDD / "cache/torch_extensions_humming"),
        MAX_JOBS="2", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    common = [str(Path(__file__).resolve()), "--output", str(args.output),
              "--source", str(args.source), "--model", args.model]

    def run_phase(phase, assignments, executable):
        children.clear()
        for gpu, variant in assignments.items():
            child_env = dict(env, CUDA_VISIBLE_DEVICES=devices[gpu], QDLM_PHYSICAL_GPU=str(gpu))
            child = subprocess.Popen([str(executable), "-u", *common, "--phase", phase,
                "--gpu", str(gpu), "--variant", variant], env=child_env, start_new_session=True)
            children.append(child)
            print(f"Phase {phase}, GPU {gpu}, {variant}: PID {child.pid}", flush=True)
        write_json(args.output / "pids.json", {"parent": os.getpid(), "phase": phase,
            "children": [c.pid for c in children]})
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None, 0) for c in children):
                raise RuntimeError(f"Phase {phase} failed")
            time.sleep(2)
        if any(c.returncode != 0 for c in children):
            raise RuntimeError(f"Phase {phase} failed")

    try:
        with (ROOT / "logs/dlm_sandwich_paired_2gpu.lock").open("a") as lock:
            while True:
                if try_lock(lock):
                    devices, busy = gpu_state()
                    if all(not busy[devices[g]] for g in (2, 3)):
                        break
                    fcntl.flock(lock, fcntl.LOCK_UN)
                if args.wait_hours <= 0:
                    raise RuntimeError("GPUs 2,3 or the QDLM lock are busy")
                print(f"Waiting for physical GPUs 2,3 and the QDLM lock. Next check in {args.wait_hours:g} hours.", flush=True)
                time.sleep(args.wait_hours * 3600)
            write_json(args.output / "devices.json", devices)
            run_phase("preflight", ASSIGNMENTS, python)
            run_phase("prepare", ASSIGNMENTS, sample.BASE_PYTHON)
            run_phase("measure", ASSIGNMENTS, python)
            run_phase("measure", {2: MIXED, 3: MIXED}, python)
            print("Short latency and memory diagnostics complete. Starting the two full accuracy controls.", flush=True)
            run_phase("accuracy", ASSIGNMENTS, python)
        # GPU phases finished; release the shared lock before CPU scoring.
        children.clear()
        scorer = subprocess.Popen([sample.BASE_PYTHON, "-u", *common, "--phase", "score"],
            env=dict(env, CUDA_VISIBLE_DEVICES=""), start_new_session=True)
        children.append(scorer)
        if scorer.wait() != 0:
            raise RuntimeError("Accuracy scoring failed; saved predictions remain available")
        summarize_run(args)
    except BaseException as exc:
        stop_children()
        write_json(args.output / "failure.json", {"error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["launch", "preflight", "prepare", "measure", "accuracy", "score"], default="launch")
    parser.add_argument("--source", type=Path,
        default=HDD / "results/humming_full/humming_full_20261106_110354_1541123")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct")
    parser.add_argument("--variant", choices=["uniform_w2", "uniform_w3", MIXED])
    parser.add_argument("--gpu", type=int, choices=[2, 3])
    parser.add_argument("--wait-hours", type=float, default=3)
    args = parser.parse_args()
    if not math.isfinite(args.wait_hours) or args.wait_hours < 0:
        parser.error("wait-hours must be finite and nonnegative")
    if args.phase in ("preflight", "prepare", "measure", "accuracy") and (args.gpu is None or args.variant is None):
        parser.error("GPU phases require a GPU and a variant")
    if args.phase in ("prepare", "accuracy") and args.variant != ASSIGNMENTS[args.gpu]:
        parser.error("Uniform control GPU assignment changed")
    os.chdir(ROOT)
    if args.phase == "prepare":
        sys.path.insert(0, str(ROOT / "AutoGPTQ"))
    if args.phase == "preflight":
        check_runtime(args)
        out = args.output / args.variant
        out.mkdir(exist_ok=True)
        sample.preflight(SimpleNamespace(output=out))
        return
    {"launch": launch, "prepare": prepare, "measure": measure,
     "accuracy": accuracy, "score": score}[args.phase](args)


if __name__ == "__main__":
    main()
