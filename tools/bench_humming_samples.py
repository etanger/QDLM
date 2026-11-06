"""Three real LLaDA requests: same GPTQ W2/W3 weights, Humming versus FP16."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HUMMING_REV = "3f2fa2238cb87fe1ac77aa9d51d3929d143e3c61"
BASE_PYTHON = "/home/zwang97/miniconda3/envs/qdlm/bin/python"
HDD = Path("/workspace/hdd/users/zwang97/QDLM")
CONFIG = ROOT / "configs/qdlm_sandwich_w23_half.json"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def select_cases(path):
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    pool = sorted((c for c in cases if c.get("latency_sample")),
                  key=lambda c: (len(c["input_ids"]), c["case_id"]))
    if len(pool) < 3 or len({c["case_id"] for c in pool}) != len(pool):
        raise ValueError("Expected at least three unique saved latency prompts")
    selected = [pool[round(p * (len(pool) - 1))] for p in (0.1, 0.5, 0.9)]
    if len({c["case_id"] for c in selected}) != 3:
        raise ValueError("The selected prompt IDs must be distinct")
    return selected


def check_specs(specs):
    if Counter(s["bits"] for s in specs) != {2: 64, 3: 64}:
        raise ValueError("Expected 64 W2 and 64 W3 linears")
    if len({s["name"] for s in specs}) != 128:
        raise ValueError("Duplicate quantized module names")
    layer_counts = Counter()
    for s in specs:
        match = re.search(r"(?:^|\.)blocks\.(\d+)\.", s["name"])
        if not match or not 0 <= int(match[1]) < 32:
            raise ValueError(f"Unknown layer: {s['name']}")
        expected = 3 if 8 <= int(match[1]) < 24 else 2
        layer_counts[int(match[1])] += 1
        if s["bits"] != expected or s["group_size"] != 128:
            raise ValueError(f"Wrong allocation: {s}")
    if layer_counts != {i: 4 for i in range(32)}:
        raise ValueError("Expected four quantized linears in every transformer block")
    average = sum(s["bits"] * s["infeatures"] * s["outfeatures"] for s in specs)
    average /= sum(s["infeatures"] * s["outfeatures"] for s in specs)
    if average != 2.5:
        raise ValueError(f"Expected parameter-weighted average 2.5, got {average}")


def cache_identity(model):
    sources = [CONFIG, ROOT / "tools/bench_humming_samples.py"]
    for directory in ("AutoGPTQ/auto_gptq/mixed_precision", "AutoGPTQ/auto_gptq/modeling",
                      "AutoGPTQ/auto_gptq/quantization", "AutoGPTQ/auto_gptq/nn_modules/qlinear"):
        sources.extend(sorted((ROOT / directory).glob("*.py")))
    model_files = []
    for p in sorted(Path(model).iterdir()):
        if p.is_file() and p.suffix in (".json", ".py", ".safetensors", ".bin", ".model"):
            stat = p.stat()
            model_files.append([p.name, stat.st_size, stat.st_mtime_ns])
    return {"format": 1, "model": str(Path(model).resolve()), "model_files": model_files,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
            "calibration": {"dataset": "wikitext-2-raw-v1", "samples": 128,
                            "length": 2048, "seed": 0, "group_size": 128, "desc_act": False}}


def assert_idle_for_us():
    from bench_flute_shape_ab import gpu_state
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    devices, busy = gpu_state()
    if uuid not in devices.values() or "," in uuid:
        raise RuntimeError("Exactly one authorized GPU UUID must be visible")
    others = busy[uuid] - {os.getpid()}
    if others:
        raise RuntimeError(f"Another compute process appeared: {sorted(others)}. Discard this run.")


def configure_torch():
    assert_idle_for_us()
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch


def prepare_cache(args):
    torch = configure_torch()
    if args.cache.exists():
        existing = torch.load(args.cache, map_location="cpu", weights_only=True, mmap=True)
        if existing["identity"] != cache_identity(args.model):
            raise ValueError("Stale cache; use a new cache filename")
        check_specs(existing["specs"])
        print(f"Using packed weight cache: {args.cache}", flush=True)
        return
    sys.path.insert(0, str(ROOT / "AutoGPTQ"))
    import numpy as np
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from datasets import load_dataset
    from transformers import AutoTokenizer
    os.environ["QDLM_SANDWICH_CONFIG"] = str(CONFIG)
    os.environ["QDLM_SANDWICH_PLAN_OUT"] = str(args.output / "sandwich_plan.json")
    os.environ["GPTQ_DUMP_CALIB_H"] = "0"
    print("Preparing original GPTQ weights (not timed inference); this first run is not a quick step.", flush=True)
    qmodel = AutoGPTQForCausalLM.from_pretrained(args.model,
        BaseQuantizeConfig(bits=2, group_size=128, desc_act=False),
        trust_remote_code=True, torch_dtype=torch.float16)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    encoded = tokenizer("\n\n".join(train["text"]), return_tensors="pt").input_ids
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    examples = []
    for _ in range(128):
        start = random.randint(0, encoded.shape[1] - 2048 - 1)
        ids = encoded[:, start:start + 2048]
        examples.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    qmodel.quantize(examples)
    specs = []
    for name, module in qmodel.model.named_modules():
        if not hasattr(module, "qweight"):
            continue
        if not type(module).__module__.endswith("qlinear_cuda_old"):
            raise ValueError(f"Unsupported source backend: {type(module)}")
        specs.append(dict(name=name, bits=module.bits, infeatures=module.infeatures,
                          outfeatures=module.outfeatures, group_size=module.group_size,
                          has_bias=module.bias is not None))
    check_specs(specs)
    if qmodel._sandwich_packed_summary["status"] != "pass":
        raise ValueError("Packed weight validation failed")
    assert_idle_for_us()
    state = qmodel.model.cpu().state_dict()
    payload = {"identity": cache_identity(args.model), "specs": specs, "state": state,
               "packed_validation": qmodel._sandwich_packed_summary}
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.cache.with_suffix(f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, args.cache)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Packed weight cache saved: {args.cache}", flush=True)


def convert_linear(torch, source, name):
    from humming.layer import HummingLayer
    from qdlm_flute_backend import reference_weight
    # This shared helper only unpacks GPTQ codes; it does not import FLUTE.
    weight = reference_weight(source).T.contiguous().cuda()
    layer = HummingLayer(shape_n=source.outfeatures, shape_k=source.infeatures,
        weight_config=dict(quant_method="gptq", bits=source.bits, group_size=128,
                           desc_act=False, sym=True),
        has_bias=source.bias is not None, torch_dtype=torch.float16).cuda()
    raw = {k: getattr(source, k).cuda() for k in ("qweight", "qzeros", "scales", "g_idx")}
    if source.bias is not None:
        raw["bias"] = source.bias.cuda()
    missing, unexpected = layer.load_state_dict(raw, strict=False)
    if set(missing) != {"locks"} or unexpected:
        raise ValueError(f"Humming state mismatch: {missing}, {unexpected}")
    layer.transform()
    layer.eval()
    x = torch.randn(32, source.infeatures, device="cuda", dtype=torch.float16,
                    generator=torch.Generator(device="cuda").manual_seed(123))
    with torch.inference_mode():
        ref = torch.nn.functional.linear(x, weight, raw.get("bias")).float()
        actual = layer(x).float()
        delta = actual - ref
        rel = (delta.norm() / ref.norm().clamp_min(1e-12)).item()
        mx = (delta.abs().max() / ref.abs().max().clamp_min(1e-12)).item()
    if not (rel <= 0.003 and mx <= 0.02):
        raise ValueError(f"Humming numerical check failed for {name}: {rel=}, {mx=}")
    class Linear(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, inputs):
            return self.inner(inputs.reshape(-1, source.infeatures).contiguous()).reshape(
                *inputs.shape[:-1], source.outfeatures)
    record = dict(name=name, bits=source.bits, relative_l2=rel, relative_max=mx,
                  backend="humming", activation_dtype="float16",
                  kernel_config=str(layer.humming_config))
    return Linear(layer), record


def preflight(args):
    torch = configure_torch()
    from bench_flute_shape_ab import make_source
    print("Checking Humming W2/W3 compilation and numerical compatibility (no latency benchmark).", flush=True)
    reports = []
    for bits in (2, 3):
        _, report = convert_linear(torch, make_source(torch, bits, 512, 512), f"preflight_w{bits}")
        reports.append(report)
    save_json(args.output / "preflight.json", reports)
    print("Humming preflight passed.", flush=True)


def benchmark(args):
    torch = configure_torch()
    from transformers import AutoModel, AutoTokenizer
    from qdlm_benchmark_core import generate_clean, summarize_latency
    payload = torch.load(args.cache, map_location="cpu", weights_only=True, mmap=True)
    if payload["identity"] != cache_identity(args.model):
        raise ValueError("Cache identity changed")
    specs, state = payload["specs"], payload["state"]
    check_specs(specs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases = select_cases(args.run / "prompts.jsonl")
    for case in cases:
        if tokenizer(case["prompt"])["input_ids"] != case["input_ids"]:
            raise ValueError("Saved prompt tokenization changed")
    print("Loading model and converting the 128 cached GPTQ linears to Humming.", flush=True)
    qmodel = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                     torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    prefixes = tuple(s["name"] + "." for s in specs)
    unquant = {k: v for k, v in state.items() if not k.startswith(prefixes)}
    missing, unexpected = qmodel.load_state_dict(unquant, strict=False)
    expected = {s["name"] + "." + k for s in specs
                for k in (["weight", "bias"] if s["has_bias"] else ["weight"])}
    if set(missing) != expected or unexpected:
        raise ValueError(f"Unquantized state mismatch: {missing}, {unexpected}")
    reports = []
    for i, spec in enumerate(specs):
        assert_idle_for_us()
        attrs = {k: state[spec["name"] + "." + k] for k in ("qweight", "qzeros", "scales", "g_idx")}
        attrs["bias"] = state[spec["name"] + ".bias"] if spec["has_bias"] else None
        layer, report = convert_linear(torch, SimpleNamespace(**spec, **attrs), spec["name"])
        parent, _, leaf = spec["name"].rpartition(".")
        setattr(qmodel.get_submodule(parent) if parent else qmodel, leaf, layer)
        reports.append(report)
        print(f"Humming conversion {i + 1}/128: {spec['name']} W{spec['bits']}", flush=True)
    save_json(args.output / "conversion.json", reports)
    packed_validation = payload["packed_validation"]
    del payload, state, unquant, attrs, layer
    gc.collect()
    qmodel.cuda().eval()
    fpmodel = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda().eval()
    models = {"fp16": fpmodel, "sandwich_w23": qmodel}
    save_json(args.output / "metadata.json", dict(
        model=args.model, source_run=str(args.run), physical_gpu=os.environ["QDLM_PHYSICAL_GPU"],
        gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"], torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=str(torch.cuda.get_device_properties(0)), humming_revision=HUMMING_REV,
        cache=str(args.cache), packed_validation=packed_validation,
        quant_backend="humming", steps=128, gen_length=64, block_length=64,
        temperature=0, cfg_scale=0, remasking="low_confidence", all_forward_calls_preserved=True,
        repeats=3, warmup_per_prompt_per_variant=1, accuracy_computed=False,
        excluded_from_timing=["loading", "quantization", "conversion", "JIT compilation", "warmup", "file writes"],
        selection="10th/50th/90th length percentile of the previous 57 latency prompts"))
    with (args.output / "prompts.jsonl").open("w") as f:
        for case in cases:
            print(json.dumps(case), file=f)

    def generate(case, variant):
        torch.cuda.synchronize()
        started = time.perf_counter()
        ids = tokenizer(case["prompt"], return_tensors="pt")["input_ids"].cuda()
        torch.cuda.synchronize()
        gen_started = time.perf_counter()
        tokens = generate_clean(models[variant], ids, steps=128, gen_length=64, block_length=64)
        torch.cuda.synchronize()
        generated = time.perf_counter() - gen_started
        new_tokens = tokens[0, ids.shape[1]:].cpu().tolist()
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        for stop in case["until"]:
            if stop:
                response = response.split(stop)[0]
        torch.cuda.synchronize()
        requested = time.perf_counter() - started
        return dict(generation_seconds=generated, request_seconds=requested,
                    tokens=new_tokens, response=response)

    rows = []
    with torch.inference_mode():
        for c, case in enumerate(cases):
            print(f"Prompt {c + 1}/3: {case['task']} doc={case['doc_id']} M={len(case['input_ids']) + 64}", flush=True)
            for variant in models:
                assert_idle_for_us()
                print(f"Warmup: {variant} (includes first-use compilation)", flush=True)
                generate(case, variant)
            for repeat in range(3):
                order = ("fp16", "sandwich_w23") if (c + repeat) % 2 == 0 else ("sandwich_w23", "fp16")
                for variant in order:
                    assert_idle_for_us()
                    measured = generate(case, variant)
                    assert_idle_for_us()
                    row = dict(phase="latency", case_id=case["case_id"], device="cuda:0",
                               physical_gpu=os.environ["QDLM_PHYSICAL_GPU"], variant=variant,
                               repeat=repeat, prompt_tokens=len(case["input_ids"]), generated_slots=64,
                               **measured)
                    rows.append(row)
                    with (args.output / "timings.jsonl").open("a") as f:
                        print(json.dumps(row), file=f)
                    print(f"{variant} repeat={repeat}: generation={measured['generation_seconds']:.3f}s "
                          f"request={measured['request_seconds']:.3f}s", flush=True)
            partial = summarize_latency([r for r in rows if r["case_id"] == case["case_id"]])
            print(f"Prompt speedup: {partial['combined']['request_seconds']['speedup']:.3f}x", flush=True)
    summary = {"quant_backend": "humming", "latency_prompts": 3, "latency": summarize_latency(rows),
               "note": "Three-prompt end-to-end latency smoke test, not accuracy or full-benchmark evidence."}
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results saved to {args.output}", flush=True)


def orchestrate(args):
    from bench_flute_shape_ab import claim_device
    old = json.loads((args.run / "metadata.json").read_text())
    if any(old["args"].get(k) != v for k, v in (("steps", 128), ("gen_length", 64), ("block_length", 64))):
        raise ValueError("Unexpected previous generation settings")
    if str(Path(old["args"]["model"]).resolve()) != str(Path(args.model).resolve()):
        raise ValueError("Model differs from saved prompts")
    select_cases(args.run / "prompts.jsonl")
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache = HDD / "cache/gptq" / (hashlib.sha256(json.dumps(cache_identity(args.model), sort_keys=True).encode()).hexdigest()[:20] + ".pt")
    lock_path = ROOT / "logs/dlm_sandwich_paired_2gpu.lock"
    with lock_path.open("a") as lock:
        index, uuid = claim_device(lock, args.gpu, False)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(CUDA_VISIBLE_DEVICES=uuid, QDLM_PHYSICAL_GPU=str(index),
                   HUMMING_DISABLE_PARALLEL_BUILD="1", HUMMING_USE_TORCH_STABLE_API="0",
                   HUMMING_CACHE_DIR=str(HDD / "cache/humming"),
                   HUMMING_TMP_DIR=str(HDD / "cache/humming_tmp"),
                   TORCH_EXTENSIONS_DIR=str(HDD / "cache/torch_extensions_humming"),
                   PIP_CACHE_DIR=str(HDD / "cache/pip"), MAX_JOBS="2", OMP_NUM_THREADS="4")
        print(f"Selected physical GPU {index}; the other GPU is not used. Output: {args.output}", flush=True)
        subprocess.run(["bash", str(ROOT / "scripts/setup_qdlm_humming_samples.sh")], env=env, check=True)
        humming_python = str(ROOT / ".venv/qdlm-humming/bin/python")
        common = [str(Path(__file__).resolve()), "--run", str(args.run), "--output", str(args.output),
                  "--model", args.model, "--cache", str(args.cache)]
        phases = [("preflight", humming_python), ("prepare", BASE_PYTHON), ("benchmark", humming_python)]
        for phase, executable in phases:
            started = time.perf_counter()
            print(f"Starting phase: {phase}", flush=True)
            subprocess.run([executable, "-u", *common, "--phase", phase], env=env, check=True)
            print(f"Phase {phase} completed in {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["launch", "preflight", "prepare", "benchmark"], default="launch")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct")
    parser.add_argument("--gpu", choices=["auto", "2", "3"], default="auto")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    started = time.perf_counter()
    try:
        {"launch": orchestrate, "preflight": preflight, "prepare": prepare_cache,
         "benchmark": benchmark}[args.phase](args)
    except Exception as exc:
        if args.phase == "launch" and args.output.is_dir():
            save_json(args.output / "failure.json", {"error": str(exc), "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
