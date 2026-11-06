"""Paired FP16/Humming latency on the saved 57 x 80 MMLU requests.

Two workers use physical GPUs 2 and 3, respectively. Each worker runs BOTH
variants on its half of the prompts. Cached GPTQ weights are never recalibrated.
"""
import argparse
from collections import Counter
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

from bench_humming_samples import (
    BASE_PYTHON, HDD, HUMMING_REV, ROOT, assert_idle_for_us, cache_identity,
    check_specs, configure_torch, convert_linear,
)

VARIANTS = ("fp16", "sandwich_w23")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_cases(cases):
    counts = Counter(c["task"] for c in cases)
    if len(cases) != 4560 or len(counts) != 57 or set(counts.values()) != {80}:
        raise ValueError(f"Expected 57 subjects x 80 requests; got {len(cases)} requests, {counts}")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate request IDs")
    for case in cases:
        if not case["prompt"] or not case["input_ids"] or not isinstance(case["until"], list):
            raise ValueError("Malformed saved prompt")
    return counts


def split_cases(cases):
    # Alternate within each subject to balance the workloads without moving a
    # prompt between GPUs for its FP16 and Humming measurements.
    ordered = sorted(cases, key=lambda c: (c["task"], c["doc_id"], c["case_id"]))
    return {2: ordered[::2], 3: ordered[1::2]}


def summarize(cases, rows, repeats):
    validate_cases(cases)
    shards = split_cases(cases)
    expected = {(c["case_id"], gpu, variant, repeat)
                for gpu, shard in shards.items() for c in shard
                for variant in VARIANTS for repeat in range(repeats)}
    observed = {}
    for row in rows:
        key = (row["case_id"], row["physical_gpu"], row["variant"], row["repeat"])
        if key not in expected or key in observed:
            raise ValueError(f"Unexpected or duplicate timing row: {key}")
        for metric in ("request_seconds", "generation_seconds"):
            if not math.isfinite(row[metric]) or row[metric] <= 0:
                raise ValueError(f"Invalid timing value: {key}")
        observed[key] = row
    if set(observed) != expected:
        raise ValueError(f"Incomplete benchmark: {len(observed)}/{len(expected)} timing rows")
    by_device = {}
    by_task = {}
    for gpu, shard in shards.items():
        report = {"paired_prompts": len(shard)}
        for metric in ("generation_seconds", "request_seconds"):
            totals = {variant: 0.0 for variant in VARIANTS}
            for case in shard:
                task_totals = by_task.setdefault(case["task"], {}).setdefault(
                    metric, {variant: 0.0 for variant in VARIANTS})
                for variant in VARIANTS:
                    value = statistics.median(observed[case["case_id"], gpu, variant, r][metric]
                                              for r in range(repeats))
                    totals[variant] += value
                    task_totals[variant] += value
            report[metric] = {"sum_of_prompt_medians": totals,
                              "speedup": totals["fp16"] / totals["sandwich_w23"]}
        by_device[f"physical_gpu_{gpu}"] = report
    combined = {}
    for metric in ("generation_seconds", "request_seconds"):
        totals = {v: sum(row[metric]["sum_of_prompt_medians"][v] for row in by_device.values())
                  for v in VARIANTS}
        combined[metric] = {"fp16_seconds": totals["fp16"],
                            "sandwich_w23_seconds": totals["sandwich_w23"],
                            "speedup": totals["fp16"] / totals["sandwich_w23"]}
    for metrics in by_task.values():
        for values in metrics.values():
            values["speedup"] = values["fp16"] / values["sandwich_w23"]
    return {"quant_backend": "humming", "paired_prompts": 4560,
            "timing_rows": len(rows), "repeats_per_variant_per_prompt": repeats,
            "latency": {"aggregation": "ratio of sums of paired per-prompt medians; with repeats=1, ratio of total request times",
                        "by_device": by_device, "combined": combined, "by_task": by_task},
            "timing_note": "Each prompt compares both models on the same GPU. Summed request latency is not parallel job wall time."}


def load_models(torch, args, cache, out):
    from transformers import AutoModel
    payload = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    if payload["identity"] != cache_identity(args.model):
        raise ValueError("GPTQ cache identity changed; no automatic recalibration is allowed")
    specs, state = payload["specs"], payload["state"]
    check_specs(specs)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    prefixes = tuple(s["name"] + "." for s in specs)
    unquant = {k: v for k, v in state.items() if not k.startswith(prefixes)}
    missing, unexpected = model.load_state_dict(unquant, strict=False)
    expected = {s["name"] + "." + k for s in specs
                for k in (["weight", "bias"] if s["has_bias"] else ["weight"])}
    if set(missing) != expected or unexpected:
        raise ValueError("Unquantized state does not match the cached model")
    reports = []
    for i, spec in enumerate(specs):
        assert_idle_for_us()
        attrs = {k: state[spec["name"] + "." + k] for k in ("qweight", "qzeros", "scales", "g_idx")}
        attrs["bias"] = state[spec["name"] + ".bias"] if spec["has_bias"] else None
        layer, report = convert_linear(torch, SimpleNamespace(**spec, **attrs), spec["name"])
        parent, _, leaf = spec["name"].rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, layer)
        reports.append(report)
        if (i + 1) % 16 == 0:
            print(f"GPU {args.gpu}: Humming conversion {i + 1}/128", flush=True)
    write_json(out / "conversion.json", reports)
    write_json(out / "packed_validation.json", payload["packed_validation"])
    del payload, state, unquant, attrs, layer
    gc.collect()
    model.cuda().eval()
    fp = AutoModel.from_pretrained(args.model, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda().eval()
    torch.cuda.empty_cache()
    return {"fp16": fp, "sandwich_w23": model}


def worker(args):
    torch = configure_torch()
    from importlib.metadata import distribution
    from transformers import AutoTokenizer
    from qdlm_benchmark_core import generate_clean
    if int(os.environ["QDLM_PHYSICAL_GPU"]) != args.gpu:
        raise ValueError("Physical GPU assignment changed")
    if str(torch.__version__) != "2.7.1+cu128" or torch.version.cuda != "12.8":
        raise ValueError("Use the existing Humming PyTorch 2.7.1/cu128 environment")
    dist = distribution("humming-kernels")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != HUMMING_REV:
        raise ValueError("Humming revision differs from the successful smoke test")
    manifest = json.loads((args.output / "metadata.json").read_text())
    cases = read_jsonl(args.output / "prompts.jsonl")
    shard = split_cases(cases)[args.gpu]
    out = args.output / f"gpu{args.gpu}"
    out.mkdir(exist_ok=False)
    write_json(out / "runtime.json", {"pid": os.getpid(), "gpu": args.gpu,
        "uuid": os.environ["CUDA_VISIBLE_DEVICES"], "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "humming_version": dist.version,
        "humming_revision": HUMMING_REV, "properties": str(torch.cuda.get_device_properties(0))})
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for case in shard:
        if tokenizer(case["prompt"])["input_ids"] != case["input_ids"]:
            raise ValueError("Prompt tokenization differs from the saved standard requests")
    models = load_models(torch, args, Path(manifest["cache"]), out)
    warmed = set()
    cumulative = {v: 0.0 for v in VARIANTS}
    started = time.perf_counter()

    def generate(case, variant):
        torch.cuda.synchronize()
        request_start = time.perf_counter()
        ids = tokenizer(case["prompt"], return_tensors="pt")["input_ids"].cuda()
        torch.cuda.synchronize()
        gen_start = time.perf_counter()
        output = generate_clean(models[variant], ids, steps=128, gen_length=64, block_length=64)
        torch.cuda.synchronize()
        gen_seconds = time.perf_counter() - gen_start
        tokens = output[0, ids.shape[1]:].cpu().tolist()
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        for stop in case["until"]:
            if stop:
                response = response.split(stop)[0]
        torch.cuda.synchronize()
        return {"generation_seconds": gen_seconds,
                "request_seconds": time.perf_counter() - request_start,
                "tokens": tokens, "response": response}

    def warm_shape(case, variant):
        # One full decoding warmup per model initializes generation operations.
        # Subsequent new M values need model-forward warmup for Humming dispatch,
        # compilation and GEMM/attention initialization, not 128 extra forwards.
        if not warmed:
            generate(case, variant)
            return
        ids = tokenizer(case["prompt"], return_tensors="pt")["input_ids"].cuda()
        x = torch.full((1, ids.shape[1] + 64), 126336, dtype=torch.long, device="cuda")
        x[:, :ids.shape[1]] = ids
        for _ in range(2):
            logits = models[variant](x).logits
            predicted = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
            torch.topk(confidence[0], k=1)
            del logits, predicted, probabilities, confidence
        torch.cuda.synchronize()

    with torch.inference_mode(), (out / "timings.jsonl").open("x") as handle:
        for i, case in enumerate(shard):
            m = len(case["input_ids"]) + 64
            if m not in warmed:
                print(f"GPU {args.gpu}: warming new M={m}; prompt {i + 1}/{len(shard)}", flush=True)
                for variant in VARIANTS:
                    assert_idle_for_us()
                    warm_shape(case, variant)
                    assert_idle_for_us()
                warmed.add(m)
            for repeat in range(args.repeats):
                order = VARIANTS if (i + repeat + args.gpu) % 2 == 0 else VARIANTS[::-1]
                for variant in order:
                    assert_idle_for_us()
                    measurement = generate(case, variant)
                    assert_idle_for_us()
                    row = {"case_id": case["case_id"], "task": case["task"], "doc_id": case["doc_id"],
                           "physical_gpu": args.gpu, "variant": variant, "repeat": repeat,
                           "prompt_tokens": len(case["input_ids"]), "generated_slots": 64, **measurement}
                    print(json.dumps(row), file=handle, flush=True)
                    cumulative[variant] += measurement["request_seconds"]
            progress = {"completed_pairs": i + 1, "total_pairs": len(shard), "gpu": args.gpu,
                        "elapsed_seconds_including_warmup": time.perf_counter() - started,
                        "measured_request_seconds": cumulative,
                        "running_speedup_ratio_of_all_repeats": cumulative["fp16"] / cumulative["sandwich_w23"]}
            write_json(out / "progress.json", progress)
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(shard):
                print(f"GPU {args.gpu}: paired {i + 1}/{len(shard)} | running request speedup "
                      f"{progress['running_speedup_ratio_of_all_repeats']:.3f}x", flush=True)
    write_json(out / "complete.json", {"paired_prompts": len(shard), "repeats": args.repeats,
                                       "warmed_m": sorted(warmed)})


def score(args):
    # Replay saved tokens through the same accuracy protocol. No GPU generation.
    sys.path.insert(0, str(ROOT / "lm-evaluation-harness"))
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from transformers import AutoTokenizer
    from benchmark_sandwich import request_key
    cases = read_jsonl(args.output / "prompts.jsonl")
    rows = read_jsonl(args.output / "timings.jsonl")
    summarize(cases, rows, args.repeats)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    by_id = {case["case_id"]: case for case in cases}
    accuracies = {}
    for variant in VARIANTS:
        answers = {}
        for row in rows:
            if row["variant"] != variant or row["repeat"] != 0:
                continue
            case = by_id[row["case_id"]]
            answer = tokenizer.decode(row["tokens"], skip_special_tokens=False)
            for stop in case["until"]:
                if stop:
                    answer = answer.split(stop)[0]
            answer = tokenizer.decode(tokenizer(answer)["input_ids"], skip_special_tokens=True)
            answers[row["case_id"]] = answer

        class Replay(LM):
            def __init__(self):
                super().__init__()
                self.seen = set()

            def loglikelihood(self, requests):
                raise NotImplementedError("Generation tasks only")

            def loglikelihood_rolling(self, requests):
                raise NotImplementedError("Generation tasks only")

            def generate_until(self, requests):
                keys = [request_key(r) for r in requests]
                if len(set(keys)) != len(keys) or self.seen.intersection(keys):
                    raise ValueError("Repeated evaluation request")
                if not set(keys).issubset(answers):
                    raise ValueError("Evaluation requests changed; saved tokens cannot be scored")
                self.seen.update(keys)
                return [answers[key] for key in keys]

        replay = Replay()
        result = evaluator.simple_evaluate(replay, tasks=["mmlu_generative"], num_fewshot=5,
            limit=80, random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
            fewshot_random_seed=1234, log_samples=False, confirm_run_unsafe_code=True)
        if replay.seen != set(answers) or len(answers) != 4560:
            raise ValueError("Accuracy replay did not cover all 4560 prompts")
        label = "humming" if variant == "sandwich_w23" else "fp16"
        write_json(args.output / f"accuracy_{label}.json", result)
        accuracy = result["results"]["mmlu_generative"]["exact_match,get_response"]
        accuracies[label] = float(accuracy)
        print(f"{label} MMLU accuracy: {accuracy:.2%} (4560 requests)", flush=True)
    summary = json.loads((args.output / "summary.json").read_text())
    summary["mmlu_accuracy"] = accuracies
    summary["accuracy_note"] = "Scored the first timed generation per variant, using the existing evaluator and answer normalization. No extra generation."
    write_json(args.output / "summary.json", summary)


def launch(args):
    from bench_flute_shape_ab import gpu_state, try_lock
    source = json.loads((args.source / "metadata.json").read_text())
    smoke = json.loads((args.source / "summary.json").read_text())
    if smoke["quant_backend"] != "humming" or smoke["latency_prompts"] != 3:
        raise ValueError("Expected the completed Humming smoke test")
    if source["model"] != args.model or source["humming_revision"] != HUMMING_REV:
        raise ValueError("Source model or Humming revision changed")
    cache = Path(source["cache"])
    if not cache.is_file():
        raise FileNotFoundError(f"Packed weight cache missing: {cache}; refusing to recalibrate")
    prompts = Path(source["source_run"]) / "prompts.jsonl"
    original_meta = json.loads((prompts.parent / "metadata.json").read_text())
    expected = {"tasks": "mmlu_generative", "num_fewshot": 5, "limit": 80,
                "steps": 128, "gen_length": 64, "block_length": 64}
    if any(original_meta["args"].get(k) != v for k, v in expected.items()):
        raise ValueError("Saved evaluation setup differs from 57 x 80 MMLU/5-shot/128 steps")
    cases = read_jsonl(prompts)
    counts = validate_cases(cases)
    python = ROOT / ".venv/qdlm-humming/bin/python"
    if not python.is_file():
        raise FileNotFoundError("The tested Humming environment is missing")
    children = []

    def stop_children():
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def interrupt(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with (ROOT / "logs/dlm_sandwich_paired_2gpu.lock").open("a") as lock:
        if not try_lock(lock):
            raise RuntimeError("Another QDLM experiment holds the lock")
        devices, busy = gpu_state()
        if any(busy[devices[gpu]] for gpu in (2, 3)):
            raise RuntimeError("Both physical GPUs 2 and 3 must be idle; launch cancelled")
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "prompts.jsonl").open("x") as handle:
            for case in cases:
                print(json.dumps(case), file=handle)
        paths = [Path(__file__), ROOT / "tools/bench_humming_samples.py",
                 ROOT / "tools/qdlm_benchmark_core.py", ROOT / "tools/qdlm_flute_backend.py"]
        write_json(args.output / "metadata.json", {"model": args.model, "cache": str(cache),
            "source_smoke_test": str(args.source), "source_prompts": str(prompts),
            "source_prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "request_count": len(cases), "requests_per_task": counts, "repeats": args.repeats,
            "physical_gpus": devices, "paired_on_same_gpu": True, "setup": expected,
            "warmup": "one complete generation per variant per GPU; two model forwards plus token selection per subsequent new M, outside timing",
            "calibration": "reuse the existing GPTQ cache; no recalibration"})
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(HUMMING_DISABLE_PARALLEL_BUILD="1", HUMMING_USE_TORCH_STABLE_API="0",
                   HUMMING_CACHE_DIR=str(HDD / "cache/humming"),
                   HUMMING_TMP_DIR=str(HDD / "cache/humming_tmp"),
                   TORCH_EXTENSIONS_DIR=str(HDD / "cache/torch_extensions_humming"),
                   MAX_JOBS="2", OMP_NUM_THREADS="4")
        common = [str(Path(__file__).resolve()), "--output", str(args.output),
                  "--model", args.model, "--repeats", str(args.repeats)]
        print(f"Running 4560 paired prompts on GPUs 2,3; repeats={args.repeats}; cache={cache}", flush=True)
        started = time.perf_counter()
        try:
            for gpu in (2, 3):
                child_env = dict(env, CUDA_VISIBLE_DEVICES=devices[gpu], QDLM_PHYSICAL_GPU=str(gpu))
                child = subprocess.Popen([str(python), "-u", *common, "--phase", "worker", "--gpu", str(gpu)], env=child_env)
                children.append(child)
                print(f"Worker GPU {gpu}: PID {child.pid}", flush=True)
            write_json(args.output / "pids.json", {"parent": os.getpid(), "workers": [p.pid for p in children]})
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("A GPU worker failed; stopping the other worker")
                time.sleep(2)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError("A GPU worker failed")
            rows = []
            for gpu in (2, 3):
                done = json.loads((args.output / f"gpu{gpu}/complete.json").read_text())
                if done["paired_prompts"] != 2280 or done["repeats"] != args.repeats:
                    raise ValueError("Worker completion count mismatch")
                rows.extend(read_jsonl(args.output / f"gpu{gpu}/timings.jsonl"))
            summary = summarize(cases, rows, args.repeats)
            summary["gpu_job_wall_seconds_including_loading_and_warmup"] = time.perf_counter() - started
            with (args.output / "timings.jsonl").open("x") as handle:
                for row in rows:
                    print(json.dumps(row), file=handle)
            write_json(args.output / "summary.json", summary)
            print("Full paired latency complete:", json.dumps(summary["latency"]["combined"], indent=2), flush=True)
            print("Scoring saved outputs on CPU; no additional model generation.", flush=True)
            scoring_env = dict(env, CUDA_VISIBLE_DEVICES="")
            scorer = subprocess.Popen([BASE_PYTHON, "-u", *common, "--phase", "score"], env=scoring_env)
            children.append(scorer)
            if scorer.wait() != 0:
                raise RuntimeError("Latency completed, but CPU accuracy scoring failed; latency summary is preserved")
            write_json(args.output / "complete.json", {"paired_prompts": 4560, "repeats": args.repeats,
                "accuracy_files": ["accuracy_fp16.json", "accuracy_humming.json"], "status": "pass"})
            print(f"Completion verified: 4560 paired prompts and both accuracy files. Results: {args.output}", flush=True)
        except BaseException as exc:
            stop_children()
            write_json(args.output / "failure.json", {"error": str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["launch", "worker", "score"], default="launch")
    parser.add_argument("--source", type=Path,
        default=HDD / "results/humming_samples/humming_samples_20261106_090130")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gpu", type=int, choices=[2, 3])
    args = parser.parse_args()
    if args.repeats < 1 or (args.phase == "worker" and args.gpu is None):
        parser.error("Positive repeats and a worker GPU assignment are required")
    os.chdir(ROOT)
    {"launch": launch, "worker": worker, "score": score}[args.phase](args)


if __name__ == "__main__":
    main()
