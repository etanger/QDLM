"""Paired single-GPU FP16 versus FLUTE diagnostic using shapes from a completed run."""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("fp16", "sandwich_w23")
DEVICES = {"cuda:0": 2, "cuda:1": 3}
GROUP_SIZE = 128  # Same symmetric GPTQ recipe as benchmark_sandwich.py.


def read_run(run):
    def load(name):
        return json.loads((run / name).read_text())
    meta, summary, routes = map(load, ("metadata.json", "summary.json", "backend_routes.json"))
    if summary["quant_backend"] != "flute" or meta["visible_devices"] != "2,3":
        raise ValueError("Expected a FLUTE run on physical GPUs 2,3")
    modules = {v["name"]: v for v in meta["quant_modules"]}
    counts = Counter()
    for r in routes["records"]:
        q = modules[r["name"]]
        if r["route"] != "flute_qgemm_fused_lut_dequantization" or r["bits"] != q["bits"]:
            raise ValueError("Profile route or bit width does not match the saved module")
        counts[q["bits"], q["infeatures"], q["outfeatures"]] += 1
    if {b for b, _, _ in counts} != {2, 3}:
        raise ValueError("Expected both W2 and W3")
    lengths, saved = {}, defaultdict(list)
    with (run / "timings.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["phase"] != "latency":
                continue
            key = row["case_id"]
            m = row["prompt_tokens"] + row["generated_slots"]
            if lengths.setdefault(key, m) != m:
                raise ValueError("Inconsistent prompt length")
            saved[key, row["device"], row["variant"]].append(row["generation_seconds"])
    if len(lengths) != summary["latency_prompts"]:
        raise ValueError("Incomplete latency sample")
    key = routes["case_id"]
    profiled = lengths[key]
    if {r["m"] for r in routes["records"]} != {profiled}:
        raise ValueError("Profile shape disagrees with timings")
    ms = sorted(lengths.values())
    selected = sorted({profiled, *(ms[round(p * (len(ms) - 1))] for p in (0.1, 0.5, 0.9))})
    print(f"Observed rows M: {selected} | profiled M={profiled} | {sum(counts.values())} calls")
    paired = {}
    for dev, physical in DEVICES.items():
        obs = [saved[key, dev, v] for v in VARIANTS]
        if any(len(v) != meta["args"]["repeats"] for v in obs):
            raise ValueError(f"Incomplete paired measurements on {dev}")
        fp, low = map(median, obs)
        paired[str(physical)] = dict(fp16_seconds=fp, flute_seconds=low, speedup=fp / low)
        print(f"Saved profiled prompt, GPU {physical}: FP16 {fp:.3f}s, FLUTE {low:.3f}s, {fp/low:.3f}x")
    return meta, counts, selected, paired


def gpu_state():
    def query(kind, fields):
        data = subprocess.check_output(["nvidia-smi", f"--query-{kind}={fields}",
                                       "--format=csv,noheader,nounits"], text=True, timeout=20)
        return [[s.strip() for s in row] for row in csv.reader(io.StringIO(data)) if row]
    devices = {int(i): u for i, u in query("gpu", "index,uuid") if i in ("2", "3")}
    if set(devices) != {2, 3} or len(set(devices.values())) != 2:
        raise RuntimeError("Cannot identify both physical GPUs 2 and 3")
    busy = defaultdict(set)
    for uuid, pid in query("compute-apps", "gpu_uuid,pid"):
        busy[uuid].add(int(pid))
    return devices, busy


def try_lock(handle):
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def claim_device(shared_lock, choice, wait):
    allowed = (2, 3) if choice == "auto" else (int(choice),)
    while True:
        if try_lock(shared_lock):
            devices, busy = gpu_state()
            for index in allowed:
                if not busy[devices[index]]:
                    return index, devices[index]
            fcntl.flock(shared_lock, fcntl.LOCK_UN)
        if not wait:
            raise SystemExit("No idle authorized device or QDLM lock is busy")
        print("Waiting for an idle authorized GPU and the QDLM lock. Next check in 3 hours.", flush=True)
        time.sleep(10800)


def make_source(torch, bits, k, n):
    # Uniform random packed codes have the same layout as GPTQ. The materialized
    # FP16 control uses exactly these codes and scales; no trained weights needed.
    g = torch.Generator().manual_seed(6000 + bits)
    word = sum(((1 << (bits - 1)) - 1) << (bits * i) for i in range(32))
    words = [(word >> (32 * i)) & 0xffffffff for i in range(bits)]
    zeros = torch.tensor([i if i < 2**31 else i - 2**32 for i in words], dtype=torch.int32)
    groups = k // GROUP_SIZE
    return SimpleNamespace(bits=bits, infeatures=k, outfeatures=n, group_size=GROUP_SIZE,
        qweight=torch.randint(-2**31, 2**31 - 1, (k // 32 * bits, n), dtype=torch.int32, generator=g),
        qzeros=zeros.repeat(n // 32).expand(groups, -1),
        g_idx=torch.arange(k, dtype=torch.int32) // GROUP_SIZE,
        scales=(0.005 + 0.01 * torch.rand(groups, n, generator=g)).half(), bias=None)


def measure_pair(torch, dense_fn, flute_fn, sweep, rounds=8):
    funcs = (dense_fn, flute_fn)
    for _ in range(5):
        for fn in funcs:
            fn()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    end.record()
    end.synchronize()
    result = {}
    for mode in ("warm", "after_sweep"):
        measurements = [[], []]
        for r in range(rounds):
            for i in ((0, 1) if r % 2 == 0 else (1, 0)):
                fn = funcs[i]
                if mode == "warm":
                    fn()
                    calls = 12
                else:
                    sweep.add_(1)  # Best-effort eviction, outside timed events.
                    calls = 1
                start.record()
                for _ in range(calls):
                    fn()
                end.record()
                end.synchronize()
                measurements[i].append(start.elapsed_time(end) / calls)
        fp, q = map(median, measurements)
        result[mode] = dict(fp16_ms=fp, flute_ms=q, speedup=fp/q, samples_ms=measurements)
    return result


def run_bench(args, meta, counts, ms, paired, index, uuid):
    import torch
    from qdlm_flute_backend import FluteQuantLinear, reference_weight, require_flute
    flute = require_flute()
    if str(flute.__version__) != str(meta["flute_version"]) or str(torch.__version__) != str(meta["torch"]):
        raise RuntimeError("Use the FLUTE and PyTorch environment from the original run")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible GPU")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda:0")
    prop = torch.cuda.get_device_properties(device)
    sweep_bytes = max(256 * 1024**2, 4 * getattr(prop, "L2_cache_size", 0))
    sweep = torch.zeros(sweep_bytes // 4, dtype=torch.int32, device=device)
    out = ROOT / "results/flute_shape_ab" / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    out.mkdir(parents=True, exist_ok=False)
    def check_occupancy():
        if gpu_state()[1][uuid] - {os.getpid()}:
            raise RuntimeError("Another compute process appeared. Discard this run.")
    template_log = meta["flute_templates"]
    quant_modules = meta["quant_modules"]
    recorded_dev = next(d for d, p in DEVICES.items() if p == index)
    print(f"Physical GPU {index}: {prop.name} | output: {out}", flush=True)
    rows = []
    for (b, k, n), count in sorted(counts.items()):
        src = make_source(torch, b, k, n)
        w = reference_weight(src).T.contiguous().to(device)
        layer = FluteQuantLinear(src).to(device).eval()
        names = [q["name"] for q in quant_modules if (q["bits"], q["infeatures"], q["outfeatures"]) == (b, k, n)]
        for m in ms:
            check_occupancy()
            expected_ids = {rec["template_id"] for name in names for rec in template_log[name]
                            if rec["actual_m"] == m and rec["device"] == recorded_dev}
            if len(expected_ids) != 1:
                raise RuntimeError(f"No unique recorded template for W{b} M={m} K={k} N={n}: {expected_ids}")
            # Replay only the tuner's saved decision. The original adapter checks
            # support and uses its real packer. No patch is active in a timed call.
            with patch.object(flute.tune, "_tune", return_value=next(iter(expected_ids))):
                layer.prepare(m)
            x = torch.randn(1, m, k, device=device, dtype=torch.float16,
                            generator=torch.Generator(device=device).manual_seed(721))
            dense_fn = lambda: torch.nn.functional.linear(x, w)
            flute_fn = lambda: layer(x)
            ref, test = dense_fn().float(), flute_fn().float()
            delta = test - ref
            rel = (delta.norm() / ref.norm().clamp_min(1e-12)).item()
            mx = (delta.abs().max() / ref.abs().max().clamp_min(1e-12)).item()
            if not (math.isfinite(rel) and rel <= 0.003 and math.isfinite(mx) and mx <= 0.02):
                raise RuntimeError(f"Numerical check failed: relative L2={rel}, scaled max={mx}")
            del ref, test, delta
            result = measure_pair(torch, dense_fn, flute_fn, sweep)
            check_occupancy()
            row = dict(bits=b, M=m, K=k, N=n, count=count, template=layer.template_id,
                       relative_l2=rel, scaled_max=mx, **result)
            rows.append(row)
            with (out / "rows.jsonl").open("a") as f:
                print(json.dumps(row), file=f)
            a = row["after_sweep"]
            print(f"W{b} M={m} K={k} N={n} calls={count}: warm {row['warm']['speedup']:.3f}x; "
                  f"after sweep FP16 {a['fp16_ms']:.3f}ms, FLUTE {a['flute_ms']:.3f}ms = {a['speedup']:.3f}x", flush=True)
        del layer, src, w, x
    estimates = {}
    for m in ms:
        times = [sum(r["count"] * r["after_sweep"][f"{v}_ms"] for r in rows if r["M"] == m)
                 for v in ("fp16", "flute")]
        estimates[m] = times[0] / times[1]
        print(f"M={m} covered linears only (call-weighted, after sweep): {estimates[m]:.3f}x")
    (out / "summary.json").write_text(json.dumps(dict(
        source_run=str(args.run.resolve()), physical_gpu=index, gpu_uuid=uuid, gpu_name=prop.name,
        torch=torch.__version__, sweep_bytes=sweep_bytes, profiled_prompt_generation=paired,
        estimates=estimates, rows=rows, note="Synthetic weights, real shapes and templates. "
        "Cache sweep is a proxy; these are not end-to-end timings or accuracy."), indent=2) + "\n")
    print(f"Results saved to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu", choices=["auto", "2", "3"], default="auto")
    parser.add_argument("--wait", action="store_true", help="Retry every three hours when busy")
    parser.add_argument("--plan-only", action="store_true", help="Read artifacts without querying CUDA")
    args = parser.parse_args()
    meta, counts, ms, paired = read_run(args.run)
    if args.plan_only:
        return
    adapter = "tools/qdlm_flute_backend.py"
    if hashlib.sha256((ROOT / adapter).read_bytes()).hexdigest() != meta["source_sha256"][adapter]:
        raise SystemExit("Local FLUTE adapter differs from the profiled run")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / "flute_shape_ab.lock").open("a") as own, (logs / "dlm_sandwich_paired_2gpu.lock").open("a") as shared:
        if not try_lock(own):
            raise SystemExit("A shape benchmark is already waiting or running")
        index, uuid = claim_device(shared, args.gpu, args.wait)
        # Set visibility before the first torch import or CUDA initialization.
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        import torch
        with torch.inference_mode():
            run_bench(args, meta, counts, ms, paired, index, uuid)


if __name__ == "__main__":
    main()
