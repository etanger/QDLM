"""Retune actual M, then compare FP16, recorded FLUTE, and retuned FLUTE.

Synthetic weights and recorded model shapes. This is an operator diagnostic,
not end-to-end latency or model accuracy. No inference source is modified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import time
from unittest.mock import patch

from bench_flute_shape_ab import (
    ROOT, DEVICES, claim_device, gpu_state, make_source, read_run, try_lock,
)

VARIANTS = ("fp16", "recorded", "exact_m")


def prepare_recorded(layer, flute, m, template_id):
    with patch.object(flute.tune, "_tune", return_value=template_id):
        layer.prepare(m)


def prepare_exact(layer, backend, flute, m):
    # Do not reuse any in-process entry: this experiment must retune actual M.
    key = flute.tune.get_template_key(
        M=m, N=layer.outfeatures, K=layer.infeatures, num_bits=layer.bits,
        group_size=layer.group_size,
        num_sms=flute.utils.get_device_num_sms(layer.scales.device),
        dtype=layer.scales.dtype, legacy=False)
    flute.tune._TEMPLATES.pop(key, None)
    with patch.object(backend, "tuning_rows", lambda rows: rows):
        layer.prepare(m)


def check_output(torch, reference, output):
    diff = output.float() - reference
    relative_l2 = (diff.norm() / reference.norm().clamp_min(1e-12)).item()
    scaled_max = (diff.abs().max() / reference.abs().max().clamp_min(1e-12)).item()
    if not (math.isfinite(relative_l2) and relative_l2 <= 0.003
            and math.isfinite(scaled_max) and scaled_max <= 0.02):
        raise RuntimeError(f"Numerical check failed: L2={relative_l2}, max={scaled_max}")
    return dict(relative_l2=relative_l2, scaled_max=scaled_max)


def measure_three(torch, funcs, sweep, rounds=12):
    # Selection measurements from the upstream tuner are never reported as
    # final timings. This is a fresh paired comparison with balanced order.
    for _ in range(5):
        for fn in funcs.values():
            fn()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    result = {}
    for mode in ("warm", "after_sweep"):
        samples = {name: [] for name in VARIANTS}
        orders = ((0, 1, 2), (2, 1, 0), (1, 2, 0),
                  (0, 2, 1), (2, 0, 1), (1, 0, 2))
        for r in range(rounds):
            for i in orders[r % len(orders)]:
                name = VARIANTS[i]
                fn = funcs[name]
                if mode == "warm":
                    fn()
                    calls = 12
                else:
                    sweep.add_(1)
                    calls = 1
                start.record()
                for _ in range(calls):
                    fn()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / calls)
        times = {name: median(values) for name, values in samples.items()}
        result[mode] = dict(ms=times, samples_ms=samples,
            recorded_speedup=times["fp16"] / times["recorded"],
            exact_m_speedup=times["fp16"] / times["exact_m"],
            tuning_gain=times["recorded"] / times["exact_m"])
    return result


def aggregate(rows, ms):
    result = {}
    for m in ms:
        group = [row for row in rows if row["M"] == m]
        total = {name: sum(row["count"] * row["after_sweep"]["ms"][name]
                           for row in group) for name in VARIANTS}
        result[m] = dict(call_weighted_ms=total,
            recorded_speedup=total["fp16"] / total["recorded"],
            exact_m_speedup=total["fp16"] / total["exact_m"],
            tuning_gain=total["recorded"] / total["exact_m"])
    return result


def run(args, meta, counts, ms, index, uuid):
    import torch
    import qdlm_flute_backend as backend
    flute = backend.require_flute()
    if str(torch.__version__) != str(meta["torch"]) or str(flute.__version__) != str(meta["flute_version"]):
        raise RuntimeError("Use the original FLUTE and PyTorch environment")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda:0")
    prop = torch.cuda.get_device_properties(device)
    sweep_bytes = max(256 * 1024**2, 4 * getattr(prop, "L2_cache_size", 0))
    sweep = torch.zeros(sweep_bytes // 4, dtype=torch.int32, device=device)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    print(f"Physical GPU {index}: {prop.name} | output: {args.output.resolve()}", flush=True)

    def check_occupancy():
        if gpu_state()[1][uuid] - {os.getpid()}:
            raise RuntimeError("Another compute process appeared. Discard this run.")

    recorded_device = next(d for d, p in DEVICES.items() if p == index)
    rows = []
    for (bits, k, n), count in sorted(counts.items()):
        src = make_source(torch, bits, k, n)
        weight = backend.reference_weight(src).T.contiguous().to(device)
        recorded = backend.FluteQuantLinear(src).to(device).eval()
        exact = backend.FluteQuantLinear(src).to(device).eval()
        names = [q["name"] for q in meta["quant_modules"]
                 if (q["bits"], q["infeatures"], q["outfeatures"]) == (bits, k, n)]
        for m in ms:
            check_occupancy()
            ids = {r["template_id"] for name in names for r in meta["flute_templates"][name]
                   if r["actual_m"] == m and r["device"] == recorded_device}
            if len(ids) != 1:
                raise RuntimeError(f"No unique recorded template: W{bits} M={m} K={k} N={n}")
            old_id = next(iter(ids))
            num_sms = flute.utils.get_device_num_sms(device)
            candidates = [tid for tid in flute.utils.get_template_ids(bits)
                          if flute.utils.is_template_supported(M=m, N=n, K=k,
                              num_bits=bits, template_id=tid, num_sms=num_sms)]
            print(f"Tuning {len(rows)+1}/{len(counts)*len(ms)}: W{bits} M={m} K={k} N={n} "
                  f"candidates={len(candidates)}", flush=True)
            prepare_recorded(recorded, flute, m, old_id)
            tune_start = time.monotonic()
            prepare_exact(exact, backend, flute, m)
            tune_seconds = time.monotonic() - tune_start
            check_occupancy()
            x = torch.randn((1, m, k), device=device, dtype=torch.float16,
                            generator=torch.Generator(device=device).manual_seed(721))
            funcs = {"fp16": lambda: torch.nn.functional.linear(x, weight),
                     "recorded": lambda: recorded(x), "exact_m": lambda: exact(x)}
            reference = funcs["fp16"]().float()
            errors = {name: check_output(torch, reference, funcs[name]())
                      for name in ("recorded", "exact_m")}
            del reference
            result = measure_three(torch, funcs, sweep)
            check_occupancy()
            row = dict(bits=bits, M=m, K=k, N=n, count=count,
                recorded_template=old_id, exact_m_template=exact.template_id,
                candidate_count=len(candidates), tuning_seconds=tune_seconds,
                errors=errors, **result)
            rows.append(row)
            with (args.output / "rows.jsonl").open("a") as handle:
                print(json.dumps(row), file=handle)
            a = row["after_sweep"]
            print(f"W{bits} M={m} K={k} N={n}: template {old_id}->{exact.template_id}; "
                  f"old {a['recorded_speedup']:.3f}x, exact-M {a['exact_m_speedup']:.3f}x "
                  f"vs FP16; tuning gain {a['tuning_gain']:.3f}x", flush=True)
        del src, weight, recorded, exact, x, funcs
    check_occupancy()
    totals = aggregate(rows, ms)
    for m, entry in totals.items():
        print(f"M={m} covered linears only: old {entry['recorded_speedup']:.3f}x, "
              f"exact-M {entry['exact_m_speedup']:.3f}x vs FP16; "
              f"tuning gain {entry['tuning_gain']:.3f}x", flush=True)
    summary = dict(source_run=str(args.run.resolve()), physical_gpu=index,
        gpu_uuid=uuid, gpu_name=prop.name, torch=torch.__version__,
        flute_version=flute.__version__, sweep_bytes=sweep_bytes,
        elapsed_seconds=time.monotonic()-started, estimates=totals, rows=rows,
        status="complete", note="Synthetic weights, real shapes. Call-weighted operator "
        "timings only; not end-to-end speed or accuracy. Cache sweep is a proxy. "
        "Exact-M templates are selected by the upstream tuner; final timings are independent.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Results saved to {args.output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=["auto", "2", "3"], default="auto")
    args = parser.parse_args()
    meta, counts, ms, _ = read_run(args.run)
    adapter = "tools/qdlm_flute_backend.py"
    if hashlib.sha256((ROOT / adapter).read_bytes()).hexdigest() != meta["source_sha256"][adapter]:
        raise SystemExit("Local FLUTE adapter differs from the profiled run")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / "flute_shape_ab.lock").open("a") as own, (logs / "dlm_sandwich_paired_2gpu.lock").open("a") as shared:
        if not try_lock(own):
            raise SystemExit("A shape benchmark is already waiting or running")
        index, uuid = claim_device(shared, args.gpu, wait=False)
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        import torch
        with torch.inference_mode():
            run(args, meta, counts, ms, index, uuid)


if __name__ == "__main__":
    main()
