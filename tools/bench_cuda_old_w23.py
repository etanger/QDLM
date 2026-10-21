import argparse
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(REPO_ROOT / "AutoGPTQ"),
)

from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear


@torch.inference_mode()
def benchmark_ms(fn, inner_iterations: int) -> float:
    # Warm-up.
    for _ in range(5):
        fn()

    torch.cuda.synchronize()

    measurements = []

    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(inner_iterations):
            fn()

        end.record()
        torch.cuda.synchronize()

        measurements.append(
            start.elapsed_time(end) / inner_iterations
        )

    return statistics.median(measurements)


def build_quant_linear(
    bits: int,
    dim: int,
    group_size: int,
    kernel_switch_threshold,
    device: torch.device,
) -> QuantLinear:
    layer = QuantLinear(
        bits=bits,
        group_size=group_size,
        infeatures=dim,
        outfeatures=dim,
        bias=False,
        use_cuda_fp16=True,
        kernel_switch_threshold=kernel_switch_threshold,
        trainable=False,
        weight_dtype=torch.float16,
    ).to(device)

    # Packed values do not need to represent a trained layer for a
    # latency benchmark. The same state is used by default and forced
    # variants.
    layer.qweight.zero_()
    layer.qzeros.zero_()
    layer.scales.fill_(0.01)

    layer.eval()

    if not layer.autogptq_cuda_available:
        raise RuntimeError(
            f"W{bits}: AutoGPTQ CUDA extension is unavailable"
        )

    return layer


def speedup(reference_ms: float, candidate_ms: float) -> float:
    return reference_ms / candidate_ms


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dim",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[
            1,
            16,
            64,
            127,
            128,
            256,
            512,
        ],
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device("cuda:0")

    print("torch:", torch.__version__)
    print("torch CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(device))
    print("dimension:", args.dim)
    print("group size:", args.group_size)
    print()

    torch.manual_seed(0)

    fp16_layer = nn.Linear(
        args.dim,
        args.dim,
        bias=False,
        dtype=torch.float16,
        device=device,
    ).eval()

    for bits in [2, 3]:
        default_layer = build_quant_linear(
            bits=bits,
            dim=args.dim,
            group_size=args.group_size,
            kernel_switch_threshold=128,
            device=device,
        )

        forced_layer = build_quant_linear(
            bits=bits,
            dim=args.dim,
            group_size=args.group_size,
            kernel_switch_threshold=False,
            device=device,
        )

        forced_layer.load_state_dict(
            default_layer.state_dict()
        )

        backend_name = getattr(
            default_layer.autogptq_cuda,
            "__name__",
            type(default_layer.autogptq_cuda).__name__,
        )

        print("=" * 100)
        print(
            f"W{bits} | backend={backend_name} | "
            f"default threshold=128"
        )
        print("=" * 100)
        print(
            f"{'M':>6}  "
            f"{'default path':>18}  "
            f"{'FP16 ms':>12}  "
            f"{'default ms':>12}  "
            f"{'forced ms':>12}  "
            f"{'default speedup':>16}  "
            f"{'forced speedup':>15}"
        )

        for m in args.m_values:
            x = torch.randn(
                m,
                args.dim,
                dtype=torch.float16,
                device=device,
            )

            inner_iterations = (
                20
                if m < 128
                else 5
            )

            fp16_ms = benchmark_ms(
                lambda: fp16_layer(x),
                inner_iterations,
            )

            default_ms = benchmark_ms(
                lambda: default_layer(x),
                inner_iterations,
            )

            forced_ms = benchmark_ms(
                lambda: forced_layer(x),
                inner_iterations,
            )

            default_path = (
                "vecquant kernel"
                if m < 128
                else "unpack+matmul"
            )

            print(
                f"{m:6d}  "
                f"{default_path:>18}  "
                f"{fp16_ms:12.4f}  "
                f"{default_ms:12.4f}  "
                f"{forced_ms:12.4f}  "
                f"{speedup(fp16_ms, default_ms):16.3f}x  "
                f"{speedup(fp16_ms, forced_ms):15.3f}x"
            )

            del x

        print()

        del default_layer
        del forced_layer

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
