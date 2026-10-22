# Executable avg2 sandwich GPTQ

This extends the existing QDLM sandwich method to W1. It is intended for
accuracy experiments at an exact parameter-weighted 2-bit budget over the
selected quantized weights. It does not claim end-to-end acceleration.

## Editable configurations

For the current 32 equal-size transformer blocks:

| Config | Layer allocation | Average selected weight bits |
| --- | --- | --- |
| `configs/qdlm_sandwich_w13_avg2.json` (default) | 0-7 W1; 8-23 W3; 24-31 W1 | 2.0 |
| `configs/qdlm_sandwich_w123_avg2.json` | 0-3 W1; 4-11 W2; 12-19 W3; 20-27 W2; 28-31 W1 | 2.0 |

Both use `stage: "w123"`, meaning executable W1/W2/W3 support. W2 transitions
may have zero width. `num_w3_layers` controls the protected middle interval.
`max_center_shift: 0` centers it; W1 is allocated from the outer edges inward.
The planner uses actual selected parameter counts, not an unweighted layer
average. If a whole-layer allocation cannot achieve exactly 2 bits, it fails
instead of rounding and claiming avg2. `stage1_target_average_bits` controls
the intermediate W2/W3 core selection; it is not the final executed budget.

The old `configs/qdlm_sandwich_w23.json` remains W2/W3 at 2.25 bits.
The selected module names in `modeling/llada.py` are unchanged. Embeddings,
the output head, and unmatched/unselected linear modules remain floating
point. In particular, avg2 does not mean the entire stored model is 2 bits
per parameter. The runner lists every quantized and unquantized linear.

## W1 implementation

`quantization/quantizer.py` fits two levels per output-channel/input-group:
`-alpha` and `+alpha`, with `alpha = mean(abs(weights))`. The affine GPTQ
representation is `scale = 2 * alpha`, `zero = 0.5`. Zero inputs select the
positive level. A small positive scale floor handles empty-valued groups.
GPTQ's Hessian-based sequential error compensation remains active. There
are no extra residual weights, high-precision outliers, or ternary values.
This is an initial binary GPTQ method, not a claim of lossless W1 conversion.

`nn_modules/qlinear/qlinear_binary.py` stores 32 signs per int32 and one
alpha per group/channel. The zero point is implicit. Its forward unpacks
the signs and performs a floating-point matmul; it is a portable reference
backend, not a fused binary GPU kernel. It works without recompiling CUDA.
W2/W3 retain their existing backends. Per-module backend dispatch prevents
W1 from being silently packed at a global 2-bit setting.

This runner quantizes and evaluates in one process. It does not produce a
Hugging Face `from_quantized` checkpoint; mixed-bit checkpoint reload is
outside this entry point. The old paired latency runner still expects W2/W3.
Use the new accuracy runner for these configurations.

## Run on the server

Use the existing `qdlm` environment. The shell entry point forces physical
GPUs 2 and 3. Calibration uses logical cuda:0 (physical GPU 2), then the
quantized model runs on logical cuda:1 (physical GPU 3). This is sequential
calibration and inference, not two-GPU model parallelism. No FP16 reference
model drives quantized generation.

```bash
cd /workspace/nvme/zwang97/QDLM
RUN_NAME="dlm_avg2_w13_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="$PWD/logs/$RUN_NAME.log"
touch "$LOG"
nohup env RUN_NAME="$RUN_NAME" \
    bash run_dlm_sandwich_avg2_2gpu.sh \
    > "$LOG" 2>&1 < /dev/null &
echo "PID: $!"
echo "Log: $LOG"
tail -n 60 -F "$LOG"
```

Ctrl+C stops `tail`; the nohup experiment continues. The lock is shared with
the paired benchmark launcher to prevent accidental overlapping runs.

To use W1/W2/W3 transitions, set the config on the launch command:

```bash
nohup env RUN_NAME="$RUN_NAME" \
    CONFIG=configs/qdlm_sandwich_w123_avg2.json \
    bash run_dlm_sandwich_avg2_2gpu.sh \
    > "$LOG" 2>&1 < /dev/null &
```

Choose a new `RUN_NAME` for every run. Other editable values are at the top
of the shell file; its final Python command is the full experiment command.
The default is MMLU generative, 5-shot, limit 80 per subject (4560 requests),
128 denoising forwards, generation/block length 64, request batch size 1,
temperature 0, and all steps use the quantized model. There are no latency
repeats, GPU swaps, profiler passes, or prerequisite sensitivity experiments.

Calibration remains 128 Wikitext-2 training segments of length 2048, seed 0,
group size 128, symmetric GPTQ, and no activation ordering. Decoding reuses
`qdlm_benchmark_core.generate_clean` from the paired benchmark. Evaluation
uses FP16 activations with autocast disabled and the same prompt seeds and
stop processing as that benchmark. This differs from older BF16/autocast
analysis scripts, so comparisons should record the dtype and runner.

For a matching FP16 accuracy baseline in a separate run, pass `--variant fp16`
to the shell script. It skips quantization and uses the same decoding/scoring.

## Outputs and checks

`results/avg2_accuracy/<RUN_NAME>/` contains:

- `sandwich_plan.json`: actual executed bits and packed-module validation.
- `metadata.json`: config, source revision, devices, coverage, byte counts,
  and request counts. Quantization metadata and unquantized tensors are
  included in the stored-tensor effective-bit report; forward temporaries
  are not included in that storage metric.
- `outputs.jsonl`: predictions written and flushed after every request.
- `accuracy.json`: harness accuracy metrics, written after evaluation.

`logs/<RUN_NAME>.pid` is the launcher PID. `logs/<RUN_NAME>.exitcode` records
the final exit code. A zero exit status alone does not establish accuracy.

CPU implementation checks (no model download or GPU experiment):

```bash
python tools/test_qdlm_avg2.py -v
python tools/test_qdlm_benchmark.py -v
```

These cover the binary numerical levels, real GPTQ compensation and packing,
group indices with activation ordering, signed int32 packing and padding,
mixed W1/W2/W3 forward outputs, budget enforcement, and the existing decoder.
Full LLaDA accuracy and GPU execution still require the server experiment.
