# Paired sandwich W2/W3 benchmark

Based on the server source snapshot `21905a0ef44a189deb1888c9b48290832db94e00`.
The new entry point is `run_dlm_sandwich_paired_2gpu.sh`. Existing quantization,
allocation, CUDA kernels, analysis scripts, and evaluation entry points are unchanged.

## What the source audit found

- `AutoGPTQ/quantize.py` loads the model called `model_fp16` in BF16, while the
  GPTQ loader defaults to FP16. The old evaluator also enables FP16 autocast.
  The new benchmark explicitly uses FP16 weights/activations and disables autocast
  for both variants. It establishes a new FP16 baseline; it does not silently
  relabel the historical BF16/AMP measurements.
- Disabling six analysis flags in `llada.py` does not disable the unconditional
  per-block forward hooks, three activation quantiles, norms, variances, maxima,
  top-k statistics, masked cross-entropy, and per-step logging. These costs were
  included in the printed `Time` field.
- The original decoder returns reference-model logits to the quantized model's
  device. That creates an asymmetric cross-GPU transfer in reference runs.
- The current 128-step / 64-slot schedule performs 64 zero-transfer forwards.
  This benchmark preserves all 128 forwards. It does not early-stop or change
  the diffusion schedule to create a speedup.
- `qlinear_cuda_old` dispatches on flattened activation rows, not request count.
  At the default threshold, rows >= 128 use weight unpacking followed by dense
  matmul. A batch of one long prompt can therefore use this fallback. The runtime
  report records the observed shapes and dispatch predicates, and a separate
  profiler trace records executed operators/kernels.
- The uploaded snapshot does not include the two CUDA source modifications that
  appeared in the server's earlier `git status`. The benchmark does not rebuild
  extensions. It records server-side source hashes, local diffs, and hashes/paths
  of the extensions actually loaded.

Historical ratios such as 0.48 / 0.36 = 1.33 describe the reported diagnostic step
times. They do not yet establish the clean, full-generation speedup measured here.

## Protocol

Both variants use the same tokenizer and exact requests built by this checkout's
lm-evaluation-harness: `mmlu_generative`, 5-shot, limit 80 per leaf subject, seed
1234 for few-shot sampling, 128 steps, 64 generated slots, block length 64,
temperature 0, CFG 0, low-confidence remasking, request batch size 1. No chat
template is added. Actual subject and request counts are recorded.

Calibration remains Wikitext-2 train, 128 samples of 2048 tokens, seed 0, GPTQ
group size 128, `desc_act=False`, and the existing sandwich configuration. The
existing packed-model validator must pass. Quantization is done once per launch;
loading, calibration, packing, warmup, and profiling are excluded from inference
times. The code does not assume the current mixed-bit model can be restored with
the stock uniform-bit checkpoint loader.

Before timing, both actual GPU models must produce identical tokens with the old
and clean decoder on one fixed request. The old decoder's diagnostic files are
isolated under `reference_check/`. Failure stops the experiment.

Latency requests are deterministically selected from each subject's standard
80-request slice (one midpoint request per subject by default, normally 57 total).
Each request is measured three times per model with FP16 on logical GPU 0 and W2/W3
on GPU 1, then again after swapping the models. The two variants run sequentially
with alternating order, so they do not contend for GPU work during measurements.
The swap stages W2/W3 through CPU and is not timed. Warmup is repeated after the
swap. Use otherwise idle cards; a swap cannot eliminate unrelated concurrent jobs.

- `generation_seconds`: GPU-resident prompt through all denoising steps to
  synchronized output tokens. CUDA is synchronized before and after the interval.
- `request_seconds`: tokenization, input transfer, generation, output transfer,
  stop-string processing, and final decoding. Model loading and queueing are excluded.
- `amortized_step_seconds`: complete generation time divided by 128. This is not
  an independently timed model-forward kernel latency.
- Peak allocated and reserved bytes include the resident model and live temporary
  tensors on that card. They are allocator measurements, not total system VRAM.

The primary speedup compares each model on the **same GPU and prompts**: sum of
per-prompt FP16 medians divided by sum of per-prompt W2/W3 medians. Results are
reported separately per device and combined. A value below 1 means slower.
Repeated/swap output differences are reported rather than hidden. This is not a
multi-GPU throughput benchmark: each model fits on one GPU, and two cards are used
to control hardware variation.

## Modes

- `MODE=latency` (default): run only the latency subset and profiler. Do not compute
  or report accuracy on this subset. The full standard prompt manifest is saved.
- `MODE=accuracy`: perform the same latency protocol **and** generate both variants'
  predictions for every standard request (80 per subject, normally 4560 per model).
  Reuse subset predictions from assignment A, then score each variant with the
  original harness filters and metrics. Request fingerprints must match exactly
  when replaying predictions for scoring. This is substantially longer.

The measured latency subset and the full accuracy set are explicitly separate.
The script changes neither an existing accuracy result nor the method's bit allocation.
The configured 2.25-bit average is over quantized weight elements, not all model
parameters. Module coverage, remaining dense modules, and packed widths are saved.

## Launch

From the repository root with the existing `qdlm` environment and CUDA extensions:

```bash
RUN_NAME="dlm_paired_latency_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
nohup env MODE=latency RUN_NAME="$RUN_NAME" CUDA_VISIBLE_DEVICES=2,3 \
    bash run_dlm_sandwich_paired_2gpu.sh \
    > "logs/$RUN_NAME.log" 2>&1 < /dev/null &
printf 'PID: %s\nLog: logs/%s.log\n' "$!" "$RUN_NAME"
tail -n 40 -f "logs/$RUN_NAME.log"
```

For full paired accuracy, change `MODE=latency` to `MODE=accuracy` and choose a fresh
run name. `PYTHON_BIN` and `MODEL_PATH` can override the server defaults. Run the
Python entry point directly with `--skip-profile` only if the profiler is unavailable.

The launcher uses a shared lock to prevent duplicate paired benchmarks. Each run
has `.pid` and `.exitcode` files under `logs/`; zero means the entire run finished.
An existing result directory is never overwritten. The lock does not detect other
experiment launchers. Do not rebuild CUDA extensions just to run this benchmark.

## Outputs

Under `results/paired_benchmark/<run-name>/`:

- `summary.json`: per-card and combined speedups, output consistency, and mode.
- `metadata.json`: arguments, true dtypes, device inventory, source/extension
  hashes, local source diffs, quantization coverage, and parity results.
- `timings.jsonl`, `outputs.jsonl`: incremental measurements and generated tokens.
- `prompts.jsonl`: the complete standard request manifest and subset membership.
- `sandwich_plan.json`: allocator output and packed-module validation.
- `backend_routes.json`: observed W2/W3 activation shapes and dispatch routes for
  a median-length, fully masked generation input. This is one forward, not the
  distribution of all timesteps or prompts.
- `profile_fp16.*`, `profile_sandwich_w23.*`: separate one-forward operator tables
  and Chrome traces, outside all latency measurements.
- `accuracy_fp16.json`, `accuracy_sandwich_w23.json`: only in accuracy mode.

## Validation and limits

`python tools/test_qdlm_benchmark.py -v` runs CPU tests against the exact original
decoder function bodies, including the full 128/64 trajectory, multi-block
generation, masks in prompts, and every intermediate model input. It also checks
subset selection and paired statistics. Python syntax and shell syntax are checked.

Development checks used CPU PyTorch 2.14.1. The target environment is CUDA PyTorch
2.7.1; this code has not yet been run on the server's L40S GPUs. The automatic
reference parity gate and backend profiling run on the server. These additions
measure the current method; they do not implement a new fast low-bit kernel or
guarantee that the quantized model will beat FP16.
