# W2/W3 half-and-half FLUTE speed experiment

This is an experimental execution backend for the existing GPTQ sandwich
weights: layers 0-7 and 24-31 W2, layers 8-23 W3, FP16 activations. The average
over selected quantized weights is 2.5 bits. Quantized module coverage and
calibration/decoding settings match the half-and-half accuracy experiment.
No model-wide 2.5-bit claim or GPU speedup is established by adding this code.

## Backend and installation

The adapter uses [FLUTE](https://github.com/HanGuo97/flute) at commit
`9eb83a12d56949bbe7fe9c836ba97a67bd1e3761` and CUTLASS v3.4.1. FLUTE's source
build provides both W2 and W3, so this attempt needs one backend. Its LUT maps
GPTQ integer codes to `code - zero`; the existing group scales are transposed
to FLUTE's layout. There is no new quantizer, rotation, activation quantization,
outlier bypass, or dense-weight cache in forward. Full floating-point matrices
are not materialized in the FLUTE forward path.

`scripts/setup_qdlm_flute.sh` builds in `.venv/qdlm-flute`, which inherits the
existing qdlm environment's packages. It does not replace its PyTorch or install
vLLM. CUDA/nvcc must match `torch.version.cuda`. Source checkouts are under
`build/`; the only upstream setup change redirects CUTLASS to that local path.
Compilation defaults to two CPU jobs. GPU architecture is read only from
physical GPUs 2 and 3. The script can be run independently for troubleshooting.

The pinned upstream tuner chooses templates for nearby power-of-two row counts.
The actual input is neither truncated nor padded by this adapter. The selected
template is checked against the actual shape. Tuning and repacking occur before
the timer; an unprepared forward raises instead of silently tuning in a sample.
The fused kernel uses FLUTE's Stream-K reduction, whose accumulation order can
change FP16 results. Kernel output must pass finite checks, relative L2 error
at most 0.003 and max error divided by reference max magnitude at most 0.02.

## Background run

```bash
(
    set -e
    cd /workspace/nvme/zwang97/QDLM
    RUN_NAME="dlm_w23_flute_$(date +%Y%m%d_%H%M%S)"
    mkdir -p logs
    LOG="$PWD/logs/$RUN_NAME.log"
    touch "$LOG"
    nohup env RUN_NAME="$RUN_NAME" MODE=latency \
        bash run_dlm_sandwich_w23_flute_2gpu.sh \
        > "$LOG" 2>&1 < /dev/null &
    printf 'Launcher PID: %s\nLog: %s\n' "$!" "$LOG"
    printf 'Results: %s/results/flute_benchmark/%s\n' "$PWD" "$RUN_NAME"
    tail -n 60 -F "$LOG"
)
```

The job waits for the shared experiment lock before installation or GPU work,
so an existing half-and-half accuracy job can finish. A second lock rejects
duplicate FLUTE launchers while one is queued or running. Ctrl+C stops `tail`,
not the background job.
The first run compiles FLUTE. Subsequent runs reuse the recorded installation.
The setup script's `BASE_PYTHON` can override the default qdlm Python executable.

## Protocol and results

1. Check W2/W3 fused execution on both authorized GPUs before model loading.
2. Calibrate using the existing 128 Wikitext-2 segments, group size 128,
   symmetric GPTQ, and no activation ordering. Verify packed coverage/bit budget.
3. Check the original and clean decoders on FP16 and the legacy quantized model.
4. Convert every selected linear and validate it against its original GPU
   forward. Compare one complete legacy/FLUTE generation and record mismatches.
5. Build the usual MMLU 5-shot/80-per-subject prompts. In latency mode use one
   fixed prompt per subject (57 total), warm each prompt on both models, then
   measure three repetitions. Execute all 128 denoising forwards and 64 slots.
   Swap GPU placement and repeat, pairing FP16 and FLUTE on each physical GPU.
6. Profile a separate forward after timing, to verify that selected linears
   actually call FLUTE. No profiling or file writes enter timed generation.

`results/flute_benchmark/<RUN_NAME>/summary.json` contains generation and request
speedups (FP16 time divided by FLUTE time; greater than 1 means faster).
Request time includes tokenization, input transfer, generation, output transfer,
stop processing and decoding. These are warm inference measurements, excluding
model loading, calibration, conversion, tuning, warmup, profiling and file I/O.
`timings.jsonl` records individual measurements and per-device peak memory.
`metadata.json` records coverage, backend, numerical checks, setup time and
templates. `backend_equivalence.json` compares legacy/FLUTE tokens for one
prompt. `backend_routes.json` and the profiler traces verify kernel execution.

Latency mode does not compute accuracy. Do not attach legacy-backend MMLU
accuracy to FLUTE as if it were measured on FLUTE. Setting `MODE=accuracy` on
a new run scores all 4560 prompts with both FP16 and FLUTE, in addition to the
paired latency protocol. Exact token agreement on one prompt is not sufficient
to establish full-set accuracy; any repeated-output differences are recorded.

CPU checks (no CUDA/FLUTE installation required):

```bash
python tools/test_qdlm_flute.py -v
python tools/test_qdlm_benchmark.py -v
```

These verify integer-code extraction (including W3 cross-word fields), LUT
and scale equivalence against the actual legacy packer and forward, unsupported
layout rejection, and decoding/timing aggregation regressions. The server
preflight and full evaluation are still required; CPU checks cannot establish
CUDA compatibility, inference accuracy, or acceleration on L40S.
