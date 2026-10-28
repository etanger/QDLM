# W2/W3 half-and-half accuracy experiment

This experiment measures the accuracy recovered by raising the outer layers
from W1 to W2 relative to the W1/W3 run that scored 53.33% on MMLU.

| Transformer layers (zero-based) | Weight bits |
| --- | --- |
| 0-7 | 2 |
| 8-23 | 3 |
| 24-31 | 2 |

Edit `configs/qdlm_sandwich_w23_half.json` to inspect the allocation.
The executed stage is `w23`; both budget fields are 2.5. The 16-layer middle
interval is centered and each edge contains eight layers. No W1 is executed.
For equal selected parameter counts per block, the average is 2.5 bits.
This average covers the selected quantized weights, not every model parameter.

The existing evaluator, quantized module selection, GPTQ implementation,
calibration data, and decoding/scoring settings are reused. Calibration uses
128 Wikitext-2 segments of length 2048, group size 128, symmetric GPTQ, and no
activation ordering. Evaluation uses FP16 activations, batch size 1, MMLU
generative with five shots and 80 examples per subject (4560 total), 128
denoising forwards, and generation/block length 64. Every forward uses the
quantized model. This is an accuracy experiment; it makes no speed claim.

## Run in the background

The launcher uses the existing qdlm Python executable even from a base shell.
Physical GPU 2 is used for calibration and physical GPU 3 for inference,
sequentially. CUDA visibility is fixed to `2,3`. A shared lock prevents overlap
with the avg2 and paired benchmark launchers.

```bash
(
    set -e
    cd /workspace/nvme/zwang97/QDLM
    RUN_NAME="dlm_w23_half_$(date +%Y%m%d_%H%M%S)"
    mkdir -p logs
    LOG="$PWD/logs/$RUN_NAME.log"
    touch "$LOG"
    nohup env RUN_NAME="$RUN_NAME" \
        bash run_dlm_sandwich_w23_half_2gpu.sh \
        > "$LOG" 2>&1 < /dev/null &
    printf 'Launcher PID: %s\nLog: %s\n' "$!" "$LOG"
    printf 'Results: %s/results/w23_half_accuracy/%s\n' "$PWD" "$RUN_NAME"
    tail -n 60 -F "$LOG"
)
```

Ctrl+C stops the log viewer; the nohup experiment continues. Inspect the log
for an early lock refusal or other startup failure before assuming it started.
Use a new run name for every experiment; existing result directories are
rejected by the evaluator.

## Results

`results/w23_half_accuracy/<RUN_NAME>/` contains `metadata.json`,
`sandwich_plan.json`, `outputs.jsonl`, and `accuracy.json`.
The executed allocation is in `sandwich_plan.json` under `current_w23_stage`.
With the current module selection, packed validation should report 64 W2 and
64 W3 modules, average weight bits 2.5, and status `pass`.
The launcher also writes `logs/<RUN_NAME>.pid` and `.exitcode`.
Compare `accuracy.json` with the previous W1/W3 result under the same evaluator.
