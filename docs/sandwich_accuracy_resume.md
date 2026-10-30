# Resume an interrupted W2/W3 accuracy run

An interrupted run may have saved predictions without a final `accuracy.json`.
A previous `exit_code=0` alone does not establish a completed evaluation or
identify why the process stopped.

The half-and-half launcher now requires a complete prediction set, scored
accuracy and a verified `complete.json` marker before accepting exit code zero.
It also checks physical GPUs 2 and 3 for compute processes before CUDA work.
This is an instantaneous check, not a GPU reservation: arrange availability
before starting. The launcher does not kill other processes or wait for them.

## Recovery behavior

- `RESUME_FROM` points to the old result directory. `RUN_NAME` must be new.
  The old files are read only; recovered predictions are copied into new output.
- Model, task, few-shot count, sample limit, decoding, configuration, PyTorch and
  CUDA versions must match. Core quantization/decoding source hashes are checked.
  For old metadata without hashes, the original Git revision supplies the
  expected core source, provided those files were not recorded as locally dirty.
- Cached rows are matched by the original hash of task, document, request index
  and prompt/generation arguments. Duplicate IDs, changed prompts, invalid token
  counts and changed total request counts cause an error. A torn, invalid final
  JSON line without a newline is ignored and reported; the source stays intact.
- The original evaluator did not save a quantized checkpoint. Model loading and
  calibration must run again. The rebuilt model then repeats up to three cached
  generations (first, middle, last) and requires identical tokens and answers.
  Only missing cases are generated afterward; all predictions are scored together.
- New runs record a model state hash; subsequent recovery requires exact state
  equality too. The old run has no such hash, so the first recovery cannot prove
  equality of every original weight. Its metadata explicitly records this limit.
- Predictions are flushed after every new case. Metadata and final JSON files
  use atomic replacement. If a recovered run is interrupted after generation
  resumes, use that newer result directory as the next `RESUME_FROM`.

## Run when GPUs 2 and 3 are available

```bash
(
    set -e
    cd /path/to/QDLM
    RUN_NAME="dlm_w23_half_resume_$(date +%Y%m%d_%H%M%S)"
    RESUME_FROM="$PWD/results/w23_half_accuracy/INTERRUPTED_RUN_NAME"
    mkdir -p logs
    LOG="$PWD/logs/$RUN_NAME.log"
    touch "$LOG"
    nohup env RUN_NAME="$RUN_NAME" RESUME_FROM="$RESUME_FROM" \
        bash run_dlm_sandwich_w23_half_2gpu.sh \
        > "$LOG" 2>&1 < /dev/null &
    printf 'Launcher PID: %s\nLog: %s\n' "$!" "$LOG"
    printf 'Results: %s/results/w23_half_accuracy/%s\n' "$PWD" "$RUN_NAME"
    tail -n 60 -F "$LOG"
)
```

Replace the repository path and `INTERRUPTED_RUN_NAME` before running.
After rebuilding and cache checks, the log reports the total, reused and
remaining request counts.
The generation progress bar starts at the number reused. Ctrl+C stops the log
viewer while the nohup job continues. Full model/GPU recovery needs server
validation; local tests exercise cache handling and launcher checks on CPU.
