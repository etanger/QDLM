#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/zwang97/miniconda3/envs/qdlm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct}"
CONFIG="configs/qdlm_sandwich_w23_half.json"
RUN_NAME="${RUN_NAME:-dlm_w23_half_$(date +%Y%m%d_%H%M%S)}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
    RESUME_ARGS=(--resume-from "$RESUME_FROM")
fi
if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, underscores, and hyphens."
    exit 2
fi
mkdir -p logs results/w23_half_accuracy
OUTPUT_DIR="results/w23_half_accuracy/$RUN_NAME"
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Result directory exists; choose a new RUN_NAME."
    exit 2
fi
# Use the same lock as the avg2 and paired benchmark runners.
exec 9>logs/dlm_sandwich_paired_2gpu.lock
if ! flock -n 9; then
    echo "A sandwich experiment is already running. Launch cancelled."
    exit 1
fi
echo "$$" > "logs/$RUN_NAME.pid"
finish() {
    local rc=$?
    if [[ "$rc" -eq 0 ]]; then
        if ! "$PYTHON_BIN" tools/qdlm_accuracy_resume.py --check-complete "$OUTPUT_DIR"; then
            echo "Evaluation is incomplete; a zero Python exit status is not sufficient."
            rc=1
        fi
    fi
    echo "$rc" > "logs/$RUN_NAME.exitcode"
    echo "Finished: $(date -Is) | exit_code=$rc"
    exit "$rc"
}
trap finish EXIT
"$PYTHON_BIN" tools/qdlm_accuracy_resume.py --check-gpus
echo "Started: $(date -Is) | PID: $$ | Physical GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG | Calibration: GPU 2 | Inference: GPU 3"
echo "Layers 0-7: W2 | Layers 8-23: W3 | Layers 24-31: W2"
echo "Selected weight average: 2.5 bits | Activations: FP16"
echo "MMLU generative: 80 per subject, 5-shot, 128 steps, 64 generated slots"
if [[ -n "$RESUME_FROM" ]]; then
    echo "Resume source: $RESUME_FROM | New results: $OUTPUT_DIR"
fi
"$PYTHON_BIN" -u tools/evaluate_sandwich.py \
    --model "$MODEL_PATH" --config "$CONFIG" --variant quantized \
    --output "$OUTPUT_DIR" \
    --tasks mmlu_generative --num-fewshot 5 --limit 80 \
    --steps 128 --gen-length 64 --block-length 64 "${RESUME_ARGS[@]}"
