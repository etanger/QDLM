#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/zwang97/miniconda3/envs/qdlm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct}"
CONFIG="${CONFIG:-configs/qdlm_sandwich_w13_avg2.json}"
RUN_NAME="${RUN_NAME:-dlm_sandwich_avg2_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, underscores, and hyphens."
    exit 2
fi
mkdir -p logs results/avg2_accuracy
# Share the paired runner lock so these experiments cannot overlap by accident.
exec 9>logs/dlm_sandwich_paired_2gpu.lock
if ! flock -n 9; then
    echo "A sandwich experiment is already running. Launch cancelled."
    exit 1
fi
echo "$$" > "logs/$RUN_NAME.pid"
finish() {
    local rc=$?
    echo "$rc" > "logs/$RUN_NAME.exitcode"
    echo "Finished: $(date -Is) | exit_code=$rc"
}
trap finish EXIT
echo "Started: $(date -Is) | PID: $$ | Physical GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG | Calibration: GPU 2 | Inference: GPU 3"
echo "MMLU generative: 80 per subject, 5-shot, 128 steps, 64 generated slots"
"$PYTHON_BIN" -u tools/evaluate_sandwich.py \
    --model "$MODEL_PATH" --config "$CONFIG" \
    --output "results/avg2_accuracy/$RUN_NAME" \
    --tasks mmlu_generative --num-fewshot 5 --limit 80 \
    --steps 128 --gen-length 64 --block-length 64 "$@"
