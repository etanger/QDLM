#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/zwang97/miniconda3/envs/qdlm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct}"
CONFIG="configs/qdlm_sandwich_w23_half.json"
RUN_NAME="${RUN_NAME:-dlm_w23_half_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, underscores, and hyphens."
    exit 2
fi
mkdir -p logs results/w23_half_accuracy
# Use the same lock as the avg2 and paired benchmark runners.
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
echo "Layers 0-7: W2 | Layers 8-23: W3 | Layers 24-31: W2"
echo "Selected weight average: 2.5 bits | Activations: FP16"
echo "MMLU generative: 80 per subject, 5-shot, 128 steps, 64 generated slots"
"$PYTHON_BIN" -u tools/evaluate_sandwich.py \
    --model "$MODEL_PATH" --config "$CONFIG" --variant quantized \
    --output "results/w23_half_accuracy/$RUN_NAME" \
    --tasks mmlu_generative --num-fewshot 5 --limit 80 \
    --steps 128 --gen-length 64 --block-length 64
