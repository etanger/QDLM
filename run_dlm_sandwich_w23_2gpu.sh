#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/nvme/zwang97/QDLM
mkdir -p logs results/mixed_precision

RUN_NAME="dlm_sandwich_w23_2gpu"

exec 9>"logs/${RUN_NAME}.lock"
if ! flock -n 9; then
    echo "This experiment is already running. Skipping duplicate launch."
    exit 1
fi

echo "$$" > "logs/${RUN_NAME}.pid"
rm -f "logs/${RUN_NAME}.exitcode"

finish() {
    local rc=$?
    echo "$rc" > "logs/${RUN_NAME}.exitcode"
    echo "Finished: $(date -Is) | exit_code=$rc"
}
trap finish EXIT

export CUDA_VISIBLE_DEVICES=2,3
export QDLM_SANDWICH_CONFIG=configs/qdlm_sandwich_w23.json
export QDLM_SANDWICH_PLAN_OUT="results/mixed_precision/${RUN_NAME}_plan.json"
export PYTHONUNBUFFERED=1

echo "Started: $(date -Is)"
echo "PID: $$ | GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $QDLM_SANDWICH_CONFIG"
echo "Plan: $QDLM_SANDWICH_PLAN_OUT"

/home/zwang97/miniconda3/envs/qdlm/bin/python -u AutoGPTQ/quantize.py \
    --model /workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct \
    --wbits 2 \
    --steps 128 \
    --tasks gsm8k \
    --limit 80
