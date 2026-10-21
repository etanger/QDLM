#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODE="${MODE:-latency}"
RUN_NAME="${RUN_NAME:-dlm_sandwich_paired_${MODE}_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zwang97/miniconda3/envs/qdlm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct}"
if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, underscores, and hyphens."
    exit 2
fi
mkdir -p logs results/paired_benchmark
exec 9>logs/dlm_sandwich_paired_2gpu.lock
if ! flock -n 9; then
    echo "A paired benchmark is already running."
    exit 1
fi
if [[ -e "results/paired_benchmark/$RUN_NAME" ]]; then
    echo "Result directory already exists. Choose a new RUN_NAME."
    exit 2
fi
echo "$$" > "logs/$RUN_NAME.pid"
finish() {
    local rc=$?
    echo "$rc" > "logs/$RUN_NAME.exitcode"
    echo "Finished: $(date -Is) | exit_code=$rc"
}
trap finish EXIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED=1
echo "Started: $(date -Is) | PID: $$ | GPUs: $CUDA_VISIBLE_DEVICES | mode: $MODE"
echo "Standard: MMLU generative, 5-shot, 80 per subject, 128 steps, 64 generated slots"
echo "Latency: fixed subset, three repeats, swapped GPU placement; no accuracy in latency mode"
"$PYTHON_BIN" -u tools/benchmark_sandwich.py \
    --model "$MODEL_PATH" \
    --output "results/paired_benchmark/$RUN_NAME" \
    --mode "$MODE" \
    --tasks mmlu_generative --num-fewshot 5 --limit 80 \
    --steps 128 --gen-length 64 --block-length 64 \
    --latency-per-task 1 --repeats 3 --warmup 1
