#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONUNBUFFERED=1
RUN_NAME="${RUN_NAME:-dlm_w23_flute_$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-latency}"
MODEL_PATH="${MODEL_PATH:-/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct}"
if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ || ! "$MODE" =~ ^(latency|accuracy)$ ]]; then
    echo "Invalid RUN_NAME or MODE."
    exit 2
fi
mkdir -p logs results/flute_benchmark
echo "$$" > "logs/$RUN_NAME.pid"
finish() {
    local rc=$?
    echo "$rc" > "logs/$RUN_NAME.exitcode"
    echo "Finished: $(date -Is) | exit_code=$rc"
}
trap finish EXIT
exec 8>logs/dlm_sandwich_flute_2gpu.lock
if ! flock -n 8; then
    echo "A FLUTE experiment is already queued or running. Launch cancelled."
    exit 1
fi
exec 9>logs/dlm_sandwich_paired_2gpu.lock
echo "Waiting for the shared experiment lock; an existing accuracy run can finish."
flock 9
if [[ -e "results/flute_benchmark/$RUN_NAME" ]]; then
    echo "Result directory exists; choose a new RUN_NAME."
    exit 2
fi
echo "Started: $(date -Is) | Physical GPUs: 2,3 | mode: $MODE"
echo "W2 edges: 0-7,24-31 | W3 middle: 8-23 | Selected average: 2.5 bits"
bash scripts/setup_qdlm_flute.sh
"$PWD/.venv/qdlm-flute/bin/python" -u tools/benchmark_sandwich.py \
    --model "$MODEL_PATH" --output "results/flute_benchmark/$RUN_NAME" \
    --config configs/qdlm_sandwich_w23_half.json --quant-backend flute \
    --mode "$MODE" --tasks mmlu_generative --num-fewshot 5 --limit 80 \
    --steps 128 --gen-length 64 --block-length 64 \
    --latency-per-task 1 --repeats 3 --warmup 1
