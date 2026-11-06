#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RUN_NAME="${RUN_NAME:-humming_samples_$(date +%Y%m%d_%H%M%S)}"
HDD=/workspace/hdd/users/zwang97/QDLM
mkdir -p "$HDD/logs" "$HDD/results/humming_samples"
trap 'code=$?; printf "Finished: %s | exit_code=%s\n" "$(date -Is)" "$code"; exit "$code"' EXIT
/home/zwang97/miniconda3/envs/qdlm/bin/python -u tools/bench_humming_samples.py \
    --run "$HDD/results/flute_benchmark/dlm_w23_flute_wait_20261102_052952" \
    --output "$HDD/results/humming_samples/$RUN_NAME" \
    --gpu "${GPU:-auto}"
