#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
HDD=/workspace/hdd/users/zwang97/QDLM
RUN_NAME="${RUN_NAME:-humming_full_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$HDD/logs" "$HDD/results/humming_full"
trap 'code=$?; printf "Finished: %s | exit_code=%s\n" "$(date -Is)" "$code"; exit "$code"' EXIT
/home/zwang97/miniconda3/envs/qdlm/bin/python -u tools/bench_humming_full.py \
    --source "$HDD/results/humming_samples/humming_samples_20261106_090130" \
    --output "$HDD/results/humming_full/$RUN_NAME" \
    --repeats "${REPEATS:-1}"
