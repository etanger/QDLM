#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
HDD=/workspace/hdd/users/zwang97/QDLM
RUN_NAME="${RUN_NAME:-humming_baselines_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$HDD/logs" "$HDD/results/humming_baselines"
trap 'code=$?; printf "Finished: %s | exit_code=%s\n" "$(date -Is)" "$code"; exit "$code"' EXIT
/home/zwang97/miniconda3/envs/qdlm/bin/python -u tools/bench_humming_baselines.py \
    --output "$HDD/results/humming_baselines/$RUN_NAME" \
    --wait-hours "${WAIT_HOURS:-3}"
