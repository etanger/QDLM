#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3

cd "$(dirname "$0")/.."

MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=2
QUANT_START_STEP=0
DIRPATH=$(pwd)

TAG="exp53_protect_w4_s20_e23_limit80"
METRIC_FILE="step_logs/protect_window/${TAG}_denoise_step_metrics.jsonl"

echo "=========================================="
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Exp53: Sliding FP16 Protection Window"
echo "Protected FP16 layers: 20-23"
echo "Quantized layers: 0-19 and 24-31"
echo "Model: ${MODEL_PATH}"
echo "W_BIT: ${W_BIT}"
echo "QUANT_START_STEP: ${QUANT_START_STEP}"
echo "LIMIT: 80"
echo "Activation dump: OFF"
echo "=========================================="

echo "Current skip_layers setting:"
grep -n "skip_layers" AutoGPTQ/auto_gptq/modeling/_base.py | tail -n 8

export LLADA_DUMP_FULL_ACT=0

mkdir -p logs
mkdir -p step_logs/protect_window

rm -f step_logs/denoise_step_metrics.jsonl
rm -f "${METRIC_FILE}"
touch step_logs/denoise_step_metrics.jsonl

python "${DIRPATH}/AutoGPTQ/quantize.py" \
    --model "${MODEL_PATH}" \
    --wbits "${W_BIT}" \
    --tasks mmlu_generative \
    --num_fewshot 5 \
    --steps 128 \
    --gen_length 64 \
    --block_length 64 \
    --limit 80 \
    --quant_start_step "${QUANT_START_STEP}"

cp step_logs/denoise_step_metrics.jsonl "${METRIC_FILE}"

echo "=========================================="
echo "Exp53 completed!"
echo "Saved denoise metrics:"
echo "${METRIC_FILE}"
echo "=========================================="
