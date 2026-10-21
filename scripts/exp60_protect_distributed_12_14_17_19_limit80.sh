#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3

cd "$(dirname "$0")/.."

MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=2
QUANT_START_STEP=0
DIRPATH=$(pwd)

TAG="exp60_protect_distributed_12_14_17_19_limit80"
METRIC_FILE="step_logs/distributed_protection/${TAG}_denoise_step_metrics.jsonl"

echo "========================================================================"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Exp60: Distributed FP16 Protection"
echo "Protected FP16 layers: 12, 14, 17, 19"
echo "Quantized 2-bit layers: 0-11, 13, 15-16, 18, 20-31"
echo "FP16 layer count: 4"
echo "Quantized layer count: 28"
echo "Model: ${MODEL_PATH}"
echo "W_BIT: ${W_BIT}"
echo "QUANT_START_STEP: ${QUANT_START_STEP}"
echo "Task: MMLU generative"
echo "Few-shot: 5"
echo "LIMIT: 80"
echo "Steps: 128"
echo "Generation length: 64"
echo "Block length: 64"
echo "Activation dump: OFF"
echo "========================================================================"

echo "Current skip_layers setting:"
grep -n "skip_layers" \
    AutoGPTQ/auto_gptq/modeling/_base.py \
    | tail -n 15

export LLADA_DUMP_FULL_ACT=0
export LLADA_DUMP_ROOT=activation_dump
export LLADA_DUMP_TAG="${TAG}"

mkdir -p logs
mkdir -p step_logs/distributed_protection

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

echo "========================================================================"
echo "Exp60 completed successfully!"
echo "Saved denoise metrics:"
echo "${METRIC_FILE}"
echo "========================================================================"
