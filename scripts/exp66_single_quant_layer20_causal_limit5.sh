#!/bin/bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qdlm

export CUDA_VISIBLE_DEVICES=2,3

cd "$(dirname "$0")/.."

MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=2
QUANT_START_STEP=0
DIRPATH=$(pwd)

PAIR_FILE="step_logs/layer_causal/exp66_single_quant_layer20.jsonl"

echo "========================================================================"
echo "Exp66: Single-Layer Causal Quantization Analysis"
echo "Quantized layer: 20 only"
echo "Other layers: FP16"
echo "Trajectory driver: FP16"
echo "Analysis steps: 0, 8, 16, 32, 48, 63"
echo "MMLU limit: 5"
echo "========================================================================"

mkdir -p logs
mkdir -p step_logs/layer_causal

rm -f "${PAIR_FILE}"
rm -f step_logs/denoise_step_metrics.jsonl
touch step_logs/denoise_step_metrics.jsonl

export LLADA_DUMP_FULL_ACT=0

python "${DIRPATH}/AutoGPTQ/quantize.py" \
    --model "${MODEL_PATH}" \
    --wbits "${W_BIT}" \
    --tasks mmlu_generative \
    --num_fewshot 5 \
    --steps 128 \
    --gen_length 64 \
    --block_length 64 \
    --limit 5 \
    --quant_start_step "${QUANT_START_STEP}"

echo "========================================================================"
echo "Exp66 completed!"
echo "Pair-analysis file:"
echo "${PAIR_FILE}"
echo "========================================================================"
