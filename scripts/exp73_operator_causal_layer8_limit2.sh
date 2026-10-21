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

OUT_FILE="step_logs/operator_causal/exp73_operator_causal_layer8_limit2.jsonl"

echo "========================================================================"
echo "Exp73: Within-Block Operator Causal Probe"
echo "Target layer: 8"
echo "Quantized layer: 8 only"
echo "Variants:"
echo "  attn_out only W2"
echo "  ff_proj only W2"
echo "  up_proj only W2"
echo "  ff_out only W2"
echo "  all four W2"
echo "Trajectory driver: FP16"
echo "Analysis steps: 0,16,32,48,63"
echo "MMLU limit: 2"
echo "========================================================================"

mkdir -p logs
mkdir -p step_logs/operator_causal

rm -f "${OUT_FILE}"

export LLADA_DUMP_FULL_ACT=0

python "${DIRPATH}/AutoGPTQ/quantize.py" \
    --model "${MODEL_PATH}" \
    --wbits "${W_BIT}" \
    --tasks mmlu_generative \
    --num_fewshot 5 \
    --steps 128 \
    --gen_length 64 \
    --block_length 64 \
    --limit 2 \
    --quant_start_step "${QUANT_START_STEP}"

echo "========================================================================"
echo "Exp73 completed!"
echo "Output:"
echo "${OUT_FILE}"
echo "========================================================================"
