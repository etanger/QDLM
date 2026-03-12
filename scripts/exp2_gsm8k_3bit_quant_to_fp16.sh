#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

cd "$(dirname "$0")/.."

MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=3
QUANT_START_STEP=64
DIRPATH=$(pwd)

echo "=========================================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "REVERSE: Quant→FP16 (步骤 0-63 量化，64-256 FP16)"
echo "Model: $MODEL_PATH"
echo "W_BIT: $W_BIT"
echo "QUANT_START_STEP: $QUANT_START_STEP"
echo "=========================================="

python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits $W_BIT \
    --tasks gsm8k \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --quant_start_step $QUANT_START_STEP

echo "=========================================="
echo "Reverse experiment completed!"
echo "Check results/ and step_logs/ for outputs"
echo "=========================================="

