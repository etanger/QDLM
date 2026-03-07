#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

cd "$(dirname "$0")/.."

MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=2
QUANT_START_STEP=64
DIRPATH=$(pwd)

echo "=========================================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Strategy: 2-bit Quant → FP16 (step 0-63: 2bit, 64-256: FP16)"
echo "Task: MMLU (Full dataset)"
echo "Model: $MODEL_PATH"
echo "W_BIT: $W_BIT"
echo "QUANT_START_STEP: $QUANT_START_STEP"
echo "=========================================="

python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits $W_BIT \
    --tasks mmlu \
    --num_fewshot 5 \
    --quant_start_step $QUANT_START_STEP

echo "=========================================="
echo "2-bit Quant→FP16 MMLU completed!"
echo "=========================================="
