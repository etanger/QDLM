#!/bin/bash

# 锁死只用 GPU 2 和 3
export CUDA_VISIBLE_DEVICES=2,3

# 回到项目根目录
cd "$(dirname "$0")/.."

MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=3
QUANT_START_STEP=64
DIRPATH=$(pwd)

echo "=========================================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Testing GENERATION task (GSM8K)"
echo "Model: $MODEL_PATH"
echo "W_BIT: $W_BIT"
echo "QUANT_START_STEP: $QUANT_START_STEP"
echo "=========================================="

python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits $W_BIT \
    --tasks gsm8k \
    --limit 3 \
    --gen_length 128 \
    --steps 128 \
    --block_length 128 \
    --quant_start_step $QUANT_START_STEP

echo "=========================================="
echo "Test completed!"
echo "Check step_logs/ for step-wise metrics!"
echo "=========================================="
