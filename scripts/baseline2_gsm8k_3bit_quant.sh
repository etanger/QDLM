#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
cd "$(dirname "$0")/.."
MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=3
QUANT_START_STEP=0
DIRPATH=$(pwd)
echo "=========================================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Baseline: GSM8K Full 3-bit Quant (all 256 steps)"
echo "Model: $MODEL_PATH"
echo "QUANT_START_STEP: $QUANT_START_STEP (=0, always quant)"
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
echo "Baseline GSM8K 3-bit Quant completed!"
echo "=========================================="
