#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
cd "$(dirname "$0")/.."
MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=2
QUANT_START_STEP=32
DIRPATH=$(pwd)
echo "=========================================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Strategy: FP16 → 2-bit Quant (step 0-31: FP16, 32-128: 2bit)"
echo "Task: MMLU (Full dataset)"
echo "Model: $MODEL_PATH"
echo "W_BIT: $W_BIT"
echo "QUANT_START_STEP: $QUANT_START_STEP"
echo "=========================================="
python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits $W_BIT \
    --tasks mmlu_generative \
    --num_fewshot 5 \
    --steps 128 \
    --gen_length 64 \
    --block_length 64 \
    --limit 80 \
    --quant_start_step $QUANT_START_STEP
echo "=========================================="
echo "FP16→2-bit Quant MMLU completed!"
echo "=========================================="
