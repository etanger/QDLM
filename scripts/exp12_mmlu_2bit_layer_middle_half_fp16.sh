#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
cd "$(dirname "$0")/.."
MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=2
QUANT_START_STEP=0
DIRPATH=$(pwd)
echo "=========================================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Exp12: MMLU Layer Sensitivity - Middle Layers FP16"
echo "  Layer 0-7:  2-bit Quantization"
echo "  Layer 8-23: FP16 (skip quantization)"
echo "  Layer 24-31: 2-bit Quantization"
echo "Model: $MODEL_PATH"
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
echo "Exp12 completed!"
echo "=========================================="
