#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
cd "$(dirname "$0")/.."

MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=2
QUANT_START_STEP=0
DIRPATH=$(pwd)

echo "=========================================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Exp20: Sliding Window Quantization"
echo "Quantized layers: 8-19"
echo "FP16 skipped layers: 0-7 and 20-31"
echo "Model: $MODEL_PATH"
echo "W_BIT: $W_BIT"
echo "QUANT_START_STEP: $QUANT_START_STEP"
echo "LIMIT: 80"
echo "Activation dump: OFF"
echo "=========================================="

echo "Current skip_layers setting:"
grep -n "skip_layers" AutoGPTQ/auto_gptq/modeling/_base.py | tail -n 8

export LLADA_DUMP_FULL_ACT=0
export LLADA_DUMP_ROOT=activation_dump
export LLADA_DUMP_TAG=exp20_sliding_w12_s8_e19_limit80

rm -f step_logs/denoise_step_metrics.jsonl
touch step_logs/denoise_step_metrics.jsonl
mkdir -p step_logs/sliding_window

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

cp step_logs/denoise_step_metrics.jsonl \
   step_logs/sliding_window/exp20_sliding_w12_s8_e19_limit80_denoise_step_metrics.jsonl

echo "=========================================="
echo "Exp20 completed!"
echo "Saved denoise metrics:"
echo "step_logs/sliding_window/exp20_sliding_w12_s8_e19_limit80_denoise_step_metrics.jsonl"
echo "=========================================="
