#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
export LLADA_DUMP_FULL_ACT=0

MODEL_PATH="/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct"

python AutoGPTQ/quantize.py \
    --model "${MODEL_PATH}" \
    --wbits 2 \
    --tasks mmlu_generative \
    --num_fewshot 5 \
    --steps 128 \
    --mc_num 128 \
    --batch_size 32 \
    --gen_length 64 \
    --block_length 64 \
    --limit 2 \
    --quant_start_step 0
