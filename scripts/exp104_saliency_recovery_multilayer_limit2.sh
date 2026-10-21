#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3
export LLADA_DUMP_FULL_ACT=0

MODEL_PATH="/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct"

for LAYER in 8 12 16 20 24
do
    echo "============================================================"
    echo "Exp104 target layer: ${LAYER}"
    echo "============================================================"

    export EXP104_TARGET_LAYER="${LAYER}"

    OUTPUT_FILE="step_logs/saliency_recovery/exp104_saliency_recovery_layer${LAYER}_limit2.jsonl"

    # Avoid appending to an old run.
    rm -f "${OUTPUT_FILE}"

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

    echo "Finished layer ${LAYER}"
    echo
done
