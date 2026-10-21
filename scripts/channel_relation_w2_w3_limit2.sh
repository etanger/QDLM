#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3
export LLADA_DUMP_FULL_ACT=0
export GPTQ_DUMP_CALIB_H=1

MODEL_PATH="/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct"

mkdir -p step_logs/channel_relation
mkdir -p step_logs/gptq_calibration
mkdir -p step_logs/gptq_probe

for BITS in 2 3
do
    for LAYER in 8 12 16 20 24
    do
        echo "============================================================"
        echo "Channel relation | W${BITS} | Layer ${LAYER}"
        echo "============================================================"

        export EXP104_TARGET_LAYER="${LAYER}"
        export GPTQ_PROBE_BITS="${BITS}"

        ENERGY_FILE="step_logs/channel_relation/inference_energy_layer${LAYER}_w${BITS}_limit2.jsonl"
        ERROR_FILE="step_logs/channel_relation/column_error_layer${LAYER}_w${BITS}.pt"
        PROBE_FILE="step_logs/gptq_probe/gptq_probe_layer${LAYER}_w${BITS}_limit2.jsonl"
        H_FILE="step_logs/gptq_calibration/attn_out_layer${LAYER}_w${BITS}.pt"

        rm -f "${ENERGY_FILE}"
        rm -f "${ERROR_FILE}"
        rm -f "${PROBE_FILE}"
        rm -f "${H_FILE}"

        python AutoGPTQ/quantize.py \
            --model "${MODEL_PATH}" \
            --wbits "${BITS}" \
            --tasks mmlu_generative \
            --num_fewshot 5 \
            --steps 128 \
            --mc_num 128 \
            --batch_size 32 \
            --gen_length 64 \
            --block_length 64 \
            --limit 2 \
            --quant_start_step 0

        echo
        echo "Finished W${BITS} Layer ${LAYER}"
        echo
    done
done
