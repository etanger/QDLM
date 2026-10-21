#!/bin/bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qdlm

export CUDA_VISIBLE_DEVICES=2,3
export LLADA_DUMP_FULL_ACT=0

cd "$(dirname "$0")/.."

MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
OUT_FILE='step_logs/activation_saliency/exp101_activation_saliency_map_limit2.jsonl'

mkdir -p logs
mkdir -p step_logs/activation_saliency

rm -f "${OUT_FILE}"

echo "========================================================================"
echo "Exp101: FP16 Activation Saliency Map"
echo "Layers: 0,4,8,12,16,20,24,28"
echo "Steps: 0,16,32,48,63"
echo "Modules: attn_out, ff_proj, up_proj, ff_out"
echo "Trajectory: FP16"
echo "MMLU limit: 2"
echo "========================================================================"

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
    --quant_start_step 500

echo "========================================================================"
echo "Exp101 completed"
echo "Output: ${OUT_FILE}"
echo "========================================================================"
