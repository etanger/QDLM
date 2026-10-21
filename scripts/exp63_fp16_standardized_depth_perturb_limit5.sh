#!/bin/bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qdlm

export CUDA_VISIBLE_DEVICES=2,3

cd "$(dirname "$0")/.."

MODEL_PATH='/workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct'
W_BIT=2
QUANT_START_STEP=0
DIRPATH=$(pwd)

TAG="exp63_fp16_standardized_depth_perturb_limit5"
PERTURB_FILE="step_logs/depth_perturbation/exp63_fp16_standardized_alpha001_limit5.jsonl"

echo "========================================================================"
echo "Exp63: FP16 Standardized Depth Perturbation"
echo "Trajectory driver: FP16"
echo "Perturbed layers: 0,4,8,12,16,20,24,28"
echo "Perturbed steps: 0,16,32,48,63"
echo "Relative perturbation: 1%"
echo "Noise directions: 2"
echo "Quantization used as intervention: NO"
echo "Task: MMLU generative"
echo "Limit: 5"
echo "========================================================================"

mkdir -p logs
mkdir -p step_logs/depth_perturbation

rm -f "${PERTURB_FILE}"
rm -f step_logs/denoise_step_metrics.jsonl
touch step_logs/denoise_step_metrics.jsonl

export LLADA_DUMP_FULL_ACT=0
export LLADA_DUMP_TAG="${TAG}"

python "${DIRPATH}/AutoGPTQ/quantize.py" \
    --model "${MODEL_PATH}" \
    --wbits "${W_BIT}" \
    --tasks mmlu_generative \
    --num_fewshot 5 \
    --steps 128 \
    --gen_length 64 \
    --block_length 64 \
    --limit 5 \
    --quant_start_step "${QUANT_START_STEP}"

echo "========================================================================"
echo "Exp63 completed!"
echo "Perturbation metrics:"
echo "${PERTURB_FILE}"
echo "========================================================================"
