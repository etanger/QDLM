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

C_FILE="step_logs/weight_activation_probe/exp68_cprobe_layer8_limit2.jsonl"

echo "========================================================================"
echo "Exp68: Weight-Activation Decomposition Probe"
echo "Target layer: 8"
echo "Quantized layer: 8 only"
echo "Other layers: FP16"
echo "Trajectory driver: FP16"
echo "Analysis steps: 0,16,32,48,63"
echo "Modules: q_proj, attn_out, ff_proj, up_proj, ff_out"
echo "Isotropic controls: 2 seeds"
echo "MMLU limit: 2"
echo "========================================================================"

mkdir -p logs
mkdir -p step_logs/weight_activation_probe

rm -f "${C_FILE}"
rm -f step_logs/denoise_step_metrics.jsonl
touch step_logs/denoise_step_metrics.jsonl

export LLADA_DUMP_FULL_ACT=0

python "${DIRPATH}/AutoGPTQ/quantize.py" \
    --model "${MODEL_PATH}" \
    --wbits "${W_BIT}" \
    --tasks mmlu_generative \
    --num_fewshot 5 \
    --steps 128 \
    --gen_length 64 \
    --block_length 64 \
    --limit 2 \
    --quant_start_step "${QUANT_START_STEP}"

echo "========================================================================"
echo "Exp68 completed!"
echo "C-probe output:"
echo "${C_FILE}"
echo "========================================================================"
