# Quantize the model using AutoGPTQ
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='/path/to/your/model'  # Replace with your model path
W_BIT=3
QUANT_START_STEP=64                        # When to switch from FP16 to quantized

# model_path: the path to the pretrained model or instruct-tuned model
# wbits: the weight bit-width for GPTQ quantization
# by default, we use 128 as the group size for GPTQ quantization

# General tasks (gen_length=1024, steps=1024, block_length=1024 by default - 1 block)
python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits $W_BIT \
    --tasks piqa \
    --quant_start_step $QUANT_START_STEP

# MMLU
python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits $W_BIT \
    --tasks mmlu \
    --num_fewshot 5 \
    --mc_num 1 \
    --quant_start_step $QUANT_START_STEP

# GSM8K (256 length, 1 block)
python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits $W_BIT \
    --tasks gsm8k \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --num_fewshot 4 \
    --quant_start_step $QUANT_START_STEP

# MATH (256 length, 1 block)
python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits $W_BIT \
    --tasks minerva_math \
    --num_fewshot 0 \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --quant_start_step $QUANT_START_STEP

# HumanEval (512 length, 1 block)
python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits $W_BIT \
    --tasks humaneval \
    --gen_length 512 \
    --steps 512 \
    --block_length 512 \
    --num_fewshot 0 \
    --quant_start_step $QUANT_START_STEP

# MBPP (512 length, 1 block)
python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits $W_BIT \
    --tasks mbpp \
    --gen_length 512 \
    --steps 512 \
    --block_length 512 \
    --num_fewshot 3 \
    --quant_start_step $QUANT_START_STEP