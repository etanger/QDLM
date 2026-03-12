
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='/path/to/your/model' 
W_BIT=8
A_BIT=8

# model_path: the path to the pretrained model
# alpha: the smooth quantization hyperparameter
# we use asymmetric per-channel quantization for weights and per-tensor quantization for activations in SmoothQuant

python $DIRPATH/DuQuant/generate_act_scale_shift.py --model $MODEL_PATH

# general tasks
# --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge
python $DIRPATH/DuQuant/main.py \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --task piqa \
    --smooth \
    --alpha 0.5 \


# MMLU
python $DIRPATH/DuQuant/main.py \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --smooth \
    --alpha 0.5 \
    --task mmlu --num_fewshot 5 --mc_num 1 \


# GSM8K
python $DIRPATH/DuQuant/main.py \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --smooth \
    --alpha 0.5 \
    --task gsm8k --gen_length 256 --steps 256 --block_length 32 --num_fewshot 4 \


# MATH
python $DIRPATH/DuQuant/main.py \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --smooth \
    --alpha 0.5 \
    --tasks minerva_math  --num_fewshot 0 --gen_length 256 --steps 256 --block_length 64 \


# HumanEval
python $DIRPATH/DuQuant/main.py \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --smooth \
    --alpha 0.5 \
    --task humaneval --gen_length 512 --steps 512 --block_length 32 --num_fewshot 0 \


# MBPP
python $DIRPATH/DuQuant/main.py \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --smooth \
    --alpha 0.5 \
    --task mbpp --gen_length 512 --steps 512 --block_length 32 --num_fewshot 3 \