
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='/path/to/your/model'  # Replace with your model path

# model_path: the path to the pretrained model or instruct-tuned model
# a_bits: the activation bit-width for QuaRot quantization
# v_bits: the value bit-width for QuaRot quantization
# k_bits: the key bit-width for QuaRot quantization
# w_bits: the weight bit-width for QuaRot quantization
# w_clip: whether to apply weight clipping for QuaRot quantization
# we use symmetric per-channel quantization for weights and per-tensor quantization for activations (query states maintain fp16) in QuaRot
# you could revise quantization settings according to your needs, please refer to ./QuaRot/fake_quant/README.md

# you could also save the GPTQ quantized model (--save_qmodel_path gptq_cache/$MODEL-w3-g128) and load the quantized model for evaluation (--load_qmodel_path gptq_cache/$MODEL-w3-g128)

# general tasks
# --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge
python $DIRPATH/QuaRot/fake_quant/main.py --model $MODEL_PATH \
    --a_bits 8 --v_bits 8 --k_bits 8 --w_bits 8 --w_clip --tasks piqa --rotate


# MMLU
python $DIRPATH/QuaRot/fake_quant/main.py --model $MODEL_PATH \
    --a_bits 8 --v_bits 8 --k_bits 8 --w_bits 8 --w_clip --tasks mmlu --num_fewshot 5 --mc_num 1 --rotate


# GSM8K
python $DIRPATH/QuaRot/fake_quant/main.py --model $MODEL_PATH \
    --a_bits 8 --v_bits 8 --k_bits 8 --w_bits 8 --w_clip --rotate \
    --task gsm8k --gen_length 256 --steps 256 --block_length 32 --num_fewshot 4 \

# MATH
python $DIRPATH/QuaRot/fake_quant/main.py --model $MODEL_PATH \
    --a_bits 8 --v_bits 8 --k_bits 8 --w_bits 8 --w_clip --rotate \
    --tasks minerva_math --gen_length 256 --steps 256 --block_length 64 --num_fewshot 0 \


# HumanEval
python $DIRPATH/QuaRot/fake_quant/main.py --model $MODEL_PATH \
    --a_bits 8 --v_bits 8 --k_bits 8 --w_bits 8 --w_clip --rotate \
    --task humaneval --gen_length 512 --steps 512 --block_length 32 --num_fewshot 0 \


# MBPP
python $DIRPATH/QuaRot/fake_quant/main.py --model $MODEL_PATH \
    --a_bits 8 --v_bits 8 --k_bits 8 --w_bits 8 --w_clip --rotate \
    --task mbpp --gen_length 512 --steps 512 --block_length 32 --num_fewshot 3 \