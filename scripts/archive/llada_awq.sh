# Quantize the model using AWQ
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='/path/to/your/model'  # Replace with your model path
W_BIT=3

# model_path: the path to the pretrained model or instruct-tuned model
# w_bit: the weight bit-width for AWQ quantization
# q_group_size: the group size for quantization

# you could also first run the quantization and save the quantized model (--dump_awq awq_cache/$MODEL-w3-g128.pt), then load the quantized model for evaluation (--load_awq awq_cache/$MODEL-w3-g128.pt)

# general tasks
# --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge
python $DIRPATH/llm-awq/entry.py  --model_path $MODEL_PATH --w_bit $W_BIT --q_group_size 128 --run_awq --tasks piqa --num_fewshot 0

# MMLU
python $DIRPATH/llm-awq/entry.py  --model_path $MODEL_PATH --w_bit $W_BIT --q_group_size 128 --run_awq --tasks mmlu --num_fewshot 5 --mc_num 1

# GSM8K
python $DIRPATH/llm-awq/entry.py  --model_path $MODEL_PATH \
    --w_bit $W_BIT --q_group_size 128 --run_awq \
    --task gsm8k --gen_length 256 --steps 256 --block_length 32 --num_fewshot 4 \

# MATH 
python $DIRPATH/llm-awq/entry.py  --model_path $MODEL_PATH \
    --w_bit $W_BIT --q_group_size 128 --run_awq \
    --tasks minerva_math --gen_length 256 --steps 256 --block_length 64 --num_fewshot 0 \


# HumanEval
python $DIRPATH/llm-awq/entry.py  --model_path $MODEL_PATH \
    --w_bit $W_BIT --q_group_size 128 --run_awq \
    --task humaneval --gen_length 512 --steps 512 --block_length 32 --num_fewshot 0 \


# MBPP
python $DIRPATH/llm-awq/entry.py  --model_path $MODEL_PATH \
    --w_bit $W_BIT --q_group_size 128 --run_awq \
    --task mbpp --gen_length 512 --steps 512 --block_length 32 --num_fewshot 3 \