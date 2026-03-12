# Quantize the model using AWQ
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH="/path/to/your/model"  # You could replace it with dream or dream-instruct model path

# model_path: the path to the pretrained model
# w_bit: the weight bit-width for AWQ quantization
# q_group_size: the group size for quantization

# you could also first run the quantization and save the quantized model (--dump_awq awq_cache/$MODEL-w3-g128.pt), then load the quantized model for evaluation (--load_awq awq_cache/$MODEL-w3-g128.pt)

# general qa tasks
# --tasks piqa,winogrande,arc_easy,arc_challenge
python $DIRPATH/llm-awq/entry.py  --model_path $MODEL_PATH --w_bit 3 --q_group_size 128 --run_awq --tasks piqa --num_fewshot 0


# for generation tasks, you may need to adjust the generation parameters (max_new_tokens, diffusion_steps) according to your needs