# Quantize the model using AutoGPTQ
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH="/path/to/your/model"  # You could replace it with dream or dream-instruct model path

# model_path: the path to the pretrained model
# wbits: the weight bit-width for GPTQ quantization
# by default, we use 128 as the group size for GPTQ quantization

# general qa tasks
# --tasks piqa,winogrande,arc_easy,arc_challenge
python $DIRPATH/AutoGPTQ/quantize.py --model $MODEL_PATH --wbits 3 --tasks piqa


# for generation tasks, you may need to adjust the generation parameters (max_new_tokens, diffusion_steps) according to your needs