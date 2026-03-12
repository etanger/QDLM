export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

lm_eval --model dream_base \
    --model_args pretrained='/path/to/your/model' \
    --tasks piqa \
    --batch_size 8 \