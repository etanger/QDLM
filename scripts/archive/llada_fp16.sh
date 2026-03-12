export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

lm_eval --model llada_dist \
    --model_args model_path='/mnt/yichen/wyc/llada-8b-instruct' \
    --tasks piqa \
    --batch_size 8 \