
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH="/path/to/your/model"  # You could replace it with dream or dream-instruct model path

# model_path: the path to the pretrained model
# block_size: the block size for DuQuant rotation
# max_rotation_step: the greedy rotation step for DuQuant quantization
# --swc: the ratio of weight clipping
# --lac: the ratio of activation clipping
# we use asymmetric per-channel quantization for weights and per-tensor quantization for activations in DuQuant

python $DIRPATH/DuQuant/generate_act_scale_shift.py --model $MODEL_PATH

# general qa tasks
# --tasks piqa,winogrande,arc_easy,arc_challenge
python $DIRPATH/DuQuant/main.py \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits 8 \
    --abits 8 \
    --model $MODEL_PATH \
    --task piqa \
    --quant_method duquant \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \