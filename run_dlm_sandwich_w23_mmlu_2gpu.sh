#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/nvme/zwang97/QDLM
mkdir -p logs results/mixed_precision

RUN_NAME="dlm_sandwich_w23_mmlu_2gpu"

exec 9>"logs/${RUN_NAME}.lock"
if ! flock -n 9; then
    echo "This experiment is already running. Skipping duplicate launch."
    exit 1
fi

echo "$$" > "logs/${RUN_NAME}.pid"
rm -f "logs/${RUN_NAME}.exitcode"

finish() {
    local rc=$?
    echo "$rc" > "logs/${RUN_NAME}.exitcode"
    echo "Finished: $(date -Is) | exit_code=$rc"
}
trap finish EXIT

/home/zwang97/miniconda3/envs/qdlm/bin/python - <<'PY_CHECK'
import ast
from pathlib import Path

flags = {
    "pair_analysis", "perturb_analysis", "c_analysis",
    "saliency_analysis", "sp_analysis", "recovery_analysis",
}
path = Path("lm-evaluation-harness/lm_eval/models/llada.py")
tree = ast.parse(path.read_text(encoding="utf-8"))
seen = set()

for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        continue

    for target in targets:
        if isinstance(target, ast.Name) and target.id in flags:
            seen.add(target.id)
            if (
                not isinstance(node.value, ast.Constant)
                or node.value.value is not False
            ):
                raise SystemExit(
                    f"Analysis flag is not disabled: {target.id}"
                )

if seen != flags:
    raise SystemExit(
        f"Analysis flag assignments not found: {sorted(flags - seen)}"
    )

print("Preflight passed: all six analysis flags are disabled.")
PY_CHECK

export CUDA_VISIBLE_DEVICES=2,3
export QDLM_SANDWICH_CONFIG=configs/qdlm_sandwich_w23.json
export QDLM_SANDWICH_PLAN_OUT="results/mixed_precision/${RUN_NAME}_plan.json"
export PYTHONUNBUFFERED=1

echo "Started: $(date -Is) | PID: $$"
echo "Mode: quant-only | GPUs: $CUDA_VISIBLE_DEVICES"

/home/zwang97/miniconda3/envs/qdlm/bin/python -u AutoGPTQ/quantize.py \
    --model /workspace/hdd/datasets/zwang97/models/LLaDA-8B-Instruct \
    --wbits 2 \
    --tasks mmlu_generative \
    --num_fewshot 5 \
    --steps 128 \
    --gen_length 64 \
    --block_length 64 \
    --limit 80 \
    --quant_start_step 0
