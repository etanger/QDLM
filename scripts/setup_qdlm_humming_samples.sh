#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_PYTHON=/home/zwang97/miniconda3/envs/qdlm/bin/python
VENV="$PWD/.venv/qdlm-humming"
REV=3f2fa2238cb87fe1ac77aa9d51d3929d143e3c61
STAMP="$VENV/.qdlm_setup_$REV"
if [[ -f "$STAMP" ]]; then
    echo "Using existing isolated Humming environment."
    exit 0
fi
export PIP_CACHE_DIR=/workspace/hdd/users/zwang97/QDLM/cache/pip
mkdir -p "$PIP_CACHE_DIR"
echo "Installing an isolated Humming environment; the qdlm environment is unchanged."
"$BASE_PYTHON" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128 --upgrade
"$VENV/bin/python" -m pip install ninja "humming-kernels[cu12] @ git+https://github.com/vllm-project/humming.git@$REV"
"$VENV/bin/python" - <<'PY'
import torch
from pathlib import Path
import sys
assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), torch.__file__
assert torch.version.cuda == "12.8", torch.version.cuda
import transformers, accelerate, datasets
print("Humming runtime:", torch.__version__, torch.version.cuda, flush=True)
PY
touch "$STAMP"
