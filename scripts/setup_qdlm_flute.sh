#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
export CUDA_VISIBLE_DEVICES=2,3
BASE_PYTHON="${BASE_PYTHON:-/home/zwang97/miniconda3/envs/qdlm/bin/python}"
FLUTE_REVISION=9eb83a12d56949bbe7fe9c836ba97a67bd1e3761
VENV_DIR="$ROOT/.venv/qdlm-flute"
FLUTE_DIR="$ROOT/build/flute-$FLUTE_REVISION"
CUTLASS_DIR="$ROOT/build/cutlass-v3.4.1"
STAMP="$VENV_DIR/qdlm-flute-build.txt"
BASE_TORCH=$("$BASE_PYTHON" -c 'import torch; print(torch.__version__)')
EXPECTED_STAMP="$FLUTE_REVISION $BASE_TORCH"
if [[ -f "$STAMP" && "$(cat "$STAMP")" == "$EXPECTED_STAMP" ]]; then
    "$VENV_DIR/bin/python" -c 'import flute, flute.utils; assert all(flute.utils.get_template_ids(b) for b in (2, 3)); print("FLUTE installation is ready")'
    exit 0
fi

# Match the existing qdlm PyTorch; do not install vLLM or replace PyTorch.
"$BASE_PYTHON" - <<'PY'
import re
import subprocess
import torch
from torch.utils.cpp_extension import CUDA_HOME
if CUDA_HOME is None or torch.version.cuda is None:
    raise RuntimeError("CUDA toolkit/nvcc is required; set CUDA_HOME to the toolkit matching torch.version.cuda")
version = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "--version"], text=True)
release = re.search(r"release (\d+\.\d+)", version).group(1)
if release != torch.version.cuda:
    raise RuntimeError(f"nvcc {release} differs from PyTorch CUDA {torch.version.cuda}; set CUDA_HOME to the matching toolkit")
print(f"Building against torch={torch.__version__}, CUDA={release}, CUDA_HOME={CUDA_HOME}")
PY

mkdir -p "$ROOT/build" "$ROOT/.venv"
"$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install ninja wheel jaxtyping click packaging 'setuptools<81'
if [[ ! -d "$CUTLASS_DIR/.git" ]]; then
    git clone --depth 1 --branch v3.4.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi
if [[ "$(git -C "$CUTLASS_DIR" rev-parse HEAD)" != "$(git -C "$CUTLASS_DIR" rev-parse 'v3.4.1^{commit}')" ]]; then
    echo "Unexpected CUTLASS checkout; expected v3.4.1."
    exit 1
fi
if [[ ! -d "$FLUTE_DIR/.git" ]]; then
    git init "$FLUTE_DIR"
    git -C "$FLUTE_DIR" remote add origin https://github.com/HanGuo97/flute.git
    git -C "$FLUTE_DIR" fetch --depth 1 origin "$FLUTE_REVISION"
    git -C "$FLUTE_DIR" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$FLUTE_DIR" rev-parse HEAD)" != "$FLUTE_REVISION" ]]; then
    echo "Unexpected FLUTE checkout."
    exit 1
fi
export QDLM_CUTLASS_PATH="$CUTLASS_DIR"
"$PYTHON" - "$FLUTE_DIR/setup.py" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
old = 'CUTLASS_PATH = "/workspace/cutlass/"'
new = 'CUTLASS_PATH = os.environ["QDLM_CUTLASS_PATH"]'
if old not in text and new not in text:
    raise RuntimeError("FLUTE setup.py changed; inspect CUTLASS_PATH before building")
path.write_text(text.replace(old, new))
PY
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST
TORCH_CUDA_ARCH_LIST=$("$PYTHON" - <<'PY'
import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError("Expected exactly physical GPUs 2,3")
caps = {torch.cuda.get_device_capability(i) for i in range(2)}
if any(major < 8 for major, minor in caps):
    raise RuntimeError("This FLUTE setup expects an Ampere-or-newer CUDA GPU")
print(";".join(f"{major}.{minor}" for major, minor in sorted(caps)))
PY
)
"$PYTHON" -m pip install --no-deps --no-build-isolation "$FLUTE_DIR"
"$PYTHON" -c 'import flute, flute.utils; assert all(flute.utils.get_template_ids(b) for b in (2, 3)); print("FLUTE W2/W3 import check passed")'
printf '%s\n' "$EXPECTED_STAMP" > "$STAMP"
