#!/usr/bin/env bash
# Check once now, then every three hours until physical GPUs 2 and 3 are idle.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONUNBUFFERED=1
export MODE=latency
export BASE_PYTHON="${BASE_PYTHON:-python}"
export RUN_NAME="${RUN_NAME:-dlm_w23_flute_wait_$(date +%Y%m%d_%H%M%S)}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-10800}"
: "${ACCURACY_RESULT:?Set ACCURACY_RESULT to the accuracy result directory}"
if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ || ! "$CHECK_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid RUN_NAME or CHECK_INTERVAL_SECONDS."
    exit 2
fi
if [[ ! -f tools/qdlm_accuracy_resume.py || ! -f run_dlm_sandwich_w23_flute_2gpu.sh ]]; then
    echo "Required completion checker or FLUTE launcher is missing."
    exit 1
fi
mkdir -p logs
exec 7>logs/dlm_sandwich_flute_wait.lock
if ! flock -n 7; then
    echo "A GPU waiter is already running. Launch cancelled."
    exit 1
fi
echo "$$" > "logs/$RUN_NAME.pid"
finish() {
    local rc=$?
    echo "$rc" > "logs/$RUN_NAME.exitcode"
    echo "Waiter finished: $(date -Is) | exit_code=$rc"
}
trap finish EXIT
echo "Waiting for physical GPUs 2,3 | check interval: $CHECK_INTERVAL_SECONDS seconds"
echo "Accuracy result to verify before launch: $ACCURACY_RESULT"

while true; do
    echo "Checking GPUs: $(date -Is)"
    if "$BASE_PYTHON" - <<'PY'
import csv
import io
import subprocess
import sys

def query(kind, fields):
    result = subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True, timeout=30)
    return [[value.strip() for value in row]
            for row in csv.reader(io.StringIO(result)) if row]

selected = {uuid: index for index, uuid in query("gpu", "index,uuid")
            if index in ("2", "3")}
if set(selected.values()) != {"2", "3"}:
    raise SystemExit("Physical GPUs 2 and 3 were not found; stopping.")
busy = [(selected[uuid], pid) for uuid, pid in query("compute-apps", "gpu_uuid,pid")
        if uuid in selected]
if busy:
    print("Busy: " + ", ".join(f"GPU {gpu} PID {pid}" for gpu, pid in busy), flush=True)
    sys.exit(2)
print("Physical GPUs 2 and 3 have no listed compute processes.", flush=True)
PY
    then
        "$BASE_PYTHON" tools/qdlm_accuracy_resume.py --check-complete "$ACCURACY_RESULT"
        echo "Accuracy complete and GPUs idle. Starting FLUTE: $(date -Is)"
        # Keep the waiter lock across exec; the launcher also has its own locks.
        exec bash run_dlm_sandwich_w23_flute_2gpu.sh
    else
        rc=$?
        if [[ "$rc" -ne 2 ]]; then
            echo "GPU query failed; refusing to launch."
            exit "$rc"
        fi
    fi
    echo "Next GPU check in $CHECK_INTERVAL_SECONDS seconds."
    sleep "$CHECK_INTERVAL_SECONDS"
done
