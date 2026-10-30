"""Accuracy cache validation, completion checks, and read-only GPU preflight."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess


RUN_FIELDS = ("model", "variant", "tasks", "num_fewshot", "limit", "steps",
              "gen_length", "block_length")
RUNTIME_FIELDS = ("torch", "cuda", "dtype", "autocast", "request_batch_size", "decoding")
SOURCE_PATHS = (
    "tools/qdlm_benchmark_core.py",
    "AutoGPTQ/auto_gptq/modeling/llada.py",
    "AutoGPTQ/auto_gptq/modeling/_base.py",
    "AutoGPTQ/auto_gptq/modeling/_utils.py",
    "AutoGPTQ/auto_gptq/mixed_precision/sandwich.py",
    "AutoGPTQ/auto_gptq/quantization/gptq.py",
    "AutoGPTQ/auto_gptq/quantization/quantizer.py",
    "AutoGPTQ/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py",
    "lm-evaluation-harness/lm_eval/evaluator.py",
)


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def request_key(request):
    return hashlib.sha256(json.dumps(
        [request.task_name, request.doc_id, request.idx, request.args],
        sort_keys=True).encode()).hexdigest()


def source_fingerprints(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def validate_settings(previous, current):
    for key in RUN_FIELDS:
        if key not in previous["args"] or previous["args"][key] != current["args"][key]:
            raise ValueError(f"Resume setting mismatch: {key}")
    for key in RUNTIME_FIELDS + ("config",):
        if key not in previous or previous[key] != current[key]:
            raise ValueError(f"Resume runtime/config mismatch: {key}")


def validate_sources(previous, current, root):
    old_hashes = previous.get("source_sha256")
    if old_hashes is None:
        revision = previous.get("source_revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Missing valid original source revision")
        old_hashes = {}
        for name in SOURCE_PATHS:
            if name in previous.get("local_changes", ""):
                raise ValueError(f"Original source was modified without a saved hash: {name}")
            blob = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
            old_hashes[name] = hashlib.sha256(blob).hexdigest()
    for name in SOURCE_PATHS:
        if old_hashes.get(name) != current["source_sha256"][name]:
            raise ValueError(f"Resume quantization/decoding source changed: {name}")


def load_records(path, gen_length, allow_torn_final=False):
    lines = path.read_bytes().splitlines(keepends=True)
    records = {}
    ignored_tail = False
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if allow_torn_final and index == len(lines) - 1 and not line.endswith(b"\n"):
                ignored_tail = True
                break
            raise ValueError(f"Invalid prediction JSON at line {index + 1}") from None
        if not isinstance(row, dict):
            raise ValueError(f"Prediction is not an object at line {index + 1}")
        key = row.get("case_id")
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError(f"Invalid case ID at line {index + 1}")
        if key in records:
            raise ValueError(f"Duplicate prediction: {key}")
        tokens = row.get("generated_token_ids")
        if (not isinstance(row.get("answer"), str) or not isinstance(row.get("task"), str)
                or "doc_id" not in row or not isinstance(tokens, list)
                or len(tokens) != gen_length
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError(f"Malformed prediction: {key}")
        records[key] = row
    return records, ignored_tail


def validate_requests(requests, records, previous=None):
    keys = [request_key(r) for r in requests]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate evaluation requests")
    if previous and previous.get("request_count") != len(keys):
        raise ValueError("Total request count changed from the original run")
    lookup = dict(zip(keys, requests))
    for key, row in records.items():
        if key not in lookup:
            raise ValueError(f"Cached prompt does not match current evaluation: {key}")
        request = lookup[key]
        if row["task"] != request.task_name or row["doc_id"] != request.doc_id:
            raise ValueError(f"Cached task/document mismatch: {key}")
    return keys


def replay_check_keys(keys, records):
    present = [key for key in keys if key in records]
    if not present:
        return []
    return list(dict.fromkeys(present[i] for i in (0, len(present) // 2, len(present) - 1)))


def verify_cached_predictions(requests, records, predict):
    keys = validate_requests(requests, records)
    lookup = dict(zip(keys, requests))
    checked = replay_check_keys(keys, records)
    for key in checked:
        actual = predict(lookup[key])
        old = records[key]
        if (actual["generated_token_ids"] != old["generated_token_ids"]
                or actual["answer"] != old["answer"]):
            raise ValueError(f"Rebuilt model disagrees with cached prediction {key}; refusing to mix results")
        print(f"Cached prediction check passed: {old['task']} doc={old['doc_id']}", flush=True)
    return checked


def stream_predictions(requests, records, predict, handle, advance):
    """Return answers in harness order and persist each case exactly once."""
    keys = validate_requests(requests, records)
    for key in keys:
        if key in records:
            handle.write(json.dumps(records[key]) + "\n")
    handle.flush()
    answers = []
    for request, key in zip(requests, keys):
        if key in records:
            row = records[key]
        else:
            row = predict(request)
            if row["case_id"] != key:
                raise ValueError("Generated prediction has the wrong case ID")
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            advance()
        answers.append(row["answer"])
    return answers


def model_fingerprint(model):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        for offset in range(0, raw.numel(), 8 * 1024 * 1024):
            chunk = raw[offset:offset + 8 * 1024 * 1024].cpu().numpy()
            digest.update(memoryview(chunk))
    return digest.hexdigest()


def validate_completion(out):
    metadata = json.loads((out / "metadata.json").read_text())
    accuracy = json.loads((out / "accuracy.json").read_text())
    complete = json.loads((out / "complete.json").read_text())
    expected = metadata["request_count"]
    if type(expected) is not int or expected <= 0:
        raise ValueError("Invalid expected request count")
    records, _ = load_records(out / "outputs.jsonl", metadata["args"]["gen_length"])
    if (len(records) != expected or complete.get("completed_requests") != expected
            or complete.get("status") != "complete" or metadata.get("status") != "complete"
            or not isinstance(accuracy.get("results"), dict) or not accuracy["results"]):
        raise ValueError("Evaluation is incomplete")
    for filename in ("outputs.jsonl", "accuracy.json"):
        actual = hashlib.sha256((out / filename).read_bytes()).hexdigest()
        if complete.get("sha256", {}).get(filename) != actual:
            raise ValueError(f"Completed result changed: {filename}")
    return expected


def require_idle_gpus():
    def query(fields, kind):
        text = subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text)) if row]
    selected = {uuid: index for index, uuid in query("index,uuid", "gpu") if index in ("2", "3")}
    if set(selected.values()) != {"2", "3"}:
        raise RuntimeError("Physical GPUs 2 and 3 were not found")
    busy = [(selected[uuid], pid) for uuid, pid in query("gpu_uuid,pid", "compute-apps")
            if uuid in selected]
    if busy:
        raise RuntimeError(f"Physical GPUs 2,3 already have compute processes: {busy}. Launch cancelled.")
    print("Physical GPUs 2,3 have no listed compute processes at this check.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-complete", type=Path)
    action.add_argument("--check-gpus", action="store_true")
    args = parser.parse_args()
    if args.check_gpus:
        require_idle_gpus()
    else:
        print(f"Completion verified: {validate_completion(args.check_complete)} predictions and accuracy.json")


if __name__ == "__main__":
    main()
