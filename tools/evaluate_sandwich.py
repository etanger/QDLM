#!/usr/bin/env python3
"""Evaluate sandwich quantization accuracy without latency repeats or FP drivers."""

import argparse
from collections import Counter
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys

from qdlm_accuracy_resume import (
    load_records, model_fingerprint, request_key, source_fingerprints,
    stream_predictions, validate_completion, validate_requests, validate_settings,
    validate_sources, verify_cached_predictions, write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/qdlm_sandwich_w13_avg2.json")
    parser.add_argument("--variant", choices=["quantized", "fp16"], default="quantized")
    parser.add_argument("--tasks", default="mmlu_generative")
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--limit", type=int, default=80, help="Examples per subject; -1 uses all.")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--resume-from", help="Reuse validated predictions from an interrupted result directory.")
    args = parser.parse_args()
    if args.limit != -1 and args.limit <= 0:
        parser.error("limit must be positive or -1")
    if args.num_fewshot < 0 or min(args.steps, args.gen_length, args.block_length) <= 0:
        parser.error("Invalid few-shot count or decoding dimensions")
    if args.gen_length % args.block_length or args.steps % (args.gen_length // args.block_length):
        parser.error("gen-length must divide into blocks; steps must divide across blocks")
    return args


def main():
    args = parse_args()
    os.chdir(ROOT)
    # Enforce the physical devices authorized for this server, before CUDA init.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("Launch with CUDA_VISIBLE_DEVICES=2,3; other GPUs are not allowed.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Launch with python, not torchrun or accelerate launch.")
    out = Path(args.output).resolve()
    resume_from = Path(args.resume_from).resolve() if args.resume_from else None
    if resume_from == out:
        raise ValueError("Resume into a new output directory; keep the original results intact")
    previous = None
    cached = {}
    out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(ROOT / "AutoGPTQ"), str(ROOT / "lm-evaluation-harness")]
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    import auto_gptq
    import lm_eval
    from qdlm_benchmark_core import generate_clean

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly the two visible CUDA GPUs are required.")
    torch.cuda.set_device(0)
    for module, directory in [(auto_gptq, ROOT / "AutoGPTQ"),
                               (lm_eval, ROOT / "lm-evaluation-harness")]:
        if not Path(module.__file__).resolve().is_relative_to(directory):
            raise RuntimeError(f"Wrong source import: {module.__file__}")
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ["GPTQ_DUMP_CALIB_H"] = "0"
    config = Path(args.config).resolve()
    os.environ["QDLM_SANDWICH_CONFIG"] = str(config)
    os.environ["QDLM_SANDWICH_PLAN_OUT"] = str(out / "sandwich_plan.json")
    metadata = dict(args=vars(args), torch=torch.__version__, cuda=torch.version.cuda,
                    source_revision=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True).strip(),
                    local_changes=subprocess.check_output(
                        ["git", "status", "--short"], text=True),
                    config=json.loads(config.read_text()) if args.variant == "quantized" else None,
                    visible_devices="2,3", calibration_device="cuda:0 (physical GPU 2)",
                    inference_device="cuda:1 (physical GPU 3)",
                    dtype="float16", autocast=False, request_batch_size=1,
                    decoding=dict(temperature=0, cfg_scale=0, remasking="low_confidence",
                                  mask_id=126336, quant_start_step=0),
                    accuracy_only=True, latency_claim=False, status="preparing",
                    source_sha256=source_fingerprints(ROOT))
    if resume_from:
        previous = json.loads((resume_from / "metadata.json").read_text())
        validate_settings(previous, metadata)
        validate_sources(previous, metadata, ROOT)
        cached, ignored_tail = load_records(resume_from / "outputs.jsonl", args.gen_length,
                                            allow_torn_final=True)
        if not cached:
            raise ValueError("No valid saved predictions to resume")
        metadata["resume"] = dict(source=str(resume_from), cached_predictions=len(cached),
                                 ignored_incomplete_final_line=ignored_tail,
                                 source_outputs_sha256=hashlib.sha256(
                                     (resume_from / "outputs.jsonl").read_bytes()).hexdigest())
        write_json(out / "resume_source_metadata.json", previous)
        print(f"Resume: found {len(cached)} saved predictions in {resume_from}", flush=True)
        print("Rebuilding the model, then checking cached predictions before continuing.", flush=True)
    write_json(out / "metadata.json", metadata)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.variant == "quantized":
        model = AutoGPTQForCausalLM.from_pretrained(
            args.model, BaseQuantizeConfig(bits=2, group_size=128, desc_act=False, sym=True),
            trust_remote_code=True, torch_dtype=torch.float16)
        try:
            calibration_tokenizer = AutoTokenizer.from_pretrained(
                args.model, use_fast=False, trust_remote_code=True)
        except Exception:
            calibration_tokenizer = AutoTokenizer.from_pretrained(
                args.model, use_fast=True, trust_remote_code=True)
        train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        encoded = calibration_tokenizer("\n\n".join(train["text"]), return_tensors="pt").input_ids
        rng = random.Random(0)
        examples = []
        for _ in range(128):
            start = rng.randint(0, encoded.shape[1] - 2048 - 1)
            ids = encoded[:, start:start + 2048]
            examples.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids)))
        model.quantize(examples)
        metadata["packed_validation"] = model._sandwich_packed_summary
        metadata["quantized_modules"] = [dict(name=n, bits=m.bits, backend=type(m).__module__)
                                           for n, m in model.named_modules() if hasattr(m, "qweight")]
        metadata["unquantized_linear_modules"] = [n for n, m in model.named_modules()
                                                   if isinstance(m, torch.nn.Linear)]
        del examples, encoded, train, calibration_tokenizer
        gc.collect()
    else:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float16)
    model.to("cuda:1").eval()
    for device in range(2):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    # Legacy CUDA extensions launch on the current device/stream.
    torch.cuda.set_device(1)
    metadata["model_state_sha256"] = model_fingerprint(model)
    if previous:
        if previous.get("quantized_modules") != metadata.get("quantized_modules"):
            raise RuntimeError("Quantized module coverage/backend changed; cannot reuse cached answers")
        old_fingerprint = previous.get("model_state_sha256")
        if old_fingerprint and old_fingerprint != metadata["model_state_sha256"]:
            raise RuntimeError("Rebuilt model weights differ from the original fingerprint")
        metadata["resume"]["weight_fingerprint_match"] = True if old_fingerprint else None
        metadata["resume"]["verification_note"] = (
            "Exact state fingerprint checked; cached generations will also be checked." if old_fingerprint else
            "Original run has no weight fingerprint. Settings/core source and three cached generations are checked; this is not proof of full original weight equality.")
    write_json(out / "metadata.json", metadata)
    print("Evaluation: quantized model only" if args.variant == "quantized" else "Evaluation: FP16 baseline", flush=True)

    def predict(request):
        prompt, gen_kwargs = request.args
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:1")
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            generated = generate_clean(model, ids, steps=args.steps,
                                       gen_length=args.gen_length, block_length=args.block_length)
        tokens = generated[0, ids.shape[1]:].tolist()
        answer = tokenizer.decode(tokens, skip_special_tokens=False)
        stops = gen_kwargs.get("until", [])
        if isinstance(stops, str):
            stops = [stops]
        for stop in stops:
            if stop:
                answer = answer.split(stop)[0]
        answer = tokenizer.decode(tokenizer(answer)["input_ids"], skip_special_tokens=True)
        return dict(case_id=request_key(request), task=request.task_name,
                    doc_id=request.doc_id, answer=answer, generated_token_ids=tokens)

    class AccuracyLM(LM):
        def loglikelihood(self, requests):
            raise NotImplementedError("Use generation tasks with this runner.")

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("Use generation tasks with this runner.")

        def generate_until(self, requests):
            requests = list(requests)
            validate_requests(requests, cached, previous)
            metadata["requests_per_task"] = dict(Counter(r.task_name for r in requests))
            metadata["request_count"] = len(requests)
            metadata["status"] = "validating_cache" if cached else "generating"
            write_json(out / "metadata.json", metadata)
            checked = verify_cached_predictions(requests, cached, predict)
            if cached:
                metadata["resume"]["checked_case_ids"] = checked
                write_json(out / "resume_validation.json", dict(status="pass", checked_case_ids=checked))
            metadata["status"] = "generating"
            write_json(out / "metadata.json", metadata)
            print(f"Evaluation requests: {len(requests)} | reused: {len(cached)} | remaining: {len(requests) - len(cached)}", flush=True)
            with (out / "outputs.jsonl").open("x") as handle:
                with tqdm(total=len(requests), initial=len(cached), desc="Generating") as progress:
                    answers = stream_predictions(requests, cached, predict, handle,
                                                 lambda: progress.update(1))
            metadata["status"] = "scoring"
            write_json(out / "metadata.json", metadata)
            return answers

    results = evaluator.simple_evaluate(
        AccuracyLM(), tasks=args.tasks.split(","), num_fewshot=args.num_fewshot,
        limit=None if args.limit == -1 else args.limit, random_seed=0,
        numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234,
        log_samples=False, confirm_run_unsafe_code=True)
    write_json(out / "accuracy.json", results)
    metadata["status"] = "complete"
    write_json(out / "metadata.json", metadata)
    write_json(out / "complete.json", dict(
        status="complete", completed_requests=metadata["request_count"],
        sha256={name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                for name in ("outputs.jsonl", "accuracy.json")}))
    validate_completion(out)
    print(json.dumps(results["results"], indent=2, default=str), flush=True)
    print(f"Results saved to {out}", flush=True)


if __name__ == "__main__":
    main()
