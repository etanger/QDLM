"""CPU checks against the unmodified server decoder and paired statistics."""

import ast
import contextlib
from datetime import datetime
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from qdlm_benchmark_core import generate_clean, select_latency_cases, summarize_latency, transfer_schedule

ROOT = Path(__file__).resolve().parents[1]


def load_reference():
    # Execute the exact original function bodies, without requiring the
    # transformers/datasets packages merely to test token transition semantics.
    tree = ast.parse((ROOT / "lm-evaluation-harness/lm_eval/models/llada.py").read_text())
    names = {"add_gumbel_noise", "get_num_transfer_tokens", "generate"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(torch=torch, F=F, np=np, os=os, json=json, time=time,
                     datetime=datetime, Path=Path)
    exec(compile(tree, "server_llada.py", "exec"), namespace)
    return namespace["generate"]


class ToyLLaDA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.transformer = torch.nn.Module()
        self.model.transformer.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.inputs = []
        self.device = torch.device("cpu")

    def forward(self, x):
        self.inputs.append(x.clone())
        hidden = x.float().unsqueeze(-1).expand(-1, -1, 4)
        hidden = self.model.transformer.blocks[0](hidden)
        positions = torch.arange(x.shape[1]).reshape(1, -1, 1)
        vocab = torch.arange(32).reshape(1, 1, -1)
        logits = torch.cos(hidden[..., :1] * 0.13 + positions * 0.17 + vocab * 0.29 + x.sum() * 0.11)
        logits[..., 31] = -100
        return SimpleNamespace(logits=logits)


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = staticmethod(load_reference())

    def compare(self, prompt, steps, gen_length, block_length):
        old, clean = ToyLLaDA(), ToyLLaDA()
        prompt = torch.tensor([prompt])
        kwargs = dict(steps=steps, gen_length=gen_length, block_length=block_length, mask_id=31)
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            for name, value in [("is_available", True), ("device_count", 2),
                                ("memory_allocated", 0),
                                ("get_device_properties", SimpleNamespace(total_memory=1))]:
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            stack.enter_context(patch.dict(os.environ, {"LLADA_DUMP_FULL_ACT": "0"}))
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    expected = self.reference(old, prompt, **kwargs)
            finally:
                os.chdir(original_cwd)
        actual = generate_clean(clean, prompt, **kwargs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(clean.inputs), steps)
        self.assertEqual(len(old.inputs), steps)
        for a, b in zip(old.inputs, clean.inputs):
            self.assertTrue(torch.equal(a, b), "Denoising trajectories differ")

    def test_exact_server_128_step_64_token_trajectory(self):
        self.compare([2, 7, 4], 128, 64, 64)

    def test_fewer_steps_than_tokens(self):
        self.compare([2, 7, 4], 4, 12, 12)

    def test_multiple_blocks(self):
        self.compare([2, 7, 4], 12, 12, 4)

    def test_prompt_contains_mask(self):
        self.compare([2, 31, 4], 12, 12, 4)

    def test_zero_transfer_steps_are_preserved(self):
        schedule = transfer_schedule(64, 128)
        self.assertEqual(schedule, [1] * 64 + [0] * 64)

    def test_stratified_latency_selection(self):
        cases = [{"task": task, "doc_id": i} for task in ("a", "b") for i in range(80)]
        chosen = select_latency_cases(cases, 1)
        self.assertEqual(chosen, [{"task": "a", "doc_id": 40}, {"task": "b", "doc_id": 40}])
        self.assertEqual(len(select_latency_cases(cases, 100)), 160)

    def make_rows(self):
        rows = []
        # Different devices have different speeds; only within-device ratios
        # are meaningful. Median removes one large outlier per prompt.
        for device, factor in [("cuda:0", 1), ("cuda:1", 3)]:
            for case, scale in [("a", 1), ("b", 2)]:
                for variant, latency in [("fp16", 2), ("sandwich_w23", 1)]:
                    for outlier in (1, 1, 100):
                        seconds = factor * scale * latency * outlier
                        rows.append(dict(case_id=case, device=device, variant=variant,
                                         phase="latency", generation_seconds=seconds,
                                         request_seconds=seconds))
        return rows

    def test_same_gpu_paired_medians(self):
        summary = summarize_latency(self.make_rows())
        self.assertEqual(summary["combined"]["generation_seconds"]["speedup"], 2)
        for result in summary["by_device"].values():
            self.assertEqual(result["generation_seconds"]["speedup"], 2)

    def test_missing_repeat_rejected(self):
        rows = self.make_rows()
        rows.pop()
        with self.assertRaises(ValueError):
            summarize_latency(rows)


if __name__ == "__main__":
    unittest.main()
