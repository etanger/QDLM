"""CPU checks for prompt selection, allocation guards, and packed cache fidelity."""
import json
from pathlib import Path
import tempfile
import unittest

from bench_humming_samples import check_specs, select_cases


class InputChecks(unittest.TestCase):
    def test_three_saved_real_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            cases = [dict(case_id=str(i), input_ids=list(range(i + 1)),
                          prompt=f"prompt {i}", latency_sample=True) for i in range(57)]
            cases.append(dict(case_id="not-a-latency-prompt", input_ids=[1], latency_sample=False))
            path.write_text("\n".join(map(json.dumps, reversed(cases))))
            selected = select_cases(path)
            self.assertEqual([c["case_id"] for c in selected], ["6", "28", "50"])
            self.assertEqual(selected, [cases[i] for i in (6, 28, 50)])
            path.write_text("\n".join(map(json.dumps, [cases[0], cases[0], cases[1]])))
            with self.assertRaises(ValueError):
                select_cases(path)

    def test_reject_wrong_allocation_despite_correct_average(self):
        specs = [dict(name=f"model.transformer.blocks.{i}.linear{j}",
                      bits=3 if 8 <= i < 24 else 2, group_size=128,
                      infeatures=512, outfeatures=512) for i in range(32) for j in range(4)]
        check_specs(specs)
        specs[0]["bits"], specs[32]["bits"] = specs[32]["bits"], specs[0]["bits"]
        with self.assertRaises(ValueError):
            check_specs(specs)

    def test_cpu_cache_preserves_w3_crossing_bit_fields(self):
        import torch
        from bench_flute_shape_ab import make_source
        from qdlm_flute_backend import unpack_gptq_axis0, reference_weight
        for bits in (2, 3):
            src = make_source(torch, bits, 512, 512)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "cache.pt"
                torch.save(vars(src), path)
                restored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                from types import SimpleNamespace
                self.assertTrue(torch.equal(reference_weight(src), reference_weight(SimpleNamespace(**restored))))
                codes = unpack_gptq_axis0(restored["qweight"], bits, 512)
                # Decode the entire packed stream independently as a Python integer.
                for col in (0, 17, 511):
                    for block in (0, 9, 15):
                        stream = sum((int(restored["qweight"][block * bits + j, col]) & 0xffffffff)
                                     << (32 * j) for j in range(bits))
                        expected = [(stream >> (bits * j)) & ((1 << bits) - 1) for j in range(32)]
                        self.assertEqual(codes[32 * block:32 * (block + 1), col].tolist(), expected)


if __name__ == "__main__":
    unittest.main()
