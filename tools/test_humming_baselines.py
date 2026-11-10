"""CPU tests for control coverage and experiment result integrity."""
import copy
import math
import unittest
from unittest.mock import patch

import bench_humming_baselines as bench


def reference_specs():
    specs = []
    for layer in range(32):
        for name, k, n in (("att_proj", 4096, 12288), ("attn_out", 4096, 4096),
                           ("ff_proj", 12288, 4096), ("up_proj", 4096, 12288)):
            specs.append({"name": f"model.transformer.blocks.{layer}.{name}",
                "bits": 3 if 8 <= layer < 24 else 2, "infeatures": k,
                "outfeatures": n, "group_size": 128, "has_bias": False})
    return specs


class ControlsTest(unittest.TestCase):
    def test_plan_pack_interface_and_parameter_budget(self):
        weights = [100 + i for i in range(32)]
        for bits in (2, 3):
            plan = bench.UniformPlan(bits, weights)
            self.assertEqual(sum(b * n for b, n in zip(plan.current_bits, plan.layer_weights)),
                             bits * sum(weights))
            self.assertEqual(set(plan.current_layer_bit_map.values()), {bits})
            self.assertEqual(getattr(plan, f"current_w{bits}_layers"), tuple(range(32)))
            self.assertEqual(plan.current_w1_layers, ())
            self.assertEqual(plan.to_dict({"status": "pass"})["packed_validation"]["status"], "pass")
        with self.assertRaises(ValueError):
            bench.UniformPlan(3, weights[:-1])

    def test_reject_wrong_coverage_even_if_bit_histogram_matches(self):
        reference = reference_specs()
        uniform = [dict(s, bits=3) for s in reference]
        bench.validate_uniform_specs(uniform, 3, reference)
        for mutation in (lambda s: s.update(name="other.module"),
                         lambda s: s.update(infeatures=2048), lambda s: s.update(group_size=64),
                         lambda s: s.update(bits=2)):
            changed = copy.deepcopy(uniform)
            mutation(changed[0])
            with self.assertRaises(ValueError):
                bench.validate_uniform_specs(changed, 3, reference)

    def test_scoped_checker_does_not_recurse_or_leak(self):
        reference = reference_specs()
        uniform = [dict(s, bits=2) for s in reference]
        original = bench.sample.check_specs
        replacement = lambda specs: bench.validate_uniform_specs_unpatched(specs, 2, reference)
        with patch.object(bench.sample, "check_specs", replacement):
            bench.sample.check_specs(uniform)
            self.assertIs(bench.sample.check_specs, replacement)
        self.assertIs(bench.sample.check_specs, original)

    def test_selection_is_fixed_and_rejects_missing_subjects(self):
        pool = [{"case_id": str(i), "task": f"subject{i}", "input_ids": [1] * (50 + i),
                 "latency_sample": True} for i in range(57)]
        chosen = bench.select_latency(pool)
        self.assertEqual(len(chosen), 9)
        self.assertEqual(chosen, bench.select_latency(pool[::-1]))
        with self.assertRaises(ValueError):
            bench.select_latency(pool[:-1])

    def test_pairing_uses_prompt_medians_and_same_gpu(self):
        cases = [{"case_id": "short"}, {"case_id": "long"}]
        rows = []
        for c, fp, q in (("short", [1, 2, 100], [1, 1, 20]),
                         ("long", [4, 6, 100], [2, 3, 20])):
            for variant, times in (("fp16", fp), ("uniform_w3", q)):
                for repeat, value in enumerate(times):
                    rows.append({"case_id": c, "variant": variant, "repeat": repeat,
                        "physical_gpu": 3, "generation_seconds": value, "request_seconds": value})
        result = bench.summarize_pairs(cases, rows, "uniform_w3", 3)
        self.assertEqual(result["request_seconds"]["speedup"], 2.0)
        for bad in (rows[:-1], rows + rows[:1], [dict(rows[0], physical_gpu=2)] + rows[1:],
                    [dict(rows[0], request_seconds=math.nan)] + rows[1:],
                    [dict(rows[0], request_seconds=0.5)] + rows[1:]):
            with self.assertRaises(ValueError):
                bench.summarize_pairs(cases, bad, "uniform_w3", 3)

    def test_accuracy_completeness_and_token_payload(self):
        cases = [{"case_id": "a"}, {"case_id": "b"}]
        rows = [{"case_id": c["case_id"], "variant": "uniform_w2", "tokens": [42] * 64}
                for c in cases]
        bench.validate_outputs(cases, rows, "uniform_w2")
        for bad in (rows[:1], rows + rows[:1], [dict(rows[0], tokens=[42])] + rows[1:],
                    [dict(rows[0], variant="uniform_w3")] + rows[1:]):
            with self.assertRaises(ValueError):
                bench.validate_outputs(cases, bad, "uniform_w2")


if __name__ == "__main__":
    unittest.main()
