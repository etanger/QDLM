"""CPU tests that prevent incomplete or unpaired full-run speed claims."""
from collections import Counter
import math
import unittest

from bench_humming_full import split_cases, summarize, validate_cases


def inputs():
    return [dict(case_id=f"{task}-{doc}", task=f"subject_{task:02}", doc_id=doc,
                 prompt=f"question {doc}", input_ids=[1, 2, 3], until=[])
            for task in range(57) for doc in range(80)]


def measurements(cases, repeats=1):
    rows = []
    for gpu, shard in split_cases(cases).items():
        for case in shard:
            for variant in ("fp16", "sandwich_w23"):
                for repeat in range(repeats):
                    factor = 2 if variant == "fp16" else 1
                    # Different GPU speeds and a repeated timing outlier must
                    # not turn the aggregate into a mean of speedup ratios.
                    elapsed = factor * (gpu + case["doc_id"] + 1) * (100 if repeat == 2 else 1)
                    rows.append(dict(case_id=case["case_id"], physical_gpu=gpu,
                                     variant=variant, repeat=repeat,
                                     generation_seconds=elapsed, request_seconds=elapsed + 0.1))
    return rows


class FullBenchmarkChecks(unittest.TestCase):
    def test_standard_slice_and_disjoint_balanced_shards(self):
        cases = inputs()
        validate_cases(cases)
        shards = split_cases(list(reversed(cases)))
        self.assertEqual([len(shards[gpu]) for gpu in (2, 3)], [2280, 2280])
        self.assertFalse({c["case_id"] for c in shards[2]} & {c["case_id"] for c in shards[3]})
        for shard in shards.values():
            self.assertEqual(set(Counter(c["task"] for c in shard).values()), {40})
        with self.assertRaises(ValueError):
            validate_cases(cases[:-1])
        cases[-1]["case_id"] = cases[0]["case_id"]
        with self.assertRaises(ValueError):
            validate_cases(cases)

    def test_total_time_and_median_aggregation(self):
        cases = inputs()
        one = summarize(cases, measurements(cases), 1)
        three = summarize(cases, measurements(cases, 3), 3)
        self.assertEqual(one["paired_prompts"], 4560)
        self.assertEqual(one["timing_rows"], 9120)
        self.assertEqual(one["latency"], three["latency"])
        self.assertEqual(one["latency"]["combined"]["generation_seconds"]["speedup"], 2)
        self.assertEqual(len(one["latency"]["by_task"]), 57)
        expected_fp = sum(r["request_seconds"] for r in measurements(cases) if r["variant"] == "fp16")
        self.assertAlmostEqual(one["latency"]["combined"]["request_seconds"]["fp16_seconds"], expected_fp)

    def test_reject_partial_duplicate_or_wrong_gpu_rows(self):
        cases = inputs()
        rows = measurements(cases)
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                summarize(cases, bad, 1)
        rows[0] = dict(rows[0], physical_gpu=3)
        with self.assertRaises(ValueError):
            summarize(cases, rows, 1)

    def test_reject_invalid_durations(self):
        cases = inputs()
        for value in (0, -1, math.nan, math.inf):
            rows = measurements(cases)
            rows[0]["request_seconds"] = value
            with self.assertRaises(ValueError):
                summarize(cases, rows, 1)


if __name__ == "__main__":
    unittest.main()
