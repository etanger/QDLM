"""CPU-only regression checks for exact-M selection and aggregation."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from bench_flute_exact_m import aggregate, prepare_exact, prepare_recorded


class ExactMTests(unittest.TestCase):
    def test_exact_rows_reach_tuner_and_patch_is_restored(self):
        for m in (387, 544, 1049):
            old_rule = lambda rows: 1 << (rows.bit_length() - 1)
            backend = SimpleNamespace(tuning_rows=old_rule)
            tune = SimpleNamespace(get_template_key=Mock(return_value=(m,)),
                _TEMPLATES={(m,): 5}, _tune=Mock(return_value=9))
            flute = SimpleNamespace(tune=tune,
                utils=SimpleNamespace(get_device_num_sms=lambda device: 142))
            layer = SimpleNamespace(outfeatures=12288, infeatures=4096,
                bits=3, group_size=128,
                scales=SimpleNamespace(device="cuda:0", dtype="float16"))
            layer.prepare = lambda rows: flute.tune._tune(M=backend.tuning_rows(rows))
            prepare_exact(layer, backend, flute, m)
            tune._tune.assert_called_once_with(M=m)
            self.assertNotIn((m,), tune._TEMPLATES)
            self.assertIs(backend.tuning_rows, old_rule)

    def test_recorded_template_does_not_retune(self):
        tune = SimpleNamespace(_tune=Mock(side_effect=AssertionError("must not tune")))
        flute = SimpleNamespace(tune=tune)
        actual = []
        layer = SimpleNamespace(prepare=lambda m: actual.append(tune._tune(M=m)))
        original = tune._tune
        prepare_recorded(layer, flute, 544, 7)
        self.assertEqual(actual, [7])
        self.assertIs(tune._tune, original)

    def test_aggregate_is_ratio_of_weighted_times(self):
        rows = [dict(M=544, count=16, after_sweep=dict(ms=dict(fp16=1, recorded=2, exact_m=1))),
                dict(M=544, count=32, after_sweep=dict(ms=dict(fp16=3, recorded=4, exact_m=2))),
                dict(M=1049, count=16, after_sweep=dict(ms=dict(fp16=10, recorded=20, exact_m=5)))]
        summary = aggregate(rows, [544, 1049])
        self.assertAlmostEqual(summary[544]["recorded_speedup"], 112 / 160)
        self.assertAlmostEqual(summary[544]["exact_m_speedup"], 112 / 80)
        self.assertAlmostEqual(summary[544]["tuning_gain"], 160 / 80)
        self.assertEqual(summary[1049]["exact_m_speedup"], 2)


if __name__ == "__main__":
    unittest.main()
