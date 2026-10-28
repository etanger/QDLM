"""CPU checks against the real legacy GPTQ packer and dense forward.

These do not claim to exercise FLUTE's CUDA kernel. The benchmark validates
that kernel against every original quantized linear on the server.
"""

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn

from qdlm_flute_backend import (
    FluteQuantLinear, inspect_source, reference_weight, tuning_rows, unpack_gptq_axis0,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "legacy_qlinear", ROOT / "AutoGPTQ/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def make_source(bits, k=128, n=96, bias=True):
    torch.manual_seed(42)
    group = 32
    codes = torch.randint(0, 1 << bits, (k, n), dtype=torch.uint8)
    # Exercise every code and the two cross-word W3 fields explicitly.
    codes[:32] = (torch.arange(32) % (1 << bits)).to(torch.uint8).unsqueeze(1)
    scales = (torch.rand(k // group, n) * 0.1 + 0.01).half()
    zeros = torch.full_like(scales, 1 << (bits - 1))
    weight = (codes.half() - (1 << (bits - 1))) * scales.repeat_interleave(group, dim=0)
    linear = nn.Linear(k, n, bias=bias, dtype=torch.float16)
    linear.weight.data.copy_(weight.T)
    if bias:
        linear.bias.data.uniform_(-0.1, 0.1)
    module = legacy.QuantLinear(bits, group, k, n, bias, kernel_switch_threshold=0)
    module.pack(linear, scales.T, zeros.T, None)
    return module, codes, weight


class FluteAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_codes_from_real_packer_including_w3_word_boundaries(self):
        for bits in (2, 3):
            with self.subTest(bits=bits):
                source, codes, _ = make_source(bits)
                self.assertTrue(bool((source.qweight < 0).any()))
                decoded = unpack_gptq_axis0(source.qweight, bits, source.infeatures)
                torch.testing.assert_close(decoded, codes, rtol=0, atol=0)

    def test_reference_weights_are_bit_exact(self):
        for bits in (2, 3):
            source, _, weight = make_source(bits)
            torch.testing.assert_close(reference_weight(source), weight, rtol=0, atol=0)

    def test_matches_actual_legacy_forward_and_bias(self):
        for bits in (2, 3):
            for bias in (False, True):
                source, _, _ = make_source(bits, bias=bias)
                for shape in ((1, 128), (19, 128), (2, 17, 128)):
                    x = torch.randn(shape, dtype=torch.float16)
                    expected = x @ reference_weight(source)
                    if bias:
                        expected += source.bias
                    torch.testing.assert_close(source(x), expected, rtol=0, atol=0)

    def test_lut_and_scale_layout_preserve_weights(self):
        for bits in (2, 3):
            source, codes, weight = make_source(bits)
            converted = FluteQuantLinear(source)
            reconstructed = converted.qmap[codes.long()] * converted.scales.T.repeat_interleave(32, 0)
            torch.testing.assert_close(reconstructed, weight, rtol=0, atol=0)
            torch.testing.assert_close(converted.bias, source.bias, rtol=0, atol=0)
            self.assertNotIn('_source_packed_cpu', dict(converted.named_buffers()))
            torch.testing.assert_close(converted._source_packed_cpu, source.qweight)

    def test_rejects_nonuniform_zeros(self):
        source, _, _ = make_source(3)
        source.qzeros[0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, 'zero point'):
            inspect_source(source)

    def test_rejects_activation_ordering(self):
        source, _, _ = make_source(2)
        source.g_idx = source.g_idx.flip(0)
        with self.assertRaisesRegex(ValueError, 'ordering'):
            inspect_source(source)

    def test_tuning_rows_do_not_exceed_real_m_above_tile_minimum(self):
        for m in (16, 31, 32, 64, 127, 128, 129, 1500, 4096):
            bucket = tuning_rows(m)
            self.assertLessEqual(bucket, m)
            self.assertLess(m, 2 * bucket)
        with self.assertRaises(ValueError):
            tuning_rows(0)

    def test_cold_forward_fails_instead_of_tuning_inside_timer(self):
        source, _, _ = make_source(2)
        converted = FluteQuantLinear(source)
        with self.assertRaisesRegex(RuntimeError, 'before generation'):
            converted(torch.randn(1, 128, dtype=torch.float16))


if __name__ == '__main__':
    unittest.main()
