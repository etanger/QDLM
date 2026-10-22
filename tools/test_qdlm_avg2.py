"""CPU tests of actual GPTQ, mixed-backend packing, and exact avg2 allocation.

Load the real source modules without AutoGPTQ's optional top-level integrations.
No quantization or packing functions are mocked.
"""

from dataclasses import replace
import importlib
import io
from pathlib import Path
import sys
import types
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "AutoGPTQ" / "auto_gptq"
for suffix in ("", ".quantization", ".modeling", ".utils", ".mixed_precision",
               ".nn_modules", ".nn_modules.qlinear"):
    name = "_qdlm_test" + suffix
    package = types.ModuleType(name)
    package.__path__ = [str(SOURCE.joinpath(*suffix.lstrip(".").split(".")))] if suffix else [str(SOURCE)]
    sys.modules[name] = package

Quantizer = importlib.import_module("_qdlm_test.quantization.quantizer").Quantizer
GPTQ = importlib.import_module("_qdlm_test.quantization.gptq").GPTQ
BinaryLinear = importlib.import_module("_qdlm_test.nn_modules.qlinear.qlinear_binary").QuantLinear
packing = importlib.import_module("_qdlm_test.modeling._utils")
sandwich = importlib.import_module("_qdlm_test.mixed_precision.sandwich")
import_utils = importlib.import_module("_qdlm_test.utils.import_utils")


class Avg2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_binary_has_two_symmetric_levels_and_optimal_scale(self):
        quantizer = Quantizer()
        quantizer.configure(1, perchannel=True, sym=True)
        weight = torch.tensor([[-4., -2., 0., 1., 3.], [0., 0., 0., 0., 0.]])
        quantizer.find_params(weight, weight=True)
        actual = quantizer.quantize(weight)
        torch.testing.assert_close(actual[0], torch.tensor([-2., -2., 2., 2., 2.]))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.all(quantizer.zero == 0.5))
        self.assertTrue(quantizer.ready())
        with self.assertRaises(ValueError):
            quantizer.configure(1, sym=False)
        with self.assertRaises(ValueError):
            quantizer.configure(1, trits=True)

    def test_w2_w3_quantizer_grid_is_unchanged(self):
        weight = torch.tensor([[-2., -0.7, 0.2, 1.3], [-0.2, 0.1, 0.5, 1.]])
        for bits in (2, 3):
            quantizer = Quantizer()
            quantizer.configure(bits, perchannel=True)
            quantizer.find_params(weight, weight=True)
            scale = 2 * weight.abs().amax(dim=1, keepdim=True) / (2 ** bits - 1)
            zero = 2 ** (bits - 1)
            expected = scale * ((weight / scale).round().add(zero).clamp(0, 2 ** bits - 1) - zero)
            torch.testing.assert_close(quantizer.quantize(weight), expected)

    def test_binary_packing_padding_sign_bit_groups_bias_and_state(self):
        torch.manual_seed(7)
        linear = nn.Linear(70, 11, bias=True)
        groups = torch.arange(70).flip(0) // 16
        alpha = torch.rand(11, 5) + 0.01
        signs = torch.randint(0, 2, (11, 70)) * 2 - 1
        signs[:, 31] = 1
        signs[:, 63] = -1
        linear.weight.data.copy_(signs * alpha[:, groups])
        packed = BinaryLinear(1, 16, 70, 11, True, weight_dtype=torch.float32)
        packed.pack(linear, 2 * alpha, torch.full_like(alpha, 0.5), groups)
        self.assertEqual(packed.qweight.numel() * 4, 3 * 11 * 4)
        torch.testing.assert_close(packed.dequantize_weight().T, linear.weight, rtol=0, atol=0)
        for shape in ((70,), (4, 70), (2, 3, 70)):
            x = torch.randn(shape)
            torch.testing.assert_close(packed(x), linear(x))
        buffer = io.BytesIO()
        torch.save(packed.state_dict(), buffer)
        buffer.seek(0)
        restored = BinaryLinear(1, 16, 70, 11, True, weight_dtype=torch.float32)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(restored(x), linear(x))
        with self.assertRaises(ValueError):
            packed.pack(linear, 2 * alpha, torch.ones_like(alpha), groups)

    def test_real_gptq_binary_roundtrip_with_grouping_and_act_order(self):
        for actorder, static_groups in ((False, False), (True, False), (True, True)):
            torch.manual_seed(12)
            linear = nn.Linear(64, 32, bias=False)
            gptq = GPTQ(linear)
            gptq.quantizer.configure(1, perchannel=True, sym=True)
            x = torch.randn(4, 40, 64)
            x[..., 1::2] += 2 * x[..., ::2]
            gptq.add_batch(x, linear(x))
            scales, zeros, groups = gptq.fasterquant(
                group_size=16, actorder=actorder, static_groups=static_groups)
            packed = BinaryLinear(1, 16, 64, 32, False, weight_dtype=torch.float32)
            packed.pack(linear, scales, zeros, groups)
            torch.testing.assert_close(packed.dequantize_weight().T, linear.weight, rtol=0, atol=0)
            torch.testing.assert_close(packed(x), linear(x))
            self.assertTrue(torch.all(torch.isfinite(packed(x))))

    def test_binary_fp16_packing_preserves_quantized_weights(self):
        torch.manual_seed(19)
        linear = nn.Linear(64, 32, bias=True, dtype=torch.float16)
        gptq = GPTQ(linear)
        gptq.quantizer.configure(1, perchannel=True, sym=True)
        x = torch.randn(2, 40, 64, dtype=torch.float16)
        gptq.add_batch(x, linear(x))
        scale, zero, groups = gptq.fasterquant(group_size=32)
        packed = BinaryLinear(1, 32, 64, 32, True)
        packed.pack(linear, scale, zero, groups)
        torch.testing.assert_close(packed.dequantize_weight().T, linear.weight, rtol=0, atol=0)
        torch.testing.assert_close(packed(x), linear(x), rtol=0.002, atol=0.001)

    def test_exact_w13_and_w123_and_existing_w23_allocations(self):
        for filename, expected in (
            ("qdlm_sandwich_w13_avg2.json", [1] * 8 + [3] * 16 + [1] * 8),
            ("qdlm_sandwich_w123_avg2.json", [1] * 4 + [2] * 8 + [3] * 8 + [2] * 8 + [1] * 4),
            ("qdlm_sandwich_w23.json", [2] * 12 + [3] * 8 + [2] * 12),
        ):
            config = sandwich.load_sandwich_config(ROOT / "configs" / filename)
            plan = sandwich.build_sandwich_plan(config, [4096] * 32)
            self.assertEqual(list(plan.current_bits), expected)
            self.assertEqual(plan.current_average_bits, sum(expected) / 32)
            self.assertEqual(plan.to_dict()["final_avg2_preview"]["executed"], config.stage == "w123")

    def test_nonuniform_parameter_budget_cannot_silently_round(self):
        config = sandwich.load_sandwich_config(ROOT / "configs/qdlm_sandwich_w13_avg2.json")
        with self.assertRaisesRegex(ValueError, "exact avg2"):
            sandwich.build_sandwich_plan(config, [4097] + [4096] * 31)

    def test_real_mixed_gptq_pack_forward_and_budget(self):
        torch.manual_seed(20)
        config = sandwich.load_sandwich_config(ROOT / "configs/qdlm_sandwich_w123_avg2.json")
        config = replace(config, num_w3_layers=2, min_edge_layers=1)
        plan = sandwich.build_sandwich_plan(config, [64 * 64] * 8)
        self.assertEqual(plan.current_bits, (1, 2, 2, 3, 3, 2, 2, 1))
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Sequential(nn.Linear(64, 64, bias=True)) for _ in range(8)])
        quantizers, expected = {}, {}
        x = torch.randn(2, 64, 64)
        for index, bits in enumerate(plan.current_bits):
            linear = model.blocks[index][0]
            gptq = GPTQ(linear)
            gptq.quantizer.configure(bits, perchannel=True, sym=True)
            gptq.add_batch(x, linear(x))
            scale, zero, groups = gptq.fasterquant(group_size=32)
            name = f"blocks.{index}.0"
            quantizers[name] = (gptq.quantizer, scale, zero, groups)
            # Float32 storage isolates bit packing from FP16 scale rounding.
            expected[name] = linear(x).detach().clone()
        packing.pack_model(model, quantizers, bits=2, group_size=32, desc_act=False)
        summary = sandwich.validate_packed_model(model, quantizers, plan, "blocks")
        self.assertEqual(summary["quantized_weight_average_bits"], 2.0)
        self.assertEqual(summary["actual_histogram"], {"1": 2, "2": 4, "3": 2})
        self.assertEqual(summary["packed_weight_bytes_including_padding"], 8 * 64 * 64 * 2 // 8)
        for name, module in model.named_modules():
            if name in expected:
                torch.testing.assert_close(module(x), expected[name], atol=1e-5, rtol=1e-5)
        with self.assertRaises(RuntimeError):
            sandwich.validate_packed_model(model, {}, plan, "blocks")

    def test_w1_dispatch_does_not_silently_use_other_backend(self):
        self.assertIs(import_utils.dynamically_import_QuantLinear(False, False, 128, bits=1), BinaryLinear)
        with self.assertRaises(ValueError):
            import_utils.dynamically_import_QuantLinear(True, False, 128, bits=1)


if __name__ == "__main__":
    unittest.main()
