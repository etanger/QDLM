"""Convert existing symmetric GPTQ W2/W3 codes to FLUTE without requantizing.

FLUTE is imported lazily so CPU packing checks need only PyTorch. Kernel
tuning and repacking must finish before timed generation starts.
"""

from collections import Counter
import math

import torch
from torch import nn

FLUTE_REVISION = "9eb83a12d56949bbe7fe9c836ba97a67bd1e3761"


def require_flute():
    try:
        import flute
        import flute.tune
        import flute.utils
    except (ImportError, OSError) as exc:
        raise RuntimeError("Build FLUTE first with scripts/setup_qdlm_flute.sh") from exc
    if flute.TEMPLATE_CONFIGS is None:
        raise RuntimeError("FLUTE template configuration is missing")
    for bits in (2, 3):
        if not flute.utils.get_template_ids(bits):
            raise RuntimeError(f"FLUTE build does not include W{bits} templates")
    return flute


def unpack_gptq_axis0(packed, bits, length):
    """Decode AutoGPTQ's 32-code blocks, including crossing W3 bit fields."""
    if bits not in (2, 3) or length % 32:
        raise ValueError("Expected W2/W3 and a dimension divisible by 32")
    if packed.ndim != 2 or packed.dtype != torch.int32:
        raise ValueError("Expected a 2-D int32 packed tensor")
    if packed.shape[0] != length // 32 * bits:
        raise ValueError("Packed dimension does not match the original dimension")
    blocks = packed.reshape(length // 32, bits, packed.shape[1])
    result = torch.empty((length // 32, 32, packed.shape[1]),
                         device=packed.device, dtype=torch.uint8)
    for index in range(32):
        word, shift = divmod(index * bits, 32)
        value = (blocks[:, word].to(torch.int64) & 0xFFFFFFFF) >> shift
        if shift + bits > 32:
            value |= (blocks[:, word + 1].to(torch.int64) & 0xFFFFFFFF) << (32 - shift)
        result[:, index] = (value & ((1 << bits) - 1)).to(torch.uint8)
    return result.reshape(length, packed.shape[1])


def inspect_source(module):
    """Reject unsupported layouts instead of silently changing quantization."""
    if module.bits not in (2, 3):
        raise ValueError("Only W2/W3 are supported by this adapter")
    k, n, group = module.infeatures, module.outfeatures, module.group_size
    if k % 32 or n % 32 or group not in (32, 64, 128, 256) or k % group:
        raise ValueError("Unsupported matrix dimensions or group size")
    if module.scales.dtype != torch.float16 or module.scales.shape != (k // group, n):
        raise ValueError("Expected FP16 group scales with shape [K/group_size, N]")
    expected_groups = torch.arange(k, device=module.g_idx.device) // group
    if not torch.equal(module.g_idx.long(), expected_groups):
        raise ValueError("Activation ordering or noncontiguous groups are unsupported")
    zeros = unpack_gptq_axis0(module.qzeros.T.contiguous(), module.bits, n).T.long() + 1
    # The legacy W2 decoder wraps after adding one; W3 does not.
    if module.bits == 2:
        zeros &= 3
    zero = 1 << (module.bits - 1)
    if not bool(torch.all(zeros == zero)):
        raise ValueError("FLUTE LUT conversion requires the symmetric GPTQ zero point")
    if not bool(torch.isfinite(module.scales).all()) or not bool((module.scales > 0).all()):
        raise ValueError("Nonfinite or nonpositive scales")
    return zero


def reference_weight(module):
    """Reconstruct [K, N] for CPU verification only; never used in forward."""
    zero = inspect_source(module)
    codes = unpack_gptq_axis0(module.qweight, module.bits, module.infeatures)
    scales = module.scales.repeat_interleave(module.group_size, dim=0)
    return (codes.to(scales.dtype) - zero) * scales


def tuning_rows(m):
    """Tune nearby smaller shapes; actual model inputs are never padded/shortened."""
    if m <= 0:
        raise ValueError("M must be positive")
    return max(16, 1 << (m.bit_length() - 1))


class FluteQuantLinear(nn.Module):
    QUANT_TYPE = "flute"

    def __init__(self, source):
        super().__init__()
        zero = inspect_source(source)
        self.bits = source.bits
        self.infeatures = source.infeatures
        self.outfeatures = source.outfeatures
        self.group_size = source.group_size
        device = source.scales.device
        # Keep only compressed source codes on CPU to repack for another tile.
        # This is deliberately not a registered buffer: model.to(cuda) must
        # not transfer a second unused copy of the weights to the GPU.
        self._source_packed_cpu = source.qweight.detach().cpu().clone()
        self.register_buffer("scales", source.scales.T.contiguous().detach().clone())
        self.register_buffer("qweight", torch.empty(0, device=device, dtype=torch.int16))
        self.register_buffer("qmap", torch.arange(1 << self.bits, device=device,
                                                  dtype=torch.float16) - zero)
        self.register_buffer("qmap2", torch.empty(0, device=device, dtype=torch.float32))
        self.register_buffer("bias", None if source.bias is None else source.bias.detach().clone())
        self._tile_p = None
        self._ready = None
        self.template_id = None
        self.num_sms = None
        self.prepared_shapes = []

    @torch.no_grad()
    def prepare(self, m):
        device = self.scales.device
        if device.type != "cuda":
            raise RuntimeError("FLUTE preparation requires a CUDA device")
        if self._ready == (str(device), m):
            return
        flute = require_flute()
        num_sms = flute.utils.get_device_num_sms(device)
        bucket = tuning_rows(m)
        # _tune is pinned to FLUTE_REVISION. Its cache is per shape/SM count.
        template_id = flute.tune._tune(
            M=bucket, N=self.outfeatures, K=self.infeatures,
            num_bits=self.bits, group_size=self.group_size, num_sms=num_sms,
            dtype=self.scales.dtype, device=device, num_seeds=1, legacy=False)
        if not flute.utils.is_template_supported(
                M=m, N=self.outfeatures, K=self.infeatures, num_bits=self.bits,
                template_id=template_id, num_sms=num_sms):
            raise RuntimeError("Tuned FLUTE template cannot execute the actual input shape")
        tile_p = flute.utils.get_template_config(self.bits, template_id, num_sms)["tileP"]
        if self._tile_p != tile_p or not self.qweight.numel():
            codes = unpack_gptq_axis0(self._source_packed_cpu, self.bits, self.infeatures)
            self.qweight = flute.utils.pack(
                W=codes, num_bits=self.bits, template_ids=[template_id],
                num_sms=num_sms).to(device)
            self._tile_p = tile_p
        if not self.qmap2.numel():
            self.qmap2 = flute.utils.make_qmap2_from_qmap(self.qmap)
        flute.utils.get_workspace_streamk(device)
        self.template_id, self.num_sms = template_id, num_sms
        self._ready = (str(device), m)
        self.prepared_shapes.append(dict(device=str(device), actual_m=m, tuning_m=bucket,
                                         template_id=template_id, tile_p=tile_p, num_sms=num_sms))

    def forward(self, x):
        if x.dtype != torch.float16 or x.device != self.scales.device:
            raise RuntimeError("FLUTE expects FP16 activations on the weight device")
        if x.shape[-1] != self.infeatures:
            raise ValueError("Unexpected input dimension")
        m = x.numel() // self.infeatures
        if self._ready != (str(x.device), m):
            raise RuntimeError("Call prepare_flute_model(model, M) before generation/timing")
        import flute
        import flute.utils
        output = flute.qgemm(
            x.reshape(m, self.infeatures).contiguous(), self.qweight, self.scales,
            self.qmap, self.qmap2, flute.utils.get_workspace_streamk(x.device),
            self.bits, self.group_size, self.template_id, self.num_sms)
        output = output.reshape(x.shape[:-1] + (self.outfeatures,))
        return output if self.bias is None else output + self.bias


@torch.no_grad()
def prepare_flute_model(model, m):
    for module in model.modules():
        if isinstance(module, FluteQuantLinear):
            module.prepare(m)


@torch.no_grad()
def convert_model_to_flute(model):
    """Validate every replaced linear against its existing GPU forward."""
    require_flute()
    reports = []
    originals = [(name, module) for name, module in model.named_modules()
                 if hasattr(module, "qweight")]
    if not originals or {module.bits for _, module in originals} != {2, 3}:
        raise RuntimeError("Expected existing packed W2 and W3 modules")
    for name, source in originals:
        if not type(source).__module__.endswith("qlinear_cuda_old"):
            raise RuntimeError(f"Unsupported source backend for {name}: {type(source)}")
        replacement = FluteQuantLinear(source).eval()
        device = source.scales.device
        with torch.cuda.device(device):
            replacement.prepare(32)
            rng = torch.Generator(device=device).manual_seed(123)
            x = torch.randn((32, source.infeatures), dtype=torch.float16,
                            device=device, generator=rng)
            threshold = source.kernel_switch_threshold
            try:
                source.kernel_switch_threshold = 0
                expected = source(x)
            finally:
                source.kernel_switch_threshold = threshold
            actual = replacement(x)
            delta = (actual.float() - expected.float())
            relative_l2 = (delta.norm() / expected.float().norm().clamp_min(1e-12)).item()
            relative_max = (delta.abs().max() / expected.float().abs().max().clamp_min(1e-12)).item()
            passed = math.isfinite(relative_l2) and math.isfinite(relative_max)
            passed = passed and relative_l2 <= 0.003 and relative_max <= 0.02
            record = dict(name=name, bits=source.bits, relative_l2=relative_l2,
                          relative_max=relative_max, passed=passed)
            if not passed:
                raise RuntimeError(f"FLUTE numerical validation failed: {record}")
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, replacement)
        source.to("cpu")
        reports.append(record)
        print(f"FLUTE conversion {len(reports)}/{len(originals)}: {name} W{source.bits} "
              f"relative_l2={relative_l2:.3g}", flush=True)
    return dict(backend="flute", source_revision=FLUTE_REVISION,
                conversion="Same GPTQ codes/scales; symmetric integer LUT; no requantization",
                histogram=dict(Counter(str(r["bits"]) for r in reports)),
                relative_l2_limit=0.003, relative_max_limit=0.02, modules=reports)


@torch.no_grad()
def preflight_flute():
    """Fail on unsupported hardware/builds before loading/calibrating LLaDA."""
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
    records = []
    for index in range(2):
        device = torch.device(f"cuda:{index}")
        with torch.cuda.device(device):
            modules = nn.ModuleList()
            for bits in (2, 3):
                source = QuantLinear(bits, 128, 4096, 4096, False,
                                     kernel_switch_threshold=0).to(device)
                rng = torch.Generator(device=device).manual_seed(321 + bits)
                source.qweight.copy_(torch.randint(
                    -(1 << 31), (1 << 31) - 1, source.qweight.shape,
                    device=device, dtype=torch.int32, generator=rng))
                source.scales.fill_(0.01)
                words = [0] * bits
                encoded_zero = (1 << (bits - 1)) - 1
                for code_index in range(32):
                    word, shift = divmod(code_index * bits, 32)
                    words[word] |= (encoded_zero << shift) & 0xFFFFFFFF
                    if shift + bits > 32:
                        words[word + 1] |= encoded_zero >> (32 - shift)
                words = [w if w < (1 << 31) else w - (1 << 32) for w in words]
                source.qzeros.copy_(torch.tensor(words, device=device, dtype=torch.int32)
                                    .repeat(source.outfeatures // 32).expand_as(source.qzeros))
                modules.append(source)
            print(f"FLUTE GPU preflight on physical GPU {index + 2}", flush=True)
            report = convert_model_to_flute(modules)
            records.append(dict(device=str(device), gpu_name=torch.cuda.get_device_name(index),
                                validation=report))
            del source, modules
            torch.cuda.empty_cache()
    return records
