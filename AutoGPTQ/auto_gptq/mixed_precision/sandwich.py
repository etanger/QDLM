from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SandwichConfig:
    """
    Configuration for DLM-specific sandwich mixed precision.

    Executable stages:
        w23: W2 edges + contiguous W3 middle core.
        w123: W1 outer edges + optional W2 transitions + W3 core.
    """

    config_path: str
    method: str
    stage: str

    stage1_target_average_bits: float
    final_target_average_bits: float

    num_w3_layers: int | None

    score_file: str | None
    score_name: str
    score_weight: float
    higher_is_more_sensitive: bool

    max_center_shift: float
    min_edge_layers: int

    plan_output: str


@dataclass(frozen=True)
class SandwichPlan:
    method: str
    stage: str
    num_layers: int

    layer_weights: tuple[int, ...]
    raw_scores: tuple[float, ...]
    normalized_scores: tuple[float, ...]
    centrality_scores: tuple[float, ...]
    combined_priorities: tuple[float, ...]

    score_name: str
    score_file: str | None
    score_weight: float

    w3_start: int
    w3_end: int

    current_bits: tuple[int, ...]
    final_preview_bits: tuple[int, ...]

    current_average_bits: float
    final_preview_average_bits: float

    stage1_target_average_bits: float
    final_target_average_bits: float

    @property
    def current_layer_bit_map(self) -> dict[int, int]:
        return {
            layer_idx: int(bits)
            for layer_idx, bits in enumerate(self.current_bits)
        }

    @property
    def current_w1_layers(self) -> tuple[int, ...]:
        return tuple(i for i, bits in enumerate(self.current_bits) if bits == 1)

    @property
    def current_w2_layers(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, bits in enumerate(self.current_bits)
            if bits == 2
        )

    @property
    def current_w3_layers(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, bits in enumerate(self.current_bits)
            if bits == 3
        )

    @property
    def final_w1_layers(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, bits in enumerate(self.final_preview_bits)
            if bits == 1
        )

    @property
    def final_w2_layers(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, bits in enumerate(self.final_preview_bits)
            if bits == 2
        )

    @property
    def final_w3_layers(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, bits in enumerate(self.final_preview_bits)
            if bits == 3
        )

    def to_dict(
        self,
        packed_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "method": self.method,
            "stage": self.stage,
            "num_layers": self.num_layers,
            "definition": {
                "structure": (
                    "aggressive_edges_conservative_middle"
                ),
                "current_stage": (
                    "W2 edges and transition layers; "
                    "contiguous W3 middle core"
                    if self.stage == "w23" else
                    "W1 outer edges; optional W2 transitions; contiguous W3 middle core"
                ),
                "final_target": (
                    "W1 outer edges; W2 transition layers; "
                    "W3 middle core"
                ),
            },
            "score": {
                "name": self.score_name,
                "file": self.score_file,
                "weight": self.score_weight,
                "raw": {
                    str(i): float(v)
                    for i, v in enumerate(self.raw_scores)
                },
                "normalized": {
                    str(i): float(v)
                    for i, v in enumerate(
                        self.normalized_scores
                    )
                },
                "centrality": {
                    str(i): float(v)
                    for i, v in enumerate(
                        self.centrality_scores
                    )
                },
                "combined_priority": {
                    str(i): float(v)
                    for i, v in enumerate(
                        self.combined_priorities
                    )
                },
            },
            "layer_weights": {
                str(i): int(v)
                for i, v in enumerate(self.layer_weights)
            },
            ("current_w23_stage" if self.stage == "w23" else "current_avg2_stage"): {
                "target_average_bits": (
                    self.stage1_target_average_bits if self.stage == "w23"
                    else self.final_target_average_bits
                ),
                "actual_average_bits": (
                    self.current_average_bits
                ),
                "w1_layers": list(self.current_w1_layers),
                "w2_layers": list(self.current_w2_layers),
                "w3_layers": list(self.current_w3_layers),
                "w3_interval": [
                    self.w3_start,
                    self.w3_end,
                ],
                "layer_bit_map": {
                    str(i): int(bits)
                    for i, bits in enumerate(self.current_bits)
                },
            },
            "final_avg2_preview": {
                "target_average_bits": (
                    self.final_target_average_bits
                ),
                "actual_average_bits": (
                    self.final_preview_average_bits
                ),
                "implemented": True,
                "executed": self.stage == "w123",
                "note": (
                    "W1 uses binary GPTQ and packed sign storage with a reference "
                    "dequantize-and-matmul backend. It is executed only in stage w123. "
                    "Average bits describe selected weights, excluding metadata and unquantized modules."
                ),
                "w1_layers": list(self.final_w1_layers),
                "w2_layers": list(self.final_w2_layers),
                "w3_layers": list(self.final_w3_layers),
                "layer_bit_map": {
                    str(i): int(bits)
                    for i, bits in enumerate(
                        self.final_preview_bits
                    )
                },
            },
        }

        if packed_summary is not None:
            result["packed_validation"] = dict(
                packed_summary
            )

        return result


def _resolve_path(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    return str(path.resolve())


def load_sandwich_config(
    config_file: str | Path,
) -> SandwichConfig:
    path = Path(config_file).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Sandwich config does not exist: {path}"
        )

    payload = json.loads(path.read_text())

    method = str(
        payload.get(
            "method",
            "dlm_sandwich_gptq",
        )
    )

    stage = str(
        payload.get(
            "stage",
            "w23",
        )
    )

    if stage not in {"w23", "w123"}:
        raise ValueError(
            f"Supported sandwich stages are 'w23' and 'w123'; got {stage!r}."
        )

    stage1_target = float(
        payload.get(
            "stage1_target_average_bits",
            2.25,
        )
    )

    if not 2.0 <= stage1_target <= 3.0:
        raise ValueError(
            "stage1_target_average_bits must be in [2, 3]."
        )

    final_target = float(
        payload.get(
            "final_target_average_bits",
            2.0,
        )
    )

    if not 1.0 <= final_target <= 3.0:
        raise ValueError(
            "final_target_average_bits must be in [1, 3]."
        )

    if stage == "w123" and final_target != 2.0:
        raise ValueError("Executable w123 requires final_target_average_bits=2.0.")

    raw_num_w3 = payload.get(
        "num_w3_layers",
        None,
    )

    num_w3_layers = (
        None
        if raw_num_w3 is None
        else int(raw_num_w3)
    )

    score_weight = float(
        payload.get(
            "score_weight",
            0.35,
        )
    )

    if not 0.0 <= score_weight <= 1.0:
        raise ValueError(
            "score_weight must be in [0, 1]."
        )

    min_edge_layers = int(
        payload.get(
            "min_edge_layers",
            4,
        )
    )

    if min_edge_layers < 1:
        raise ValueError(
            "min_edge_layers must be at least 1."
        )

    return SandwichConfig(
        config_path=str(path),
        method=method,
        stage=stage,
        stage1_target_average_bits=stage1_target,
        final_target_average_bits=final_target,
        num_w3_layers=num_w3_layers,
        score_file=_resolve_path(
            payload.get("score_file")
        ),
        score_name=str(
            payload.get(
                "score_name",
                "dlm_layer_sensitivity",
            )
        ),
        score_weight=score_weight,
        higher_is_more_sensitive=bool(
            payload.get(
                "higher_is_more_sensitive",
                True,
            )
        ),
        max_center_shift=float(
            payload.get(
                "max_center_shift",
                2.0,
            )
        ),
        min_edge_layers=min_edge_layers,
        plan_output=(
            _resolve_path(
                str(
                    payload.get(
                        "plan_output",
                        "results/mixed_precision/"
                        "dlm_sandwich_w23_plan.json",
                    )
                )
            )
            or ""
        ),
    )


def _coerce_scores(
    raw: Sequence[Any] | Mapping[Any, Any],
    num_layers: int,
) -> tuple[float, ...]:
    if isinstance(raw, Mapping):
        converted = {
            int(key): float(value)
            for key, value in raw.items()
        }

        expected = set(range(num_layers))
        actual = set(converted)

        if actual != expected:
            raise ValueError(
                "Score mapping must contain exactly layers "
                f"0-{num_layers - 1}; "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )

        values = tuple(
            converted[i]
            for i in range(num_layers)
        )

    elif isinstance(raw, Sequence) and not isinstance(
        raw,
        (str, bytes),
    ):
        if len(raw) != num_layers:
            raise ValueError(
                f"Expected {num_layers} scores, "
                f"received {len(raw)}."
            )

        values = tuple(float(value) for value in raw)

    else:
        raise TypeError(
            "Scores must be a JSON list or layer-index mapping."
        )

    for layer_idx, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(
                f"Layer {layer_idx} has invalid score {value}."
            )

    return values


def _load_scores(
    config: SandwichConfig,
    num_layers: int,
) -> tuple[float, ...]:
    if config.score_file is None:
        # Pure structural sandwich prior.
        return tuple(0.5 for _ in range(num_layers))

    path = Path(config.score_file)

    if not path.is_file():
        raise FileNotFoundError(
            f"Score file does not exist: {path}"
        )

    payload = json.loads(path.read_text())

    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        if "layer_scores" in payload:
            raw = payload["layer_scores"]
        elif "scores" in payload:
            raw = payload["scores"]
        elif all(
            str(key).lstrip("-").isdigit()
            for key in payload
        ):
            raw = payload
        else:
            raise ValueError(
                "Score file must contain 'layer_scores' or "
                "'scores'."
            )
    else:
        raise TypeError(
            "Score JSON root must be a list or object."
        )

    return _coerce_scores(
        raw=raw,
        num_layers=num_layers,
    )


def _normalize_scores(
    values: Sequence[float],
    higher_is_more_sensitive: bool,
) -> tuple[float, ...]:
    minimum = min(values)
    maximum = max(values)

    if math.isclose(minimum, maximum):
        normalized = [
            0.5
            for _ in values
        ]
    else:
        scale = maximum - minimum
        normalized = [
            (value - minimum) / scale
            for value in values
        ]

    if not higher_is_more_sensitive:
        normalized = [
            1.0 - value
            for value in normalized
        ]

    return tuple(normalized)


def _weighted_average_bits(
    bits: Sequence[int],
    weights: Sequence[int],
) -> float:
    denominator = sum(weights)

    if denominator <= 0:
        raise ValueError(
            "Layer weights must have a positive total."
        )

    numerator = sum(
        int(bit) * int(weight)
        for bit, weight in zip(bits, weights)
    )

    return numerator / denominator


def build_sandwich_plan(
    config: SandwichConfig,
    layer_weights: Sequence[int],
) -> SandwichPlan:
    weights = tuple(int(value) for value in layer_weights)

    if not weights:
        raise ValueError("layer_weights cannot be empty.")

    if any(value <= 0 for value in weights):
        raise ValueError(
            "Every transformer block must have positive "
            "quantized parameter count."
        )

    num_layers = len(weights)

    if (
        config.min_edge_layers * 2
        >= num_layers
    ):
        raise ValueError(
            "min_edge_layers leaves no room for a middle core."
        )

    raw_scores = _load_scores(
        config=config,
        num_layers=num_layers,
    )

    normalized_scores = _normalize_scores(
        values=raw_scores,
        higher_is_more_sensitive=(
            config.higher_is_more_sensitive
        ),
    )

    model_center = (num_layers - 1) / 2.0

    centrality = tuple(
        1.0
        - (
            abs(layer_idx - model_center)
            / max(model_center, 1.0)
        )
        for layer_idx in range(num_layers)
    )

    combined_priority = tuple(
        (
            (1.0 - config.score_weight)
            * centrality[layer_idx]
        )
        + (
            config.score_weight
            * normalized_scores[layer_idx]
        )
        for layer_idx in range(num_layers)
    )

    if config.num_w3_layers is not None:
        candidate_lengths = [
            config.num_w3_layers
        ]
    else:
        candidate_lengths = list(
            range(
                1,
                num_layers
                - 2 * config.min_edge_layers
                + 1,
            )
        )

    total_weight = sum(weights)
    candidates: list[
        tuple[
            tuple[float, float, float, int, int],
            int,
            int,
            float,
        ]
    ] = []

    for core_length in candidate_lengths:
        if core_length <= 0:
            continue

        if (
            core_length
            > num_layers
            - 2 * config.min_edge_layers
        ):
            continue

        first_start = config.min_edge_layers
        last_start = (
            num_layers
            - config.min_edge_layers
            - core_length
        )

        for start in range(
            first_start,
            last_start + 1,
        ):
            end = start + core_length - 1
            interval_center = (start + end) / 2.0
            center_shift = abs(
                interval_center - model_center
            )

            if (
                center_shift
                > config.max_center_shift
            ):
                continue

            w3_weight = sum(
                weights[start : end + 1]
            )

            average_bits = (
                2.0
                + w3_weight / total_weight
            )

            budget_error = abs(
                average_bits
                - config.stage1_target_average_bits
            )

            utility = sum(
                combined_priority[layer_idx]
                * weights[layer_idx]
                for layer_idx in range(
                    start,
                    end + 1,
                )
            ) / w3_weight

            # Prefer:
            # 1. exact average-bit budget;
            # 2. high DLM middle/sensitivity priority;
            # 3. center alignment;
            # 4. deterministic smaller start.
            key = (
                budget_error,
                -utility,
                center_shift,
                start,
                core_length,
            )

            candidates.append(
                (
                    key,
                    start,
                    end,
                    average_bits,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No valid W3 middle interval satisfies the "
            "sandwich constraints."
        )

    candidates.sort(key=lambda item: item[0])

    _, w3_start, w3_end, _ = candidates[0]

    current_bits = tuple(
        3
        if w3_start <= layer_idx <= w3_end
        else 2
        for layer_idx in range(num_layers)
    )

    current_average_bits = _weighted_average_bits(
        bits=current_bits,
        weights=weights,
    )

    # --------------------------------------------------------
    # Final exact-average-2 preview:
    #
    # Every W3 parameter contributes +1 bit over the W2 base.
    # An equal parameter mass at W1 contributes -1 bit.
    # Therefore W1 parameter mass == W3 parameter mass gives
    # an exact average of 2 bits.
    # --------------------------------------------------------

    edge_order: list[int] = []

    for depth in range(
        (num_layers + 1) // 2
    ):
        left = depth
        right = num_layers - 1 - depth

        if current_bits[left] == 2:
            edge_order.append(left)

        if (
            right != left
            and current_bits[right] == 2
        ):
            edge_order.append(right)

    final_candidates: list[
        tuple[
            tuple[float, int, float, int],
            tuple[int, ...],
            tuple[int, ...],
            float,
        ]
    ] = []

    for count in range(len(edge_order) + 1):
        selected_w1 = tuple(
            edge_order[:count]
        )

        final_bits_list = list(current_bits)

        for layer_idx in selected_w1:
            final_bits_list[layer_idx] = 1

        final_bits = tuple(final_bits_list)

        final_average = _weighted_average_bits(
            bits=final_bits,
            weights=weights,
        )

        left_count = sum(
            layer_idx < model_center
            for layer_idx in selected_w1
        )
        right_count = sum(
            layer_idx > model_center
            for layer_idx in selected_w1
        )

        asymmetry = abs(
            left_count - right_count
        )

        selected_priority = sum(
            combined_priority[layer_idx]
            for layer_idx in selected_w1
        )

        key = (
            abs(
                final_average
                - config.final_target_average_bits
            ),
            asymmetry,
            selected_priority,
            count,
        )

        final_candidates.append(
            (
                key,
                selected_w1,
                final_bits,
                final_average,
            )
        )

    final_candidates.sort(
        key=lambda item: item[0]
    )

    (
        _,
        _,
        final_preview_bits,
        final_preview_average_bits,
    ) = final_candidates[0]

    if config.stage == "w123":
        # Do not silently call a nearest-budget allocation exact avg2.
        if sum(b * n for b, n in zip(final_preview_bits, weights)) != 2 * sum(weights):
            raise ValueError(
                "The selected middle interval cannot reach exact avg2 with whole-layer "
                "edge allocation. Change num_w3_layers or the quantized module coverage."
            )
        current_bits = final_preview_bits
        current_average_bits = final_preview_average_bits

    return SandwichPlan(
        method=config.method,
        stage=config.stage,
        num_layers=num_layers,
        layer_weights=weights,
        raw_scores=raw_scores,
        normalized_scores=normalized_scores,
        centrality_scores=centrality,
        combined_priorities=combined_priority,
        score_name=config.score_name,
        score_file=config.score_file,
        score_weight=config.score_weight,
        w3_start=w3_start,
        w3_end=w3_end,
        current_bits=current_bits,
        final_preview_bits=final_preview_bits,
        current_average_bits=current_average_bits,
        final_preview_average_bits=(
            final_preview_average_bits
        ),
        stage1_target_average_bits=(
            config.stage1_target_average_bits
        ),
        final_target_average_bits=(
            config.final_target_average_bits
        ),
    )


def save_sandwich_plan(
    plan: SandwichPlan,
    output_file: str | Path,
    packed_summary: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(output_file).expanduser().resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            plan.to_dict(
                packed_summary=packed_summary,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    temporary.replace(path)

    return path


def validate_packed_model(
    model,
    quantizers: Mapping[str, Any],
    plan: SandwichPlan,
    layers_block_name: str,
) -> dict[str, Any]:
    """
    Verify every packed GPTQ module against the planned block bit.

    This catches cases where GPTQ used W2/W3 correctly but pack_model
    accidentally recreated every QuantLinear with one global bit.
    """

    named_modules = dict(
        model.named_modules()
    )

    prefix = layers_block_name + "."

    histogram: dict[int, int] = {}
    expected_histogram: dict[int, int] = {}
    checked_modules = 0
    mismatches: list[dict[str, Any]] = []

    for module_name in quantizers:
        if not module_name.startswith(prefix):
            continue

        suffix = module_name[len(prefix) :]

        try:
            layer_token = suffix.split(
                ".",
                1,
            )[0]

            layer_idx = int(layer_token)
        except (ValueError, IndexError) as exc:
            raise RuntimeError(
                "Cannot recover transformer block index from "
                f"quantized module name {module_name!r}."
            ) from exc

        expected_bits = int(
            plan.current_bits[layer_idx]
        )

        expected_histogram[expected_bits] = (
            expected_histogram.get(
                expected_bits,
                0,
            )
            + 1
        )

        module = named_modules.get(module_name)

        if module is None:
            mismatches.append(
                {
                    "name": module_name,
                    "reason": "missing_after_pack",
                    "expected_bits": expected_bits,
                }
            )
            continue

        actual_bits = getattr(
            module,
            "bits",
            None,
        )

        if actual_bits is None:
            mismatches.append(
                {
                    "name": module_name,
                    "reason": "module_has_no_bits_attribute",
                    "expected_bits": expected_bits,
                    "actual_type": type(module).__name__,
                }
            )
            continue

        actual_bits = int(actual_bits)

        histogram[actual_bits] = (
            histogram.get(
                actual_bits,
                0,
            )
            + 1
        )

        checked_modules += 1

        if actual_bits != expected_bits:
            mismatches.append(
                {
                    "name": module_name,
                    "layer": layer_idx,
                    "expected_bits": expected_bits,
                    "actual_bits": actual_bits,
                }
            )

    if mismatches:
        raise RuntimeError(
            "Packed mixed-precision validation failed. "
            f"First mismatches: {mismatches[:10]}"
        )

    if histogram != expected_histogram:
        raise RuntimeError(
            "Packed bit histogram does not match the plan: "
            f"expected={expected_histogram}, "
            f"actual={histogram}"
        )

    packed_modules = [named_modules[n] for n in quantizers if n.startswith(prefix)]
    parameter_count = sum(m.infeatures * m.outfeatures for m in packed_modules)
    weighted_bits = sum(m.bits * m.infeatures * m.outfeatures for m in packed_modules)
    if not packed_modules or parameter_count != sum(plan.layer_weights):
        raise RuntimeError("Packed module coverage does not match the planned parameter count.")
    expected_bits = sum(b * n for b, n in zip(plan.current_bits, plan.layer_weights))
    if weighted_bits != expected_bits:
        raise RuntimeError("Packed weight bit budget does not match the plan.")
    packed_weight_bytes = sum(m.qweight.numel() * m.qweight.element_size() for m in packed_modules)
    packed_module_bytes = sum(t.numel() * t.element_size()
                              for m in packed_modules for t in m.state_dict().values())
    state_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    # Packed linear biases are buffers but were parameters in the source model.
    original_parameters = parameter_count + sum(p.numel() for p in model.parameters())
    original_parameters += sum(m.bias.numel() for m in packed_modules if m.bias is not None)

    return {
        "quantized_weight_parameters": parameter_count,
        "quantized_weight_average_bits": weighted_bits / parameter_count,
        "quantized_parameter_fraction": parameter_count / original_parameters,
        "packed_weight_bytes_including_padding": packed_weight_bytes,
        "quantization_metadata_and_bias_bytes": packed_module_bytes - packed_weight_bytes,
        "unquantized_state_bytes": state_bytes - packed_module_bytes,
        "state_dict_tensor_bytes": state_bytes,
        "state_dict_bits_per_original_parameter": 8 * state_bytes / original_parameters,
        "runtime_memory_note": "Forward dequantization temporaries are not included in stored tensor bytes.",
        "checked_modules": checked_modules,
        "expected_histogram": {
            str(key): value
            for key, value in sorted(
                expected_histogram.items()
            )
        },
        "actual_histogram": {
            str(key): value
            for key, value in sorted(
                histogram.items()
            )
        },
        "status": "pass",
    }
