from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LayerScores:
    """
    One scalar sensitivity score for every transformer block.

    By default, a larger score means that the block is more sensitive
    to quantization and should therefore receive W3.
    """

    values: tuple[float, ...]
    score_name: str = "layer_sensitivity"
    higher_is_more_sensitive: bool = True
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("LayerScores cannot be empty.")

        for layer_idx, value in enumerate(self.values):
            if isinstance(value, bool):
                raise TypeError(
                    f"Layer {layer_idx} score must be numeric, "
                    "not bool."
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Layer {layer_idx} has a non-finite score: "
                    f"{value!r}"
                )


@dataclass(frozen=True)
class MixedPrecisionPlan:
    """
    Deterministic transformer-block-level W2/W3 allocation.
    """

    score_name: str
    score_source: str | None
    higher_is_more_sensitive: bool

    scores: tuple[float, ...]
    ranking: tuple[int, ...]

    w2_layers: tuple[int, ...]
    w3_layers: tuple[int, ...]
    layer_bits: tuple[int, ...]

    requested_average_bits: float | None
    actual_average_bits: float

    @property
    def num_layers(self) -> int:
        return len(self.layer_bits)

    @property
    def num_w2_layers(self) -> int:
        return len(self.w2_layers)

    @property
    def num_w3_layers(self) -> int:
        return len(self.w3_layers)

    @property
    def layer_bit_map(self) -> dict[int, int]:
        return {
            layer_idx: bits
            for layer_idx, bits in enumerate(self.layer_bits)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": (
                "dlm_score_guided_transformer_block_w2_w3_gptq"
            ),
            "num_layers": self.num_layers,
            "score": {
                "name": self.score_name,
                "source": self.score_source,
                "higher_is_more_sensitive": (
                    self.higher_is_more_sensitive
                ),
                "layer_scores": {
                    str(layer_idx): float(score)
                    for layer_idx, score in enumerate(self.scores)
                },
                "ranking_most_to_least_sensitive": list(
                    self.ranking
                ),
            },
            "budget": {
                "requested_average_bits": (
                    self.requested_average_bits
                ),
                "actual_average_bits": (
                    self.actual_average_bits
                ),
                "num_w2_layers": self.num_w2_layers,
                "num_w3_layers": self.num_w3_layers,
            },
            "w2_layers": list(self.w2_layers),
            "w3_layers": list(self.w3_layers),
            "layer_bit_map": {
                str(layer_idx): bits
                for layer_idx, bits in enumerate(self.layer_bits)
            },
        }


def _coerce_score_values(
    raw_scores: Sequence[Any] | Mapping[Any, Any],
    num_layers: int,
) -> tuple[float, ...]:
    if isinstance(raw_scores, Mapping):
        converted: dict[int, float] = {}

        for raw_layer_idx, raw_score in raw_scores.items():
            try:
                layer_idx = int(raw_layer_idx)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Layer-score mapping keys must be integer "
                    f"indices; got {raw_layer_idx!r}."
                ) from exc

            if layer_idx in converted:
                raise ValueError(
                    f"Duplicate layer index after conversion: "
                    f"{layer_idx}"
                )

            if isinstance(raw_score, bool):
                raise TypeError(
                    f"Layer {layer_idx} score cannot be bool."
                )

            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Layer {layer_idx} score is not numeric: "
                    f"{raw_score!r}"
                ) from exc

            converted[layer_idx] = score

        expected_indices = set(range(num_layers))
        actual_indices = set(converted)

        missing = sorted(expected_indices - actual_indices)
        unexpected = sorted(actual_indices - expected_indices)

        if missing or unexpected:
            raise ValueError(
                "Layer-score mapping does not match the model. "
                f"Missing={missing}, unexpected={unexpected}, "
                f"expected range=0-{num_layers - 1}."
            )

        values = tuple(
            converted[layer_idx]
            for layer_idx in range(num_layers)
        )

    elif isinstance(raw_scores, Sequence) and not isinstance(
        raw_scores,
        (str, bytes),
    ):
        if len(raw_scores) != num_layers:
            raise ValueError(
                f"Expected {num_layers} layer scores, "
                f"but received {len(raw_scores)}."
            )

        converted_values: list[float] = []

        for layer_idx, raw_score in enumerate(raw_scores):
            if isinstance(raw_score, bool):
                raise TypeError(
                    f"Layer {layer_idx} score cannot be bool."
                )

            try:
                converted_values.append(float(raw_score))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Layer {layer_idx} score is not numeric: "
                    f"{raw_score!r}"
                ) from exc

        values = tuple(converted_values)

    else:
        raise TypeError(
            "Layer scores must be either a JSON list or a mapping "
            "from layer index to score."
        )

    for layer_idx, score in enumerate(values):
        if not math.isfinite(score):
            raise ValueError(
                f"Layer {layer_idx} has a non-finite score: "
                f"{score!r}"
            )

    return values


def load_layer_scores(
    score_file: str | Path,
    num_layers: int,
) -> LayerScores:
    """
    Supported JSON formats:

    1. Bare list:
       [0.1, 0.2, ...]

    2. Metadata object:
       {
         "score_name": "temporal_outlier_density",
         "higher_is_more_sensitive": true,
         "layer_scores": {
           "0": 0.1,
           "1": 0.2
         }
       }
    """

    path = Path(score_file).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Layer-score file does not exist: {path}"
        )

    payload = json.loads(path.read_text())

    score_name = "layer_sensitivity"
    higher_is_more_sensitive = True

    if isinstance(payload, list):
        raw_scores = payload

    elif isinstance(payload, dict):
        if "layer_scores" in payload:
            raw_scores = payload["layer_scores"]
        elif "scores" in payload:
            raw_scores = payload["scores"]
        elif all(
            str(key).lstrip("-").isdigit()
            for key in payload
        ):
            # Also allow a bare {"0": score0, ...} mapping.
            raw_scores = payload
        else:
            raise ValueError(
                "Score JSON object must contain 'layer_scores' "
                "or 'scores'."
            )

        score_name = str(
            payload.get(
                "score_name",
                payload.get(
                    "method",
                    "layer_sensitivity",
                ),
            )
        )

        raw_direction = payload.get(
            "higher_is_more_sensitive",
            True,
        )

        if not isinstance(raw_direction, bool):
            raise TypeError(
                "'higher_is_more_sensitive' must be true or false."
            )

        higher_is_more_sensitive = raw_direction

    else:
        raise TypeError(
            "The score JSON root must be a list or an object."
        )

    values = _coerce_score_values(
        raw_scores=raw_scores,
        num_layers=num_layers,
    )

    return LayerScores(
        values=values,
        score_name=score_name,
        higher_is_more_sensitive=higher_is_more_sensitive,
        source=str(path),
    )


def _resolve_num_w3_layers(
    num_layers: int,
    num_w3_layers: int | None,
    target_average_bits: float | None,
) -> tuple[int, float | None]:
    if (num_w3_layers is None) == (
        target_average_bits is None
    ):
        raise ValueError(
            "Specify exactly one of num_w3_layers or "
            "target_average_bits."
        )

    if num_w3_layers is not None:
        if isinstance(num_w3_layers, bool):
            raise TypeError(
                "num_w3_layers must be an integer, not bool."
            )

        num_w3_layers = int(num_w3_layers)

        if not 0 <= num_w3_layers <= num_layers:
            raise ValueError(
                f"num_w3_layers must be in [0, {num_layers}], "
                f"but got {num_w3_layers}."
            )

        return num_w3_layers, None

    assert target_average_bits is not None

    target_average_bits = float(target_average_bits)

    if not 2.0 <= target_average_bits <= 3.0:
        raise ValueError(
            "For W2/W3 allocation, target_average_bits must be "
            f"in [2.0, 3.0], but got {target_average_bits}."
        )

    ideal_w3_count = (
        target_average_bits - 2.0
    ) * num_layers

    # Deterministic round-half-up. Python round() uses banker's
    # rounding, which is undesirable for a reproducible bit budget.
    resolved_w3_count = int(
        math.floor(ideal_w3_count + 0.5)
    )

    resolved_w3_count = min(
        max(resolved_w3_count, 0),
        num_layers,
    )

    return resolved_w3_count, target_average_bits


def allocate_w2_w3(
    layer_scores: LayerScores,
    *,
    num_w3_layers: int | None = None,
    target_average_bits: float | None = None,
) -> MixedPrecisionPlan:
    """
    Assign W3 to the most sensitive transformer blocks and W2 to
    all remaining blocks.

    Ties are broken by the lower layer index, ensuring deterministic
    and reproducible plans.
    """

    num_layers = len(layer_scores.values)

    resolved_num_w3, requested_average_bits = (
        _resolve_num_w3_layers(
            num_layers=num_layers,
            num_w3_layers=num_w3_layers,
            target_average_bits=target_average_bits,
        )
    )

    if layer_scores.higher_is_more_sensitive:
        ranking = tuple(
            sorted(
                range(num_layers),
                key=lambda layer_idx: (
                    -layer_scores.values[layer_idx],
                    layer_idx,
                ),
            )
        )
    else:
        ranking = tuple(
            sorted(
                range(num_layers),
                key=lambda layer_idx: (
                    layer_scores.values[layer_idx],
                    layer_idx,
                ),
            )
        )

    selected_w3 = set(
        ranking[:resolved_num_w3]
    )

    w3_layers = tuple(
        layer_idx
        for layer_idx in range(num_layers)
        if layer_idx in selected_w3
    )

    w2_layers = tuple(
        layer_idx
        for layer_idx in range(num_layers)
        if layer_idx not in selected_w3
    )

    layer_bits = tuple(
        3 if layer_idx in selected_w3 else 2
        for layer_idx in range(num_layers)
    )

    actual_average_bits = (
        sum(layer_bits) / num_layers
    )

    assert len(w2_layers) + len(w3_layers) == num_layers
    assert set(w2_layers).isdisjoint(w3_layers)
    assert all(bits in (2, 3) for bits in layer_bits)

    return MixedPrecisionPlan(
        score_name=layer_scores.score_name,
        score_source=layer_scores.source,
        higher_is_more_sensitive=(
            layer_scores.higher_is_more_sensitive
        ),
        scores=layer_scores.values,
        ranking=ranking,
        w2_layers=w2_layers,
        w3_layers=w3_layers,
        layer_bits=layer_bits,
        requested_average_bits=requested_average_bits,
        actual_average_bits=actual_average_bits,
    )


def save_mixed_precision_plan(
    plan: MixedPrecisionPlan,
    output_file: str | Path,
) -> Path:
    """
    Atomically save a reproducible allocation plan.
    """

    path = Path(output_file).expanduser().resolve()
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            plan.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    temporary_path.replace(path)

    return path
