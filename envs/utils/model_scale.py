from collections.abc import Mapping
from typing import Any


def scaled_model_data(
    model_data: Mapping[str, Any], scale_multiplier: float
) -> dict[str, Any]:
    """Return actor metadata with its native scale multiplied uniformly."""

    result = dict(model_data)
    result["scale"] = [
        float(value) * float(scale_multiplier)
        for value in model_data["scale"]
    ]
    return result
