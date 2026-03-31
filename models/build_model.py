from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .egovideo_adapter import build_egovideo_model, prepare_egovideo_inputs


@dataclass
class ModelBundle:
    model: nn.Module
    prepare_inputs: callable
    notes: list[str]


def build_model(config: dict, num_classes: int) -> ModelBundle:
    """Build the RGB action classifier used by the demo-ready pipeline."""

    model_name = config["model"]["name"].lower()
    if model_name == "egovideo_singleclip":
        model, meta = build_egovideo_model(config, num_classes=num_classes)
        return ModelBundle(model=model, prepare_inputs=prepare_egovideo_inputs, notes=meta["notes"])

    raise ValueError(f"Unsupported model name: {config['model']['name']}")
