from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from model import RedCharNet, build_model
from tta import tta_views


def load_models(checkpoints: Iterable[Path], device: torch.device) -> tuple[list[RedCharNet], list[dict]]:
    models: list[RedCharNet] = []
    payloads: list[dict] = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location=device)
        model_size = payload.get("config", {}).get("model_size", "base")
        model = build_model(model_size).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
        payloads.append(payload)
    if not models:
        raise ValueError("at least one checkpoint is required")
    return models, payloads


def normalize_weights(weights: Iterable[float] | None, model_count: int, label: str) -> list[float]:
    values = [1.0] * model_count if weights is None else [float(value) for value in weights]
    if len(values) != model_count:
        raise ValueError(f"{label} weights count must match checkpoints count")
    if any(value < 0 for value in values):
        raise ValueError(f"{label} weights must be non-negative")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"{label} weights must have a positive sum")
    return [value / total for value in values]


@torch.no_grad()
def average_model_logits(
    models: list[torch.nn.Module],
    images: torch.Tensor,
    char_weights: Iterable[float] | None = None,
    color_weights: Iterable[float] | None = None,
    tta: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not models:
        raise ValueError("at least one model is required")
    normalized_char = normalize_weights(char_weights, len(models), "char")
    normalized_color = normalize_weights(color_weights, len(models), "color")
    char_sum = None
    color_sum = None
    views = tta_views(images) if tta else (images,)
    for model, char_weight, color_weight in zip(models, normalized_char, normalized_color):
        model_char_sum = None
        model_color_sum = None
        for view in views:
            char_logits, color_logits = model(view)
            model_char_sum = char_logits if model_char_sum is None else model_char_sum + char_logits
            model_color_sum = color_logits if model_color_sum is None else model_color_sum + color_logits
        weighted_char = model_char_sum.mul(char_weight / len(views))
        weighted_color = model_color_sum.mul(color_weight / len(views))
        char_sum = weighted_char if char_sum is None else char_sum + weighted_char
        color_sum = weighted_color if color_sum is None else color_sum + weighted_color
    return char_sum, color_sum
