from __future__ import annotations

import torch


TTA_OFFSETS = ((0, 0), (-2, 0), (2, 0), (0, -1), (0, 1))


def translate_images(images: torch.Tensor, dx: int, dy: int, fill: float = 1.0) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    height, width = images.shape[-2:]
    if abs(dx) >= width or abs(dy) >= height:
        raise ValueError("translation must be smaller than the image dimensions")

    translated = torch.full_like(images, fill)
    src_x_start = max(-dx, 0)
    src_x_end = min(width - dx, width)
    dst_x_start = max(dx, 0)
    dst_x_end = min(width + dx, width)
    src_y_start = max(-dy, 0)
    src_y_end = min(height - dy, height)
    dst_y_start = max(dy, 0)
    dst_y_end = min(height + dy, height)
    translated[..., dst_y_start:dst_y_end, dst_x_start:dst_x_end] = images[
        ..., src_y_start:src_y_end, src_x_start:src_x_end
    ]
    return translated


def tta_views(images: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(
        images if dx == 0 and dy == 0 else translate_images(images, dx=dx, dy=dy)
        for dx, dy in TTA_OFFSETS
    )
