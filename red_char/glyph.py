from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

import config
from dataset import RedCharDataset, TrainAugmentation
from model import ResidualSEStage


GLYPH_CROP_WIDTH = 64


def _slot_center(position: int) -> int:
    if position < 0 or position >= config.NUM_POSITIONS:
        raise ValueError(f"position must be in [0, {config.NUM_POSITIONS}), got {position}")
    return (position * 2 + 1) * config.IMAGE_WIDTH // (2 * config.NUM_POSITIONS)


def extract_glyph_crop(image: torch.Tensor, position: int, crop_width: int = GLYPH_CROP_WIDTH) -> torch.Tensor:
    if image.ndim != 3 or image.shape[-2:] != (config.IMAGE_HEIGHT, config.IMAGE_WIDTH):
        raise ValueError(f"expected [3,{config.IMAGE_HEIGHT},{config.IMAGE_WIDTH}], got {tuple(image.shape)}")
    half = crop_width // 2
    padded = F.pad(image, (half, half, 0, 0), value=1.0)
    center = _slot_center(position)
    return padded[..., center : center + crop_width]


def extract_glyph_crops(images: torch.Tensor, crop_width: int = GLYPH_CROP_WIDTH) -> torch.Tensor:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[-2:] != (config.IMAGE_HEIGHT, config.IMAGE_WIDTH):
        raise ValueError(f"expected [B,3,{config.IMAGE_HEIGHT},{config.IMAGE_WIDTH}], got {tuple(images.shape)}")
    half = crop_width // 2
    padded = F.pad(images, (half, half, 0, 0), value=1.0)
    crops = []
    for position in range(config.NUM_POSITIONS):
        center = _slot_center(position)
        crops.append(padded[..., center : center + crop_width])
    return torch.stack(crops, dim=1)


class GlyphDataset(Dataset):
    def __init__(
        self,
        base: RedCharDataset,
        image_indices: list[int],
        red_only: bool = True,
        augment: bool = False,
        red_line_p: float = 0.0,
        faint_p: float = 0.0,
        cutout_p: float = 0.0,
        crop_width: int = GLYPH_CROP_WIDTH,
    ) -> None:
        self.base = base
        self.crop_width = crop_width
        self.items: list[tuple[int, int]] = []
        for image_idx in image_indices:
            sample = base.samples[image_idx]
            if sample.color is None:
                raise ValueError("glyph dataset requires labelled samples")
            for position, color in enumerate(sample.color):
                if not red_only or color == "r":
                    self.items.append((image_idx, position))
        if augment:
            augment_values = dict(config.AUGMENT_PRESETS["medium"])
            augment_values["red_line_p"] = red_line_p
            augment_values["faint_p"] = faint_p
            augment_values["cutout_p"] = cutout_p
            self.transform = TrainAugmentation(**augment_values)
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_idx, position = self.items[index]
        image, char_target, _ = self.base[image_idx]
        crop = extract_glyph_crop(image, position, crop_width=self.crop_width)
        if self.transform is not None:
            crop = self.transform(crop)
        return crop, char_target[position]


class GlyphNet(nn.Module):
    def __init__(
        self,
        dropout: float = 0.2,
        input_mode: str = "rgb",
        hires: bool = False,
        crop_width: int = GLYPH_CROP_WIDTH,
        head_mode: str = "flat",
    ) -> None:
        super().__init__()
        if input_mode not in {"rgb", "red", "red2"}:
            raise ValueError(f"unknown glyph input mode: {input_mode}")
        if head_mode not in {"flat", "gap"}:
            raise ValueError(f"unknown glyph head mode: {head_mode}")
        self.input_mode = input_mode
        self.hires = hires
        self.crop_width = crop_width
        self.head_mode = head_mode
        in_channels = 5 if input_mode in {"red", "red2"} else 3
        widths = (48, 96, 192, 256)
        pools = (True, False, False, False) if hires else (True, True, False, False)
        stages = []
        stage_channels = in_channels
        for out_channels, pool in zip(widths, pools):
            stages.append(ResidualSEStage(stage_channels, out_channels, pool=pool))
            stage_channels = out_channels
        self.backbone = nn.Sequential(*stages)
        if head_mode == "gap":
            self.reduce = nn.Identity()
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(widths[-1], 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, config.NUM_CHARS),
            )
        else:
            self.reduce = nn.Sequential(
                nn.Conv2d(widths[-1], 32, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, config.IMAGE_HEIGHT, crop_width)
                feat_dim = self.reduce(self.backbone(dummy)).flatten(1).shape[1]
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, config.NUM_CHARS),
            )

    def _expand_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode not in {"red", "red2"}:
            return x
        red = x[:, 0:1]
        redness = (red - x[:, 1:3].amax(dim=1, keepdim=True)).relu()
        if self.input_mode == "red2":
            redness = redness / (redness.amax(dim=(2, 3), keepdim=True) + 1e-4)
        darkness = 1.0 - x.mean(dim=1, keepdim=True)
        return torch.cat([x, redness, darkness], dim=1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.head[:-1](self.reduce(self.backbone(self._expand_input(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head[-1](self.features(x))


def load_glyph_model(checkpoint: Path, device: torch.device, use_ema: bool = True) -> GlyphNet:
    payload = torch.load(checkpoint, map_location=device)
    model = GlyphNet(
        input_mode=payload.get("input_mode", "rgb"),
        hires=payload.get("hires", False),
        crop_width=payload.get("crop_width", GLYPH_CROP_WIDTH),
        head_mode=payload.get("head_mode", "flat"),
    ).to(device)
    state = payload.get("ema_state_dict") if use_ema else None
    model.load_state_dict(state if state is not None else payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def glyph_probabilities(models: list[GlyphNet], images: torch.Tensor) -> torch.Tensor:
    if not models:
        raise ValueError("at least one glyph model is required")
    crop_width = getattr(models[0], "crop_width", GLYPH_CROP_WIDTH)
    batch_size = images.shape[0]
    crops = extract_glyph_crops(images, crop_width=crop_width).flatten(0, 1)
    probabilities = None
    for model in models:
        current = model(crops).softmax(dim=-1)
        probabilities = current if probabilities is None else probabilities + current
    return (probabilities / len(models)).view(batch_size, config.NUM_POSITIONS, config.NUM_CHARS)
