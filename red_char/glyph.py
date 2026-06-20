from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

import config
from augment import TrainAugment
from dataset import RedCharDataset
from model import ResidualSEStage


GLYPH_CROP_WIDTH = 64
PAIR_GROUPS = ("1ILTVP", "2Z7T", "3S568E", "CGOQ", "7NY", "PF")


def extract_glyph_crops(images: torch.Tensor, crop_width: int = GLYPH_CROP_WIDTH) -> torch.Tensor:
    """Extract overlapping crops centred on the five nominal character slots.

    Padding keeps edge slots the same size as middle slots. The crop is wider
    than one 40-pixel slot so position jitter does not clip the target glyph.
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[-2:] != (config.IMAGE_HEIGHT, config.IMAGE_WIDTH):
        raise ValueError(f"expected [B,3,60,200], got {tuple(images.shape)}")
    half = crop_width // 2
    padded = F.pad(images, (half, half, 0, 0), value=1.0)
    crops = []
    for position in range(config.NUM_POSITIONS):
        center = (position * 2 + 1) * config.IMAGE_WIDTH // (2 * config.NUM_POSITIONS)
        crops.append(padded[..., center : center + crop_width])
    return torch.stack(crops, dim=1)


class GlyphDataset(Dataset):
    """Position-level view over full images, optionally restricted to red glyphs."""

    def __init__(
        self,
        base: RedCharDataset,
        image_indices: list[int],
        red_only: bool = True,
        augment: bool = False,
        red_line_p: float = 0.0,
        cutout_p: float = 0.0,
        faint_p: float = 0.0,
        crop_width: int = GLYPH_CROP_WIDTH,
        boost_chars: str = "",
        boost_factor: int = 1,
    ) -> None:
        self.base = base
        self.crop_width = crop_width
        self.items: list[tuple[int, int]] = []
        boost_set = set(boost_chars)
        for image_idx in image_indices:
            sample = base.samples[image_idx]
            if sample.color is None:
                raise ValueError("glyph dataset requires labelled samples")
            for position, color in enumerate(sample.color):
                if not red_only or color == "r":
                    # oversample hard confusion-group chars to focus capacity
                    reps = boost_factor if sample.all_label[position] in boost_set else 1
                    for _ in range(reps):
                        self.items.append((image_idx, position))
        self.transform = (
            TrainAugment(translate=0.08, scale=(0.94, 1.06), degrees=5.0, noise_std=0.015,
                         red_line_p=red_line_p, cutout_p=cutout_p, faint_p=faint_p)
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_idx, position = self.items[index]
        image, char_target, _ = self.base[image_idx]
        crop = extract_glyph_crops(image, crop_width=self.crop_width)[0, position]
        if self.transform is not None:
            crop = self.transform(crop)
        return crop, char_target[position]


class GlyphNet(nn.Module):
    """Shared high-resolution classifier for one approximately centred glyph.

    ``hires=True`` keeps one more pooling stage's worth of resolution (30x32
    feature map instead of 15x16) so thin-stroke confusions (I/1/L/T/J, E/F)
    keep their distinguishing serif/length detail. Feature dim is inferred so
    crop width / pooling can change freely.
    """

    def __init__(self, dropout: float = 0.2, input_mode: str = "rgb",
                 hires: bool = False, crop_width: int = GLYPH_CROP_WIDTH,
                 head_mode: str = "flat") -> None:
        super().__init__()
        if input_mode not in {"rgb", "red", "red2"}:
            raise ValueError(f"unknown glyph input mode: {input_mode}")
        if head_mode not in {"flat", "gap"}:
            raise ValueError(f"unknown head_mode: {head_mode}")
        self.input_mode = input_mode
        self.hires = hires
        self.crop_width = crop_width
        self.head_mode = head_mode
        widths = (48, 96, 192, 256)
        pools = (True, False, False, False) if hires else (True, True, False, False)
        stages = []
        base_in = 5 if input_mode in {"red", "red2"} else 3
        in_channels = base_in
        for out_channels, pool in zip(widths, pools):
            stages.append(ResidualSEStage(in_channels, out_channels, pool=pool))
            in_channels = out_channels
        self.backbone = nn.Sequential(*stages)
        if head_mode == "gap":
            # Global average pooling keeps the hi-res conv detail but a tiny head
            # (no 30k-wide FC), so deeper/finer features don't overfit.
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
                dummy = torch.zeros(1, base_in, config.IMAGE_HEIGHT, crop_width)
                feat_dim = self.reduce(self.backbone(dummy)).flatten(1).shape[1]
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, config.NUM_CHARS),
            )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode in {"red", "red2"}:
            red = x[:, 0:1]
            redness = (red - x[:, 1:3].amax(dim=1, keepdim=True)).relu()
            if self.input_mode == "red2":
                # intensity-robust: normalise redness per-crop by its own max so
                # FAINT red strokes (e.g. the light right stroke of a V) stay
                # visible relative to the strongest red, instead of being
                # suppressed by the absolute-difference redness (fixes V->I).
                m = redness.amax(dim=(2, 3), keepdim=True)
                redness = redness / (m + 1e-4)
            darkness = 1.0 - x.mean(dim=1, keepdim=True)
            x = torch.cat([x, redness, darkness], dim=1)
        return self.head[:-1](self.reduce(self.backbone(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head[-1](self.features(x))


class PairGlyphNet(nn.Module):
    """Glyph network with dedicated heads for known hard confusion groups."""

    def __init__(self, input_mode: str = "rgb") -> None:
        super().__init__()
        self.base = GlyphNet(input_mode=input_mode)
        self.pair_heads = nn.ModuleList(
            [nn.Linear(512, len(chars)) for chars in PAIR_GROUPS]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.base.features(x)
        full_logits = self.base.head[-1](features)
        return full_logits, [head(features) for head in self.pair_heads]


def load_glyph_model(checkpoint: Path, device: torch.device, use_ema: bool = True) -> GlyphNet:
    payload = torch.load(checkpoint, map_location=device)
    model = GlyphNet(input_mode=payload.get("input_mode", "rgb"),
                     hires=payload.get("hires", False),
                     head_mode=payload.get("head_mode", "flat"),
                     crop_width=payload.get("crop_width", GLYPH_CROP_WIDTH)).to(device)
    state = payload.get("ema_state_dict") if use_ema else None
    model.load_state_dict(state if state is not None else payload["state_dict"])
    model.eval()
    return model


def load_pair_glyph_model(
    checkpoint: Path, device: torch.device, use_ema: bool = True
) -> PairGlyphNet:
    payload = torch.load(checkpoint, map_location=device)
    model = PairGlyphNet(input_mode=payload.get("input_mode", "rgb")).to(device)
    state = payload.get("ema_state_dict") if use_ema else None
    model.load_state_dict(state if state is not None else payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def glyph_probabilities(models: list[GlyphNet], images: torch.Tensor) -> torch.Tensor:
    """Return averaged probabilities shaped [B, 5, 36]."""
    batch_size = images.shape[0]
    crop_width = getattr(models[0], "crop_width", GLYPH_CROP_WIDTH)
    crops = extract_glyph_crops(images, crop_width=crop_width).flatten(0, 1)
    probabilities = None
    for model in models:
        current = model(crops).softmax(dim=-1)
        probabilities = current if probabilities is None else probabilities + current
    if probabilities is None:
        raise ValueError("at least one glyph model is required")
    return (probabilities / len(models)).view(batch_size, config.NUM_POSITIONS, config.NUM_CHARS)
