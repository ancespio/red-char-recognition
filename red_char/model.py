from __future__ import annotations

import torch
from torch import nn

import config


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RedCharNet(nn.Module):
    def __init__(
        self,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        neck_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.backbone = nn.Sequential(
            ConvBlock(3, c1),
            ConvBlock(c1, c2),
            ConvBlock(c2, c3),
            ConvBlock(c3, c4),
        )
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4 * 3 * 12, neck_dim, bias=False),
            nn.BatchNorm1d(neck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.char_head = nn.Linear(neck_dim, config.NUM_POSITIONS * config.NUM_CHARS)
        self.color_head = nn.Linear(neck_dim, config.NUM_POSITIONS * config.NUM_COLORS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.neck(self.backbone(x))
        char_logits = self.char_head(features).view(-1, config.NUM_POSITIONS, config.NUM_CHARS)
        color_logits = self.color_head(features).view(-1, config.NUM_POSITIONS, config.NUM_COLORS)
        return char_logits, color_logits


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_model(model_size: str = "base", dropout: float | None = None) -> RedCharNet:
    if model_size == "base":
        return RedCharNet(dropout=0.3 if dropout is None else dropout)
    if model_size == "wide":
        return RedCharNet(
            channels=(48, 96, 192, 384),
            neck_dim=768,
            dropout=0.4 if dropout is None else dropout,
        )
    raise ValueError(f"unknown model_size: {model_size}")


if __name__ == "__main__":
    net = build_model("base")
    char, color = net(torch.randn(2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    params = count_parameters(net)
    print(char.shape, color.shape, params)
    assert char.shape == (2, config.NUM_POSITIONS, config.NUM_CHARS)
    assert color.shape == (2, config.NUM_POSITIONS, config.NUM_COLORS)
    assert 5_000_000 <= params <= 8_000_000

    wide = build_model("wide")
    wide_char, wide_color = wide(torch.randn(2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    wide_params = count_parameters(wide)
    print(wide_char.shape, wide_color.shape, wide_params)
    assert wide_char.shape == (2, config.NUM_POSITIONS, config.NUM_CHARS)
    assert wide_color.shape == (2, config.NUM_POSITIONS, config.NUM_COLORS)
    assert wide_params > params
