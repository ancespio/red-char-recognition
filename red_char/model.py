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
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 12, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.char_head = nn.Linear(512, config.NUM_POSITIONS * config.NUM_CHARS)
        self.color_head = nn.Linear(512, config.NUM_POSITIONS * config.NUM_COLORS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.neck(self.backbone(x))
        char_logits = self.char_head(features).view(-1, config.NUM_POSITIONS, config.NUM_CHARS)
        color_logits = self.color_head(features).view(-1, config.NUM_POSITIONS, config.NUM_COLORS)
        return char_logits, color_logits


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    net = RedCharNet()
    char, color = net(torch.randn(2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    params = count_parameters(net)
    print(char.shape, color.shape, params)
    assert char.shape == (2, config.NUM_POSITIONS, config.NUM_CHARS)
    assert color.shape == (2, config.NUM_POSITIONS, config.NUM_COLORS)
    assert 5_000_000 <= params <= 8_000_000
