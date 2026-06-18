from __future__ import annotations

import torch
from torch import nn

import config


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.activation(self.main(x) + self.shortcut(x)))


class RedCharNet(nn.Module):
    def __init__(
        self,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        neck_dim: int = 512,
        dropout: float = 0.3,
        kernels: tuple[int, int, int, int] = (3, 3, 3, 3),
        block_type: str = "conv",
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        block_cls = ResidualBlock if block_type == "residual" else ConvBlock
        self.backbone = nn.Sequential(
            block_cls(3, c1) if block_type == "residual" else block_cls(3, c1, kernels[0]),
            block_cls(c1, c2) if block_type == "residual" else block_cls(c1, c2, kernels[1]),
            block_cls(c2, c3) if block_type == "residual" else block_cls(c2, c3, kernels[2]),
            block_cls(c3, c4) if block_type == "residual" else block_cls(c3, c4, kernels[3]),
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


class Deep3RedCharNet(nn.Module):
    def __init__(self, neck_dim: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 25, neck_dim, bias=False),
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


def build_model(model_size: str = "base", dropout: float | None = None) -> nn.Module:
    if model_size == "base":
        return RedCharNet(dropout=0.3 if dropout is None else dropout)
    if model_size == "wide":
        return RedCharNet(
            channels=(48, 96, 192, 384),
            neck_dim=768,
            dropout=0.4 if dropout is None else dropout,
        )
    if model_size == "k5":
        return RedCharNet(
            kernels=(5, 5, 3, 3),
            dropout=0.3 if dropout is None else dropout,
        )
    if model_size == "resblock":
        return RedCharNet(
            block_type="residual",
            dropout=0.3 if dropout is None else dropout,
        )
    if model_size == "deep3":
        return Deep3RedCharNet(dropout=0.3 if dropout is None else dropout)
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

    for size in ("k5", "resblock", "deep3"):
        model = build_model(size)
        char_logits, color_logits = model(torch.randn(2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
        print(size, char_logits.shape, color_logits.shape, count_parameters(model))
        assert char_logits.shape == (2, config.NUM_POSITIONS, config.NUM_CHARS)
        assert color_logits.shape == (2, config.NUM_POSITIONS, config.NUM_COLORS)
