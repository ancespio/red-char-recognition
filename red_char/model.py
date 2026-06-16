from __future__ import annotations

import torch
from torch import nn

import config


# =============================================================================
# v1: original plain 4-block CNN (kept verbatim for a fair baseline comparison)
# =============================================================================
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


# =============================================================================
# v2: residual + Squeeze-Excite + CoordConv backbone
# -----------------------------------------------------------------------------
# Why each piece helps this specific task:
#   * Residual blocks       -> deeper effective backbone, better gradient flow,
#                              so character-shape features are learned more
#                              reliably (char accuracy is the exact-match
#                              bottleneck since exact ~= joint_pos_acc**5).
#   * Squeeze-Excite        -> cheap channel attention; lets the colour head
#                              emphasise the channels that separate red from
#                              the orange/purple distractors.
#   * CoordConv (x/y coord) -> injects absolute column position, which is
#                              exactly what a "read the i-th character" head
#                              needs and what a translation-invariant CNN
#                              otherwise has to infer indirectly.
# The Flatten neck and the (char_logits[B,5,36], color_logits[B,5,2]) output
# contract are unchanged, so train / evaluate / predict stay drop-in.
# =============================================================================
class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class ResidualSEStage(nn.Module):
    """Two 3x3 conv-BN-ReLU layers + SE + identity skip, then 2x downsample."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcite(out_channels)
        self.act = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act(out + identity)
        return self.pool(out)


class CoordConv2d(nn.Module):
    """Append normalised x/y coordinate channels before a 1x1 conv mix."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = xs.view(1, 1, 1, w).expand(b, 1, h, w)
        ys = ys.view(1, 1, h, 1).expand(b, 1, h, w)
        return self.conv(torch.cat([x, xs, ys], dim=1))


class RedCharNetV2(nn.Module):
    def __init__(
        self,
        dropout: float = 0.3,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        pools: tuple[bool, ...] | None = None,
    ) -> None:
        super().__init__()
        self.coord = CoordConv2d(3, widths[0] // 2)
        in_ch = widths[0] // 2
        if pools is None:
            pools = (True,) * len(widths)
        stages = []
        h, w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
        for out_ch, pool in zip(widths, pools):
            stages.append(ResidualSEStage(in_ch, out_ch, pool=pool))
            in_ch = out_ch
            if pool:
                h, w = h // 2, w // 2
        self.backbone = nn.Sequential(*stages)
        feat_dim = widths[-1] * h * w
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.char_head = nn.Linear(512, config.NUM_POSITIONS * config.NUM_CHARS)
        self.color_head = nn.Linear(512, config.NUM_POSITIONS * config.NUM_COLORS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.neck(self.backbone(self.coord(x)))
        char_logits = self.char_head(features).view(-1, config.NUM_POSITIONS, config.NUM_CHARS)
        color_logits = self.color_head(features).view(-1, config.NUM_POSITIONS, config.NUM_COLORS)
        return char_logits, color_logits


# =============================================================================
# v3: ImageNet-pretrained ResNet-18 backbone + per-position flatten neck
# -----------------------------------------------------------------------------
# Why: the from-scratch v1/v2 backbones plateaued at ~0.988 per-position char
# accuracy (the exact-match bottleneck) with train_loss == val_loss, i.e. no
# overfitting -> the ceiling is feature quality, not regularisation. An
# ImageNet-pretrained ResNet-18 brings far stronger low/mid-level stroke and
# edge features, which is exactly what separates the hard glyph confusions
# (O/0, G/C, I/1/L, Q/O, S/5...). We keep the *width* resolution high (dilate
# the last stage to stride 1) so the 5 per-position heads still get a wide
# feature grid to read columns from, and we Flatten (never GAP) to preserve
# absolute column position. Output contract is identical to v1/v2.
# =============================================================================
class RedCharNetV3(nn.Module):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, dropout: float = 0.3, pretrained: bool = True, keep_wide: bool = True) -> None:
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        if keep_wide:
            # Make layer4 stride 1 (dilate) to keep a 4x13 grid instead of 2x7,
            # giving the per-column heads finer horizontal resolution.
            net.layer4[0].conv1.stride = (1, 1)
            net.layer4[0].downsample[0].stride = (1, 1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

        # Infer flattened feature dim from a dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
            feat = self._features((dummy - self.mean) / self.std)
            feat_dim = feat.flatten(1).shape[1]
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.char_head = nn.Linear(512, config.NUM_POSITIONS * config.NUM_CHARS)
        self.color_head = nn.Linear(512, config.NUM_POSITIONS * config.NUM_COLORS)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x - self.mean) / self.std
        features = self.neck(self._features(x))
        char_logits = self.char_head(features).view(-1, config.NUM_POSITIONS, config.NUM_CHARS)
        color_logits = self.color_head(features).view(-1, config.NUM_POSITIONS, config.NUM_COLORS)
        return char_logits, color_logits


# =============================================================================
# v2hi: same residual+SE+CoordConv recipe as v2 but keeps higher spatial
# resolution. v2 collapses 60x200 -> 3x12 (~2-3px per glyph), which is too
# coarse to separate detail confusions (O/0, G/C, I/1/L...). v2hi pools only
# 3x (-> 7x25) so each character keeps ~5px of horizontal detail, then a 1x1
# conv reduces channels to keep the flatten neck the same size as v2 (no
# over-parameterised FC -> no overfit). Wider stages add capacity for the
# extra detail. Output contract identical.
# =============================================================================
class RedCharNetV2Hi(nn.Module):
    def __init__(self, dropout: float = 0.3, widths: tuple[int, ...] = (48, 96, 192, 384),
                 reduce_ch: int = 48, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.coord = CoordConv2d(in_channels, widths[0] // 2)
        in_ch = widths[0] // 2
        pools = (True, True, True, False)  # 60x200 -> 30x100 -> 15x50 -> 7x25 -> 7x25
        stages = []
        h, w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
        for out_ch, pool in zip(widths, pools):
            stages.append(ResidualSEStage(in_ch, out_ch, pool=pool))
            in_ch = out_ch
            if pool:
                h, w = h // 2, w // 2
        self.backbone = nn.Sequential(*stages)
        self.reduce = nn.Sequential(
            nn.Conv2d(widths[-1], reduce_ch, 1, bias=False),
            nn.BatchNorm2d(reduce_ch),
            nn.ReLU(inplace=True),
        )
        feat_dim = reduce_ch * h * w
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.char_head = nn.Linear(512, config.NUM_POSITIONS * config.NUM_CHARS)
        self.color_head = nn.Linear(512, config.NUM_POSITIONS * config.NUM_COLORS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.neck(self.reduce(self.backbone(self.coord(x))))
        char_logits = self.char_head(features).view(-1, config.NUM_POSITIONS, config.NUM_CHARS)
        color_logits = self.color_head(features).view(-1, config.NUM_POSITIONS, config.NUM_COLORS)
        return char_logits, color_logits


def build_model(name: str | None = None, dropout: float = 0.3) -> nn.Module:
    name = (name or config.MODEL).lower()
    if name == "v1":
        return RedCharNet(dropout=dropout)
    if name == "v2":
        return RedCharNetV2(dropout=dropout)
    if name == "v2hi":
        return RedCharNetV2Hi(dropout=dropout)
    if name == "v2hi6":
        return RedCharNetV2Hi(dropout=dropout, in_channels=6)
    if name == "v3":
        return RedCharNetV3(dropout=dropout)
    raise ValueError(f"unknown model name: {name!r} (expected 'v1', 'v2', 'v2hi' or 'v3')")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    dummy = torch.randn(2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    for name in ("v1", "v2", "v3"):
        net = build_model(name).eval()
        char, color = net(dummy)
        params = count_parameters(net)
        print(f"{name}: char={tuple(char.shape)} color={tuple(color.shape)} params={params:,}")
        assert char.shape == (2, config.NUM_POSITIONS, config.NUM_CHARS)
        assert color.shape == (2, config.NUM_POSITIONS, config.NUM_COLORS)
        if name in ("v1", "v2"):
            assert 4_000_000 <= params <= 12_000_000
    print("model self-test passed")
