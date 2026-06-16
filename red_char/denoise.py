"""Small U-Net for interference-line removal (60x200 RGB -> RGB)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LineRemoverUNet(nn.Module):
    """4-level U-Net. The extra depth (down to 1/8 resolution) gives a large
    receptive field so full-span *thick* interference lines can be identified
    from global context and removed, not just thin ones."""

    def __init__(self, base: int = 40) -> None:
        super().__init__()
        self.e1 = _Block(3, base)
        self.e2 = _Block(base, base * 2)
        self.e3 = _Block(base * 2, base * 4)
        self.e4 = _Block(base * 4, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = _Block(base * 4, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.d3 = _Block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _Block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _Block(base * 2, base)
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        s1 = self.e1(x)                 # H,    W
        s2 = self.e2(self.pool(s1))     # H/2
        s3 = self.e3(self.pool(s2))     # H/4
        s4 = self.e4(self.pool(s3))     # H/8
        m = self.mid(s4)
        d3 = self.d3(torch.cat([self._fit(self.up3(m), s3), s3], 1))
        d2 = self.d2(torch.cat([self._fit(self.up2(d3), s2), s2], 1))
        d1 = self.d1(torch.cat([self._fit(self.up1(d2), s1), s1], 1))
        return (x + self.out(d1)).clamp(0.0, 1.0)

    @staticmethod
    def _fit(x, ref):
        # handle odd sizes from pooling (60->30->15, 200->100->50 are clean here)
        if x.shape[-2:] != ref.shape[-2:]:
            x = nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x


def load_denoiser(checkpoint: Path, device: torch.device, use_ema: bool = True) -> LineRemoverUNet:
    payload = torch.load(checkpoint, map_location=device)
    model = LineRemoverUNet(base=payload.get("base", 40)).to(device)
    state = payload.get("ema_state_dict") if use_ema else None
    model.load_state_dict(state if state is not None else payload["state_dict"])
    model.eval()
    return model
