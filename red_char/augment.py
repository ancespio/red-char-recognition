from __future__ import annotations

import random

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import config


def _draw_red_lines(image: torch.Tensor, n: int) -> torch.Tensor:
    """Overlay n random RED interference lines on a [3,H,W] image in [0,1].

    After red-isolation the only residual interference is red lines; adding
    synthetic red lines that DON'T change the character label teaches the
    recogniser to read through them. Colour is constrained to the red gamut.
    """
    _, h, w = image.shape
    out = image.clone()
    yy = torch.arange(h, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, w)
    for _ in range(n):
        # random full-span line: a*x + b*y + c = 0 via two edge endpoints
        x0, y0 = random.uniform(-5, w + 5), random.uniform(-5, h + 5)
        ang = random.uniform(-0.5, 0.5) if random.random() < 0.5 else random.uniform(2.64, 3.64)
        dx, dy = torch.cos(torch.tensor(ang)), torch.sin(torch.tensor(ang))
        # distance of each pixel to the line through (x0,y0) with direction (dx,dy)
        dist = ((xx - x0) * (-dy) + (yy - y0) * dx).abs()
        width = random.uniform(0.6, 2.6)
        mask = (dist <= width).float()  # [h,w]
        r = random.uniform(0.55, 1.0); g = random.uniform(0.0, 0.45); b = random.uniform(0.0, 0.45)
        colour = torch.tensor([r, g, b]).view(3, 1, 1)
        out = out * (1 - mask) + colour * mask
    return out.clamp_(0.0, 1.0)


class TrainAugment:
    """On-the-fly geometric + light additive-noise augmentation.

    Applied to a single image tensor ``[3, H, W]`` in ``[0, 1]``.

    Deliberately *colour-preserving*: only translation / scale / rotation and
    a tiny amount of Gaussian noise. No hue, saturation, channel-swap or
    grayscale operations, because the red colour is the supervision signal for
    the colour heads (see DEV_PLAN risk #2). Border pixels exposed by the
    affine warp are filled with ~white to match the light noisy background.
    """

    def __init__(
        self,
        translate: float | None = None,
        scale: tuple[float, float] | None = None,
        degrees: float | None = None,
        noise_std: float | None = None,
        fill: float = 1.0,
        heavy: bool | None = None,
        red_line_p: float = 0.0,
    ) -> None:
        self.red_line_p = red_line_p
        self.heavy = config.AUG_HEAVY if heavy is None else heavy
        if self.heavy:
            self.translate = config.AUG_H_TRANSLATE if translate is None else translate
            self.scale = config.AUG_H_SCALE if scale is None else scale
            self.degrees = config.AUG_H_DEGREES if degrees is None else degrees
            self.noise_std = config.AUG_H_NOISE_STD if noise_std is None else noise_std
            self.perspective = config.AUG_H_PERSPECTIVE
            self.perspective_p = config.AUG_H_PERSPECTIVE_P
            self.blur_p = config.AUG_H_BLUR_P
        else:
            self.translate = config.AUG_TRANSLATE if translate is None else translate
            self.scale = config.AUG_SCALE if scale is None else scale
            self.degrees = config.AUG_DEGREES if degrees is None else degrees
            self.noise_std = config.AUG_NOISE_STD if noise_std is None else noise_std
            self.perspective = 0.0
            self.perspective_p = 0.0
            self.blur_p = 0.0
        self.fill = fill

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        c, h, w = image.shape
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees).item())
        max_dx = self.translate * w
        max_dy = self.translate * h
        tx = int(torch.empty(1).uniform_(-max_dx, max_dx).round().item())
        ty = int(torch.empty(1).uniform_(-max_dy, max_dy).round().item())
        scale = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())

        image = F.affine(
            image,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=F.InterpolationMode.BILINEAR,
            fill=[self.fill] * c,
        )

        # Colour-preserving perspective warp (geometry only).
        if self.perspective_p > 0 and float(torch.rand(1)) < self.perspective_p:
            startpoints, endpoints = T.RandomPerspective.get_params(w, h, self.perspective)
            image = F.perspective(
                image, startpoints, endpoints,
                interpolation=F.InterpolationMode.BILINEAR, fill=[self.fill] * c,
            )

        # Light Gaussian blur (focus variation); intensity-neutral on hue.
        if self.blur_p > 0 and float(torch.rand(1)) < self.blur_p:
            sigma = float(torch.empty(1).uniform_(0.1, 1.1).item())
            image = F.gaussian_blur(image, kernel_size=5, sigma=sigma)

        if self.red_line_p > 0 and float(torch.rand(1)) < self.red_line_p:
            image = _draw_red_lines(image, n=random.randint(1, 3))

        if self.noise_std > 0:
            image = image + torch.randn_like(image) * self.noise_std
            image = image.clamp_(0.0, 1.0)
        return image
