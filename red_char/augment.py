from __future__ import annotations

import random

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import config


def _faint_fade(image: torch.Tensor) -> torch.Tensor:
    """Fade the ink toward the white background along a random linear gradient.

    Simulates a stroke whose colour is uneven / very light on one side (e.g. a V
    whose right stroke is faint red), which otherwise gets dropped and turns the
    glyph into a simpler shape (V->I). Teaches the model that a faint stroke is
    still part of the character. Background (already white) is unchanged; only
    ink darkens-> whitens proportionally to the gradient."""
    _, h, w = image.shape
    lo = float(torch.empty(1).uniform_(0.25, 0.6))   # strongest fade factor
    ang = float(torch.empty(1).uniform_(0, 6.2832))
    dx, dy = float(torch.cos(torch.tensor(ang))), float(torch.sin(torch.tensor(ang)))
    xs = torch.linspace(0, 1, w).view(1, w)
    ys = torch.linspace(0, 1, h).view(h, 1)
    coord = (xs * dx + ys * dy)
    coord = (coord - coord.min()) / (coord.max() - coord.min() + 1e-6)  # [0,1]
    if random.random() < 0.5:
        coord = 1 - coord
    alpha = (lo + (1 - lo) * coord).unsqueeze(0)     # [1,h,w] in [lo,1]
    # ink = 1-image; fade ink by alpha -> out = 1 - (1-image)*alpha
    return (1 - (1 - image) * alpha).clamp_(0.0, 1.0)


def _cutout(image: torch.Tensor, n: int) -> torch.Tensor:
    """Erase n small rectangles (occlusion sim) so the model learns to read a
    partially-covered glyph. Boxes are kept small (≤~35% of each side) so the
    character is never fully destroyed; fill is random (white / grey / a colour)
    to mimic background or an opaque distractor covering part of the stroke."""
    _, h, w = image.shape
    out = image.clone()
    for _ in range(n):
        bh = int(h * float(torch.empty(1).uniform_(0.12, 0.35)))
        bw = int(w * float(torch.empty(1).uniform_(0.12, 0.35)))
        y0 = int(torch.randint(0, max(1, h - bh), (1,)))
        x0 = int(torch.randint(0, max(1, w - bw), (1,)))
        fill = float(torch.empty(1).uniform_(0.0, 1.0))
        out[:, y0:y0 + bh, x0:x0 + bw] = fill
    return out


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
        cutout_p: float = 0.0,
        faint_p: float = 0.0,
    ) -> None:
        self.red_line_p = red_line_p
        self.cutout_p = cutout_p
        self.faint_p = faint_p
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
            image = _draw_red_lines(image, n=random.randint(1, 5))  # heavier line occlusion

        if self.cutout_p > 0 and float(torch.rand(1)) < self.cutout_p:
            image = _cutout(image, n=random.randint(1, 2))

        if self.faint_p > 0 and float(torch.rand(1)) < self.faint_p:
            image = _faint_fade(image)

        if self.noise_std > 0:
            image = image + torch.randn_like(image) * self.noise_std
            image = image.clamp_(0.0, 1.0)
        return image
