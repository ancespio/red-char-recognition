"""Synthetic captcha generator for training a line-removal U-Net.

Produces (clean, lined) pairs: clean = chars on noisy light background (NO
interference lines); lined = clean + random straight interference lines. A
U-Net trained input=lined -> target=clean learns to remove line-like structures
while preserving the compact character blobs. Exact font match is not required:
the discriminating cue is geometric (lines are long/straight/spanning; chars are
compact), so we randomise over several bold fonts for robustness.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config

W, H = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
CHARSET = config.CHARSET

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]
_FONTS = [p for p in _FONT_PATHS if Path(p).exists()]


def _rand_color(light: bool = False) -> tuple[int, int, int]:
    if light:
        return tuple(random.randint(160, 235) for _ in range(3))
    # saturated-ish ink colour (avoid near-white)
    base = [random.randint(0, 200) for _ in range(3)]
    base[random.randint(0, 2)] = random.randint(0, 120)  # ensure one dark channel
    return tuple(base)


def _background() -> np.ndarray:
    g = random.randint(244, 252)
    img = np.full((H, W, 3), g, np.uint8).astype(np.int16)
    img += np.random.randint(-3, 4, (H, W, 3))
    # sparse coloured speckle
    n = random.randint(60, 260)
    ys = np.random.randint(0, H, n); xs = np.random.randint(0, W, n)
    for y, x in zip(ys, xs):
        img[y, x] = _rand_color()
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_chars(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)
    n = config.NUM_POSITIONS
    for pos in range(n):
        ch = random.choice(CHARSET)
        size = random.randint(26, 50)
        font = ImageFont.truetype(random.choice(_FONTS), size)
        # nominal slot centre + jitter
        cx = int((pos * 2 + 1) * W / (2 * n) + random.randint(-12, 12))
        cy = int(H / 2 + random.randint(-8, 8))
        glyph = Image.new("RGBA", (size + 20, size + 20), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glyph)
        gd.text((10, 5), ch, font=font, fill=_rand_color() + (255,))
        glyph = glyph.rotate(random.uniform(-18, 18), expand=True, resample=Image.BILINEAR)
        img.paste(glyph, (cx - glyph.width // 2, cy - glyph.height // 2), glyph)


def _add_lines(img: Image.Image, lo: int = 3, hi: int = 7) -> None:
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(lo, hi)):
        # endpoints near opposite edges so the line spans the image
        if random.random() < 0.5:
            p0 = (random.randint(-10, 30), random.randint(0, H))
            p1 = (random.randint(W - 30, W + 10), random.randint(0, H))
        else:
            p0 = (random.randint(0, W), random.randint(-10, 20))
            p1 = (random.randint(0, W), random.randint(H - 20, H + 10))
        # ~45% thick lines (width 3-6, comparable to character strokes) so the
        # U-Net must learn to remove thick full-span lines, not just thin ones.
        width = random.randint(3, 6) if random.random() < 0.45 else random.randint(1, 2)
        draw.line([p0, p1], fill=_rand_color(), width=width)


def _red_ink() -> tuple[int, int, int]:
    # red gamut: R clearly dominant; covers bright to dull red
    r = random.randint(150, 240)
    g = random.randint(0, max(0, r - 110))
    b = random.randint(0, max(0, r - 110))
    return (r, g, b)


def _nonred_ink() -> tuple[int, int, int]:
    # any saturated colour that is NOT red-dominant (green/blue/purple/teal/...)
    for _ in range(8):
        c = [random.randint(0, 200) for _ in range(3)]
        c[random.randint(0, 2)] = random.randint(0, 120)
        if c[0] - max(c[1], c[2]) < 30:  # ensure not red-dominant
            return tuple(c)
    return (40, 90, 180)


def make_labeled() -> tuple[np.ndarray, list[int], list[int]]:
    """Return (lined_image[3,H,W] float in [0,1], char_idx[5], color_idx[5]).

    Procedural captcha matching the real style: generic bold sans glyph at the
    real ~25-36px size, ~50% red (red gamut) / 50% non-red, jitter, then lines +
    noise. Labels are exact by construction (for mixing into recognition training)."""
    img = Image.fromarray(_background())
    draw = ImageDraw.Draw(img)
    n = config.NUM_POSITIONS
    chars, colors = [], []
    for pos in range(n):
        ch = random.choice(CHARSET)
        is_red = random.random() < 0.5
        size = random.randint(24, 36)
        font = ImageFont.truetype(random.choice(_FONTS), size)
        cx = int((pos * 2 + 1) * W / (2 * n) + random.randint(-10, 10))
        cy = int(H / 2 + random.randint(-7, 7))
        glyph = Image.new("RGBA", (size + 24, size + 24), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((12, 6), ch, font=font,
                                   fill=(_red_ink() if is_red else _nonred_ink()) + (255,))
        glyph = glyph.rotate(random.uniform(-16, 16), expand=True, resample=Image.BILINEAR)
        img.paste(glyph, (cx - glyph.width // 2, cy - glyph.height // 2), glyph)
        chars.append(config.CHAR_TO_IDX[ch])
        colors.append(config.RED_INDEX if is_red else config.NON_RED_INDEX)
    _add_lines(img, lo=2, hi=4)  # real images are sparser than the denoiser-synth
    arr = (np.asarray(img, np.float32) / 255.0).transpose(2, 0, 1)
    noisy = np.clip(arr + np.random.randn(*arr.shape).astype(np.float32) * 0.02, 0, 1)
    return noisy, chars, colors


def make_pair() -> tuple[np.ndarray, np.ndarray]:
    """Return (lined, clean) float32 arrays [3,H,W] in [0,1]."""
    bg = _background()
    clean = Image.fromarray(bg.copy())
    _draw_chars(clean)
    lined = clean.copy()
    _add_lines(lined)
    to_t = lambda im: (np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)
    return to_t(lined), to_t(clean)


if __name__ == "__main__":
    # dump a synthetic-vs-real comparison montage for visual validation
    from dataset import build_train_dataset, deterministic_split_indices
    config.ensure_output_dirs()
    rows = []
    for _ in range(6):
        lined, clean = make_pair()
        l = (lined.transpose(1, 2, 0) * 255).astype(np.uint8)
        c = (clean.transpose(1, 2, 0) * 255).astype(np.uint8)
        rows.append(np.concatenate([l, c], axis=1))
    base = build_train_dataset(cache_in_ram=False)
    ti, _ = deterministic_split_indices(len(base))
    for i in ti[:6]:
        r = (base[i][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rows.append(np.concatenate([r, np.full_like(r, 255)], axis=1))
    sheet = np.concatenate(rows, axis=0)
    out = config.EDA_DIR / "synth_vs_real.png"
    Image.fromarray(sheet).resize((sheet.shape[1] * 2, sheet.shape[0] * 2), Image.NEAREST).save(out)
    print("fonts:", _FONTS)
    print("saved (top6=synth lined|clean, bottom6=real|blank):", out)
