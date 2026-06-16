"""Run the line-removal U-Net over all train+test images and cache the results
as PNGs, so the recognizer can be (re)trained on de-lined images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from denoise import load_denoiser


@torch.no_grad()
def denoise_dir(model, src: Path, dst: Path, device, names: list[str], batch: int = 256) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    buf, fn = [], []
    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device)
        y = model(x).cpu().numpy()
        for name, arr in zip(fn, y):
            im = (arr.transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(im).save(dst / name)
        buf.clear(); fn.clear()
    for i, name in enumerate(names):
        im = Image.open(src / name).convert("RGB")
        buf.append((np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1))
        fn.append(name)
        if len(buf) >= batch:
            flush()
        if (i + 1) % 5000 == 0:
            print(f"  {src.name}: {i+1}/{len(names)}")
    flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser", type=Path, default=config.CHECKPOINT_DIR / "denoiser_v2.pt")
    args = p.parse_args()
    device = torch.device(config.DEVICE)
    model = load_denoiser(args.denoiser, device)

    train_names = sorted(x.name for x in config.TRAIN_IMAGES.iterdir() if x.suffix == ".png")
    test_names = sorted(x.name for x in config.TEST_IMAGES.iterdir() if x.suffix == ".png")
    print(f"denoising {len(train_names)} train + {len(test_names)} test images with {args.denoiser.name}")
    denoise_dir(model, config.TRAIN_IMAGES, config.DENOISED_TRAIN, device, train_names)
    denoise_dir(model, config.TEST_IMAGES, config.DENOISED_TEST, device, test_names)
    print(f"done -> {config.DENOISED_TRAIN.parent.parent}")


if __name__ == "__main__":
    main()
