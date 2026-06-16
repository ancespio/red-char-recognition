"""Quantify ensemble exact-match on the fixed validation split.

Reuses the same deterministic split as training/evaluate so the number is
directly comparable to the single-model val exact-match.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import config
from dataset import (
    build_train_dataset,
    deterministic_split_indices,
    decode_prediction,
)
from predict import load_model


@torch.no_grad()
def ensemble_exact(models, loader, device) -> dict:
    n = 0
    exact = 0
    char_correct = 0
    color_correct = 0
    pos_total = 0
    for images, char_t, color_t in loader:
        images = images.to(device, non_blocking=True)
        char_prob = color_prob = None
        for m in models:
            cl, kl = m(images)
            cp, kp = F.softmax(cl, -1), F.softmax(kl, -1)
            char_prob = cp if char_prob is None else char_prob + cp
            color_prob = kp if color_prob is None else color_prob + kp
        char_pred = char_prob.argmax(-1).cpu()
        color_pred = color_prob.argmax(-1).cpu()
        char_correct += (char_pred == char_t).sum().item()
        color_correct += (color_pred == color_t).sum().item()
        pos_total += char_t.numel()
        for i in range(images.size(0)):
            pred_s = decode_prediction(char_pred[i].tolist(), color_pred[i].tolist())
            true_s = decode_prediction(char_t[i].tolist(), color_t[i].tolist())
            exact += int(pred_s == true_s)
            n += 1
    return {
        "exact": exact / n,
        "char_acc": char_correct / pos_total,
        "color_acc": color_correct / pos_total,
        "n": n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, nargs="+",
                   default=[config.CHECKPOINT_DIR / "best.pt"])
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split", choices=["val", "train", "both"], default="val",
                   help="evaluate on the clean (no-aug) train split, the val split, or both")
    p.add_argument("--denoised", action="store_true", help="evaluate on line-removed images")
    p.add_argument("--concat-denoised", action="store_true", help="6ch: original + denoised")
    args = p.parse_args()

    device = torch.device(config.DEVICE)
    if args.denoised:
        config.TRAIN_IMAGES = config.DENOISED_TRAIN
    base = build_train_dataset(cache_in_ram=False,
                               denoised_dir=config.DENOISED_TRAIN if args.concat_denoised else None)
    train_idx, val_idx = deterministic_split_indices(len(base))
    models = [load_model(c, device, use_ema=args.use_ema) for c in args.checkpoints]
    splits = {"val": val_idx, "train": train_idx} if args.split == "both" else {args.split: (train_idx if args.split == "train" else val_idx)}
    for name, idx in splits.items():
        ds = Subset(base, idx)
        loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=device.type == "cuda")
        print(f"[{name}] {len(models)} model(s) on {len(ds)} images: {ensemble_exact(models, loader, device)}")


if __name__ == "__main__":
    main()
