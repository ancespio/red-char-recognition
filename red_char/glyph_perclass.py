"""Per-class glyph accuracy on val red glyphs (focus on the confusion groups).

Lets us verify a targeted enhancement improves the hard classes (I/1/V/Z...)
WITHOUT regressing overall accuracy, before spending a submission.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import config
from dataset import build_train_dataset, deterministic_split_indices
from glyph import GlyphDataset, glyph_probabilities, load_glyph_model


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--glyph-checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--focus", type=str, default="I1LVZTSGCQ")
    args = p.parse_args()
    device = torch.device(config.DEVICE)
    base = build_train_dataset(cache_in_ram=False)
    _, val_idx = deterministic_split_indices(len(base))
    models = [load_glyph_model(c, device) for c in args.glyph_checkpoints]
    cw = getattr(models[0], "crop_width", 64)
    ds = GlyphDataset(base, val_idx, red_only=True, augment=False, crop_width=cw)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)

    from glyph import extract_glyph_crops  # noqa
    CH = config.CHARSET
    total = Counter(); correct = Counter()
    n = ok = 0
    for crops, targets in loader:
        crops = crops.to(device)
        prob = None
        for m in models:
            cur = m(crops).softmax(-1)
            prob = cur if prob is None else prob + cur
        pred = prob.argmax(-1).cpu()
        for t, pr in zip(targets.tolist(), pred.tolist()):
            total[CH[t]] += 1
            n += 1
            if t == pr:
                correct[CH[t]] += 1; ok += 1
    print(f"overall red-glyph acc = {ok}/{n} = {ok/n:.5f}")
    print("focus classes (char: correct/total = acc):")
    for ch in args.focus:
        if total[ch]:
            print(f"  {ch}: {correct[ch]}/{total[ch]} = {correct[ch]/total[ch]:.4f}")
    # worst classes overall
    worst = sorted((c for c in total if total[c] >= 30),
                   key=lambda c: correct[c] / total[c])[:8]
    print("worst classes (>=30 occ):", [(c, f"{correct[c]}/{total[c]}") for c in worst])


if __name__ == "__main__":
    main()
