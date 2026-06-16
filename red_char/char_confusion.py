"""Validation-set character-confusion statistics for the deployment reranker.

Over ALL red character positions in the val split (~6231 glyphs), compares the
final predicted char vs the true char and tabulates confusions (true->pred),
per-true-char recall, and per-position counts. Colour is taken as ground truth
here so this isolates the *character* recognition problem.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF

import config
from dataset import build_train_dataset, deterministic_split_indices
from glyph import glyph_probabilities, load_glyph_model
from predict import load_model
from eval_reranker import selective_rerank


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--glyph-checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--primary-margin-max", type=float, default=0.40)
    p.add_argument("--glyph-margin-min", type=float, default=0.35)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    device = torch.device(config.DEVICE)
    base = build_train_dataset(cache_in_ram=False)
    _, val_idx = deterministic_split_indices(len(base))
    loader = DataLoader(Subset(base, val_idx), batch_size=256, shuffle=False, num_workers=4)
    prim = [load_model(c, device) for c in args.checkpoints]
    glyph = [load_glyph_model(c, device) for c in args.glyph_checkpoints]

    P = []; G = []; CT = []; KT = []
    for images, ct, kt in loader:
        images = images.to(device)
        pc = None
        for shift in (0, -4, 4):
            sh = images if shift == 0 else VF.affine(images, angle=0, translate=[shift, 0], scale=1.0,
                shear=[0., 0.], interpolation=InterpolationMode.BILINEAR, fill=[1., 1., 1.])
            for m in prim:
                pc = F.softmax(m(sh)[0], -1) if pc is None else pc + F.softmax(m(sh)[0], -1)
        P.append((pc / (len(prim) * 3)).cpu()); G.append(glyph_probabilities(glyph, images).cpu())
        CT.append(ct); KT.append(kt)
    P = torch.cat(P); G = torch.cat(G); CT = torch.cat(CT); KT = torch.cat(KT)
    char_pred = selective_rerank(P, G, args.top_k, args.primary_margin_max, args.glyph_margin_min)

    CH = config.CHARSET
    red = KT == config.RED_INDEX
    n_red = int(red.sum())
    wrong = red & (char_pred != CT)
    pairs = Counter()
    per_true_total = Counter(); per_true_wrong = Counter()
    pos_wrong = Counter()
    for i in range(CT.shape[0]):
        for pos in range(config.NUM_POSITIONS):
            if not red[i, pos]:
                continue
            t = CH[int(CT[i, pos])]
            per_true_total[t] += 1
            if char_pred[i, pos] != CT[i, pos]:
                pairs[(t, CH[int(char_pred[i, pos])])] += 1
                per_true_wrong[t] += 1
                pos_wrong[int(pos)] += 1

    print(f"red glyphs={n_red}  char errors={int(wrong.sum())}  red-char acc={1-int(wrong.sum())/n_red:.5f}")
    print(f"\nconfusion pairs (true->pred : count):")
    for (t, pp), c in pairs.most_common():
        print(f"  {t} -> {pp} : {c}")
    print(f"\nworst characters (true char : errors/occurrences = error-rate):")
    for t in sorted(per_true_wrong, key=lambda x: -per_true_wrong[x]):
        print(f"  {t}: {per_true_wrong[t]}/{per_true_total[t]} = {per_true_wrong[t]/per_true_total[t]:.3f}")
    print(f"\nchar errors by position 0-4: {dict(sorted(pos_wrong.items()))}")


if __name__ == "__main__":
    main()
