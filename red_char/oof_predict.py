"""Assemble out-of-fold (OOF) predictions over all 50000 training images.

For each fold k, loads that fold's primary v2hi model(s) and glyph model(s)
(which trained WITHOUT fold k), predicts fold k's images, and writes them into
global arrays. Result: an unbiased prediction for every training image, saved
to outputs/oof/oof.pt for threshold tuning / large-scale error analysis.

Naming convention (see run_kfold.sh):
  primary: best_f{k}{tag}.pt   for tag in --primary-tags  (e.g. s1 s2)
  glyph:   best_gff{k}.pt       (or --glyph-prefix)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF

import config
from dataset import build_train_dataset
from glyph import glyph_probabilities, load_glyph_model
from kfold import N_FOLDS, fold_split
from predict import load_model


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--primary-tags", nargs="+", default=["s1", "s2"],
                   help="per-fold primary checkpoint suffixes; loads best_f{k}{tag}.pt")
    p.add_argument("--glyph-prefix", default="gff", help="per-fold glyph checkpoint = best_{prefix}{k}.pt")
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--x-tta", action="store_true", help="average original + horizontal +/-4px for primary")
    p.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "oof" / "oof.pt")
    args = p.parse_args()

    device = torch.device(config.DEVICE)
    base = build_train_dataset(cache_in_ram=False)
    n = len(base)
    CK = config.CHECKPOINT_DIR

    primary_char = torch.zeros(n, config.NUM_POSITIONS, config.NUM_CHARS)
    color = torch.zeros(n, config.NUM_POSITIONS, config.NUM_COLORS)
    glyph = torch.zeros(n, config.NUM_POSITIONS, config.NUM_CHARS)
    char_t = torch.zeros(n, config.NUM_POSITIONS, dtype=torch.long)
    color_t = torch.zeros(n, config.NUM_POSITIONS, dtype=torch.long)
    filled = torch.zeros(n, dtype=torch.bool)

    for k in range(args.n_folds):
        _, oof_idx = fold_split(n, k, args.n_folds)
        prim_ckpts = [CK / f"best_f{k}{t}.pt" for t in args.primary_tags]
        glyph_ckpts = [CK / f"best_{args.glyph_prefix}{k}.pt"]
        for c in prim_ckpts + glyph_ckpts:
            if not c.exists():
                raise FileNotFoundError(f"missing fold-{k} checkpoint: {c}")
        prim_models = [load_model(c, device) for c in prim_ckpts]
        glyph_models = [load_glyph_model(c, device) for c in glyph_ckpts]

        loader = DataLoader(Subset(base, oof_idx), batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=device.type == "cuda")
        cursor = 0
        for images, ct, kt in loader:
            images = images.to(device, non_blocking=True)
            shifts = (0, -4, 4) if args.x_tta else (0,)
            pc = kp = None
            for shift in shifts:
                sh = images if shift == 0 else VF.affine(
                    images, angle=0, translate=[shift, 0], scale=1.0, shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR, fill=[1.0, 1.0, 1.0])
                for m in prim_models:
                    cl, kl = m(sh)
                    a, b = F.softmax(cl, -1), F.softmax(kl, -1)
                    pc = a if pc is None else pc + a
                    kp = b if kp is None else kp + b
            div = len(prim_models) * len(shifts)
            gp = glyph_probabilities(glyph_models, images)
            bs = images.size(0)
            idx = torch.tensor(oof_idx[cursor:cursor + bs])
            primary_char[idx] = (pc / div).cpu()
            color[idx] = (kp / div).cpu()
            glyph[idx] = gp.cpu()
            char_t[idx] = ct
            color_t[idx] = kt
            filled[idx] = True
            cursor += bs
        print(f"fold {k}: filled {len(oof_idx)} OOF preds "
              f"(primary={[c.name for c in prim_ckpts]}, glyph={glyph_ckpts[0].name})")

    assert bool(filled.all()), f"unfilled OOF entries: {(~filled).sum().item()}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"primary_char": primary_char, "color": color, "glyph": glyph,
                "char_target": char_t, "color_target": color_t,
                "x_tta": args.x_tta, "primary_tags": args.primary_tags}, args.out)
    print(f"saved OOF predictions for {n} images -> {args.out}")


if __name__ == "__main__":
    main()
