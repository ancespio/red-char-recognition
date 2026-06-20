"""Pull out the most-uncertain TEST images (inputs only, no labels) for human
inspection. Uncertainty = the lowest per-position decision confidence (char and
colour) over the 5 slots, using the deployment ensemble + glyph reranker.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF

import config
from dataset import RedCharDataset, Sample, decode_prediction, load_submission_sample
from glyph import glyph_probabilities, load_glyph_model
from predict import load_model
from eval_reranker import selective_rerank


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--glyph-checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--primary-margin-max", type=float, default=0.40)
    p.add_argument("--glyph-margin-min", type=float, default=0.05)
    p.add_argument("--red-threshold", type=float, default=0.20)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--outdir", type=Path, default=config.EDA_DIR / "test_uncertain")
    args = p.parse_args()

    device = torch.device(config.DEVICE)
    sample = load_submission_sample()
    samples = [Sample(row.id) for row in sample.itertuples(index=False)]
    ds = RedCharDataset(samples, config.TEST_IMAGES, is_test=True, cache_in_ram=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    prim = [load_model(c, device) for c in args.checkpoints]
    glyph = [load_glyph_model(c, device) for c in args.glyph_checkpoints]

    records = []  # (score, filename, pred_string, detail)
    for images, filenames in loader:
        images = images.to(device)
        pc = kp = None
        for shift in (0, -4, 4):
            sh = images if shift == 0 else VF.affine(images, angle=0, translate=[shift, 0], scale=1.0,
                shear=[0., 0.], interpolation=InterpolationMode.BILINEAR, fill=[1., 1., 1.])
            for m in prim:
                cl, kl = m(sh)
                pc = F.softmax(cl, -1) if pc is None else pc + F.softmax(cl, -1)
                kp = F.softmax(kl, -1) if kp is None else kp + F.softmax(kl, -1)
        pc /= len(prim) * 3; kp /= len(prim) * 3
        gp = glyph_probabilities(glyph, images)
        char_pred = selective_rerank(pc, gp, args.top_k, args.primary_margin_max, args.glyph_margin_min)
        red_prob = kp[..., config.RED_INDEX]
        color_pred = red_prob.ge(args.red_threshold).long()
        char_conf = pc.max(-1).values            # [B,5]
        # OUTPUT-relevant confidence per position:
        #  - colour decision margin from the red threshold (borderline = risky)
        #  - char confidence ONLY counts when the position is output as red
        d_color = (red_prob - args.red_threshold).abs().clamp(max=0.34) * 3  # ~0..1
        is_red = red_prob >= args.red_threshold
        pos_conf = torch.where(is_red, torch.minimum(char_conf, d_color), d_color)  # [B,5]
        img_score = pos_conf.min(-1).values
        for i in range(images.size(0)):
            pred = decode_prediction(char_pred[i].tolist(), color_pred[i].tolist())
            worst = int(pos_conf[i].argmin())
            det = (f"p{worst} {'RED' if bool(is_red[i,worst]) else 'nonred'} "
                   f"charconf={char_conf[i,worst]:.2f} P(red)={red_prob[i,worst]:.2f}")
            records.append((float(img_score[i]), filenames[i], pred, det))

    records.sort(key=lambda r: r[0])
    top = records[:args.n]
    if args.outdir.exists():
        shutil.rmtree(args.outdir)
    args.outdir.mkdir(parents=True)
    print(f"{args.n} most-uncertain test images (score = lowest per-position confidence):")
    for rank, (score, fn, pred, det) in enumerate(top, 1):
        src = config.TEST_IMAGES / fn
        tag = f"{rank:02d}_score{score:.2f}_{fn[:-4]}_pred-{pred or 'EMPTY'}"
        shutil.copy(src, args.outdir / f"{tag}.png")
        Image.open(src).convert("RGB").resize((1000, 300), Image.LANCZOS).save(args.outdir / f"{tag}_5x.png")
        print(f"  {rank:02d} {fn} score={score:.3f} pred='{pred}' [{det}]")
    print("saved ->", args.outdir.resolve())


if __name__ == "__main__":
    main()
