from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF

import config
from dataset import RedCharDataset, Sample, decode_prediction, load_submission_sample
from eval_reranker import rerank, selective_rerank
from glyph import glyph_probabilities, load_glyph_model
from predict import load_model, validate_submission, write_submission


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--glyph-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--selective", action="store_true")
    parser.add_argument("--primary-margin-max", type=float, default=0.20)
    parser.add_argument("--glyph-margin-min", type=float, default=0.0)
    parser.add_argument("--red-threshold", type=float, default=0.20)
    parser.add_argument("--x-tta", action="store_true")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "submission_reranker.csv")
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    sample = load_submission_sample()
    samples = [Sample(row.id) for row in sample.itertuples(index=False)]
    dataset = RedCharDataset(samples, config.TEST_IMAGES, is_test=True, cache_in_ram=False)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    primary_models = [load_model(path, device) for path in args.checkpoints]
    glyph_models = [load_glyph_model(path, device) for path in args.glyph_checkpoints]

    ids = []
    labels = []
    for images, filenames in tqdm(loader, desc="predict-reranker"):
        images = images.to(device, non_blocking=True)
        char_prob = color_prob = None
        shifts = (0, -4, 4) if args.x_tta else (0,)
        for shift in shifts:
            shifted = (
                images
                if shift == 0
                else VF.affine(
                    images,
                    angle=0,
                    translate=[shift, 0],
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=[1.0, 1.0, 1.0],
                )
            )
            for model in primary_models:
                char_logits, color_logits = model(shifted)
                current_char = F.softmax(char_logits, dim=-1)
                current_color = F.softmax(color_logits, dim=-1)
                char_prob = current_char if char_prob is None else char_prob + current_char
                color_prob = current_color if color_prob is None else color_prob + current_color
        divisor = len(primary_models) * len(shifts)
        char_prob /= divisor
        color_prob /= divisor
        glyph_prob = glyph_probabilities(glyph_models, images)
        if args.selective:
            char_pred = selective_rerank(
                char_prob,
                glyph_prob,
                args.top_k,
                args.primary_margin_max,
                args.glyph_margin_min,
            ).cpu()
        else:
            char_pred = rerank(char_prob, glyph_prob, args.top_k, args.alpha).cpu()
        color_pred = color_prob[..., config.RED_INDEX].ge(args.red_threshold).long().cpu()
        for filename, chars, colors in zip(filenames, char_pred, color_pred):
            ids.append(filename)
            labels.append(decode_prediction(chars, colors))

    write_submission(ids, labels, args.output)
    validate_submission(args.output)
    print("submission written:", args.output)


if __name__ == "__main__":
    main()
