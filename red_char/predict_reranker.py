from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import RedCharDataset, Sample, decode_prediction, load_submission_sample
from ensemble import average_model_logits, load_models
from eval_reranker import rerank, selective_rerank
from glyph import glyph_probabilities, load_glyph_model
from predict import validate_submission, write_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--glyph-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--char-weights", type=float, nargs="+")
    parser.add_argument("--color-weights", type=float, nargs="+")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--selective", action="store_true")
    parser.add_argument("--primary-margin-max", type=float, default=0.20)
    parser.add_argument("--glyph-margin-min", type=float, default=0.0)
    parser.add_argument("--red-threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "submission_reranker.csv")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return parser


@torch.no_grad()
def predict_labels(
    primary_models: list[torch.nn.Module],
    glyph_models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    char_weights: list[float] | None = None,
    color_weights: list[float] | None = None,
    top_k: int = 3,
    alpha: float = 0.25,
    selective: bool = False,
    primary_margin_max: float = 0.20,
    glyph_margin_min: float = 0.0,
    red_threshold: float | None = None,
) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    labels: list[str] = []
    for images, filenames in tqdm(loader, desc="predict-reranker"):
        images = images.to(device, non_blocking=True)
        char_logits, color_logits = average_model_logits(
            primary_models,
            images,
            char_weights=char_weights,
            color_weights=color_weights,
        )
        primary_prob = F.softmax(char_logits, dim=-1)
        glyph_prob = glyph_probabilities(glyph_models, images)
        if selective:
            char_pred = selective_rerank(
                primary_prob,
                glyph_prob,
                top_k=top_k,
                primary_margin_max=primary_margin_max,
                glyph_margin_min=glyph_margin_min,
            )
        else:
            char_pred = rerank(primary_prob, glyph_prob, top_k=top_k, alpha=alpha)
        if red_threshold is None:
            color_pred = color_logits.argmax(dim=-1)
        else:
            color_pred = F.softmax(color_logits, dim=-1)[..., config.RED_INDEX].ge(red_threshold).long()
        for filename, chars, colors in zip(filenames, char_pred.cpu(), color_pred.cpu()):
            ids.append(filename)
            labels.append(decode_prediction(chars, colors))
    return ids, labels


def main() -> None:
    args = build_parser().parse_args()
    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    sample = load_submission_sample()
    samples = [Sample(row.id) for row in sample.itertuples(index=False)]
    dataset = RedCharDataset(samples, config.TEST_IMAGES, is_test=True, cache_in_ram=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    primary_models, _ = load_models(args.checkpoints, device)
    glyph_models = [load_glyph_model(path, device) for path in args.glyph_checkpoints]
    ids, labels = predict_labels(
        primary_models,
        glyph_models,
        loader,
        device,
        char_weights=args.char_weights,
        color_weights=args.color_weights,
        top_k=args.top_k,
        alpha=args.alpha,
        selective=args.selective,
        primary_margin_max=args.primary_margin_max,
        glyph_margin_min=args.glyph_margin_min,
        red_threshold=args.red_threshold,
    )
    write_submission(ids, labels, args.output)
    validate_submission(args.output)
    print("submission written:", args.output)


if __name__ == "__main__":
    main()
