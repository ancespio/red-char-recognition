from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from dataset import TransformSubset, build_train_dataset
from ensemble import load_models
from eval_reranker import average_primary_logits
from glyph import glyph_probabilities, load_glyph_model
from kfold import N_FOLDS, fold_split


def format_fold_paths(patterns: list[str], fold: int) -> list[Path]:
    return [Path(pattern.format(fold=fold)) for pattern in patterns]


def paths_for_fold(paths: list[Path] | None, patterns: list[str], fold: int, n_folds: int, label: str) -> list[Path]:
    if paths is None:
        return format_fold_paths(patterns, fold)
    if len(paths) != n_folds:
        raise ValueError(f"{label} path count must match n_folds")
    return [paths[fold]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-patterns", nargs="+", default=[])
    parser.add_argument("--primary-paths", type=Path, nargs="+")
    parser.add_argument("--glyph-patterns", nargs="*", default=[])
    parser.add_argument("--glyph-paths", type=Path, nargs="+")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--x-tta", action="store_true")
    parser.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "oof" / "oof.pt")
    return parser


def _check_paths(paths: list[Path], fold: int) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing fold {fold} checkpoint(s): {missing}")


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(config.DEVICE)
    dataset = build_train_dataset(cache_in_ram=False)
    n_items = len(dataset)

    primary_prob = torch.zeros(n_items, config.NUM_POSITIONS, config.NUM_CHARS)
    color_prob = torch.zeros(n_items, config.NUM_POSITIONS, config.NUM_COLORS)
    glyph_prob = torch.zeros(n_items, config.NUM_POSITIONS, config.NUM_CHARS) if args.glyph_patterns else None
    char_target = torch.zeros(n_items, config.NUM_POSITIONS, dtype=torch.long)
    color_target = torch.zeros(n_items, config.NUM_POSITIONS, dtype=torch.long)
    filled = torch.zeros(n_items, dtype=torch.bool)

    for fold in range(args.n_folds):
        _, val_indices = fold_split(n_items, fold=fold, n_folds=args.n_folds)
        primary_paths = paths_for_fold(args.primary_paths, args.primary_patterns, fold, args.n_folds, "primary")
        glyph_paths = paths_for_fold(args.glyph_paths, args.glyph_patterns, fold, args.n_folds, "glyph")
        _check_paths(primary_paths + glyph_paths, fold)

        primary_models, _ = load_models(primary_paths, device)
        glyph_models = [load_glyph_model(path, device) for path in glyph_paths]
        loader = DataLoader(
            TransformSubset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        cursor = 0
        for images, chars, colors in loader:
            batch_size = images.size(0)
            images = images.to(device, non_blocking=True)
            char_logits, color_logits = average_primary_logits(primary_models, images, x_tta=args.x_tta)
            batch_indices = torch.tensor(val_indices[cursor : cursor + batch_size], dtype=torch.long)
            primary_prob[batch_indices] = F.softmax(char_logits, dim=-1).cpu()
            color_prob[batch_indices] = F.softmax(color_logits, dim=-1).cpu()
            if glyph_prob is not None:
                glyph_prob[batch_indices] = glyph_probabilities(glyph_models, images).cpu()
            char_target[batch_indices] = chars
            color_target[batch_indices] = colors
            filled[batch_indices] = True
            cursor += batch_size

        print(f"fold {fold}: filled {len(val_indices)} OOF predictions")
        del primary_models
        del glyph_models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not bool(filled.all()):
        raise RuntimeError(f"unfilled OOF rows: {int((~filled).sum())}")

    payload = {
        "primary_char": primary_prob,
        "color": color_prob,
        "char_target": char_target,
        "color_target": color_target,
        "primary_patterns": args.primary_patterns,
        "glyph_patterns": args.glyph_patterns,
        "n_folds": args.n_folds,
        "x_tta": args.x_tta,
    }
    if glyph_prob is not None:
        payload["glyph"] = glyph_prob
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"saved OOF predictions for {n_items} rows -> {args.out}")


if __name__ == "__main__":
    main()
