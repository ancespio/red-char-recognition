from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import config
from dataset import build_train_dataset, decode_prediction, deterministic_split_indices
from ensemble import average_model_logits, load_models
from glyph import glyph_probabilities, load_glyph_model
from tta import translate_images


X_TTA_DX = (0, -4, 4)


def rerank(primary_prob: torch.Tensor, glyph_prob: torch.Tensor, top_k: int, alpha: float) -> torch.Tensor:
    candidate_prob, candidate_idx = primary_prob.topk(top_k, dim=-1)
    local_prob = glyph_prob.gather(-1, candidate_idx)
    scores = candidate_prob.clamp_min(1e-8).log() + alpha * local_prob.clamp_min(1e-8).log()
    return candidate_idx.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)


def selective_rerank(
    primary_prob: torch.Tensor,
    glyph_prob: torch.Tensor,
    top_k: int,
    primary_margin_max: float,
    glyph_margin_min: float,
) -> torch.Tensor:
    primary_value, candidate_idx = primary_prob.topk(top_k, dim=-1)
    candidate_glyph = glyph_prob.gather(-1, candidate_idx)
    glyph_value, glyph_order = candidate_glyph.topk(2, dim=-1)
    glyph_winner = candidate_idx.gather(-1, glyph_order[..., :1]).squeeze(-1)
    primary_winner = candidate_idx[..., 0]
    primary_margin = primary_value[..., 0] - primary_value[..., 1]
    glyph_margin = glyph_value[..., 0] - glyph_value[..., 1]
    override = (primary_margin <= primary_margin_max) & (glyph_margin >= glyph_margin_min)
    return torch.where(override, glyph_winner, primary_winner)


def expert_selective_rerank(
    primary_prob: torch.Tensor,
    expert_glyph_prob: torch.Tensor,
    top_k: int,
    primary_margin_max: float,
    glyph_margin_min: float,
) -> torch.Tensor:
    primary_value, candidate_idx = primary_prob.topk(top_k, dim=-1)
    expanded_idx = candidate_idx.unsqueeze(0).expand(expert_glyph_prob.shape[0], -1, -1, -1)
    candidate_glyph = expert_glyph_prob.gather(-1, expanded_idx)
    glyph_value, glyph_order = candidate_glyph.topk(2, dim=-1)
    expert_margin = glyph_value[..., 0] - glyph_value[..., 1]
    best_expert = expert_margin.argmax(dim=0, keepdim=True)
    best_order = glyph_order[..., 0].gather(0, best_expert).squeeze(0)
    glyph_winner = candidate_idx.gather(-1, best_order.unsqueeze(-1)).squeeze(-1)
    glyph_margin = expert_margin.gather(0, best_expert).squeeze(0)
    primary_winner = candidate_idx[..., 0]
    primary_margin = primary_value[..., 0] - primary_value[..., 1]
    override = (primary_margin <= primary_margin_max) & (glyph_margin >= glyph_margin_min)
    return torch.where(override, glyph_winner, primary_winner)


@torch.no_grad()
def average_primary_logits(
    primary_models: list[torch.nn.Module],
    images: torch.Tensor,
    char_weights: list[float] | None = None,
    color_weights: list[float] | None = None,
    x_tta: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    views = (
        tuple(images if dx == 0 else translate_images(images, dx=dx, dy=0) for dx in X_TTA_DX)
        if x_tta
        else (images,)
    )
    char_sum = None
    color_sum = None
    for view in views:
        char_logits, color_logits = average_model_logits(
            primary_models,
            view,
            char_weights=char_weights,
            color_weights=color_weights,
        )
        char_sum = char_logits if char_sum is None else char_sum + char_logits
        color_sum = color_logits if color_sum is None else color_sum + color_logits
    return char_sum / len(views), color_sum / len(views)


@torch.no_grad()
def collect_probabilities(
    checkpoints: list[Path],
    glyph_checkpoints: list[Path],
    device: torch.device,
    char_weights: list[float] | None = None,
    color_weights: list[float] | None = None,
    batch_size: int = config.BATCH_SIZE,
    x_tta: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = build_train_dataset(cache_in_ram=False)
    _, val_indices = deterministic_split_indices(len(dataset), seed=config.SPLIT_SEED)
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    primary_models, _ = load_models(checkpoints, device)
    glyph_models = [load_glyph_model(path, device) for path in glyph_checkpoints]
    primary_probs = []
    color_probs = []
    glyph_probs = []
    char_targets = []
    color_targets = []
    for images, char_target, color_target in loader:
        images = images.to(device, non_blocking=True)
        char_logits, color_logits = average_primary_logits(
            primary_models,
            images,
            char_weights=char_weights,
            color_weights=color_weights,
            x_tta=x_tta,
        )
        primary_probs.append(F.softmax(char_logits, dim=-1).cpu())
        color_probs.append(F.softmax(color_logits, dim=-1).cpu())
        glyph_probs.append(glyph_probabilities(glyph_models, images).cpu())
        char_targets.append(char_target)
        color_targets.append(color_target)
    return (
        torch.cat(primary_probs),
        torch.cat(color_probs),
        torch.cat(glyph_probs),
        torch.cat(char_targets),
        torch.cat(color_targets),
    )


def exact_count(char_pred: torch.Tensor, color_pred: torch.Tensor, char_target: torch.Tensor, color_target: torch.Tensor) -> int:
    true_strings = [decode_prediction(char_target[i], color_target[i]) for i in range(len(char_target))]
    return sum(decode_prediction(char_pred[i], color_pred[i]) == true_strings[i] for i in range(len(char_target)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--glyph-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--char-weights", type=float, nargs="+")
    parser.add_argument("--color-weights", type=float, nargs="+")
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--x-tta", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(config.DEVICE)
    primary_prob, color_prob, glyph_prob, char_target, color_target = collect_probabilities(
        args.checkpoints,
        args.glyph_checkpoints,
        device,
        char_weights=args.char_weights,
        color_weights=args.color_weights,
        x_tta=args.x_tta,
    )
    true_color = color_target
    for top_k in args.top_k:
        best = (-1, None)
        for alpha_i in range(int(args.alpha_max * 20) + 1):
            alpha = alpha_i / 20
            char_pred = rerank(primary_prob, glyph_prob, top_k, alpha)
            exact = exact_count(char_pred, true_color, char_target, color_target)
            if exact > best[0]:
                best = (exact, alpha)
        print(f"top_k={top_k} true-color exact={best[0]}/{len(char_target)}={best[0]/len(char_target):.5f} alpha={best[1]:.2f}")


if __name__ == "__main__":
    main()
