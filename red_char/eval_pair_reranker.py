from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF

import config
from dataset import build_train_dataset, decode_prediction, deterministic_split_indices
from glyph import PAIR_GROUPS, extract_glyph_crops, load_pair_glyph_model
from predict import load_model


def pair_rerank(
    primary_prob: torch.Tensor,
    pair_scores: torch.Tensor,
    top_k: int,
    primary_margin_max: float,
    pair_margin_min: float,
) -> torch.Tensor:
    primary_value, candidate_idx = primary_prob.topk(top_k, dim=-1)
    expanded_idx = candidate_idx.unsqueeze(0).expand(pair_scores.shape[0], -1, -1, -1)
    candidate_pair = pair_scores.gather(-1, expanded_idx)
    pair_value, pair_order = candidate_pair.topk(2, dim=-1)
    valid = pair_value[..., 1].isfinite()
    pair_margin = pair_value[..., 0] - pair_value[..., 1]
    pair_margin = torch.where(valid, pair_margin, torch.full_like(pair_margin, -1.0))
    best_group = pair_margin.argmax(dim=0, keepdim=True)
    best_order = pair_order[..., 0].gather(0, best_group).squeeze(0)
    pair_winner = candidate_idx.gather(-1, best_order.unsqueeze(-1)).squeeze(-1)
    best_margin = pair_margin.gather(0, best_group).squeeze(0)
    primary_winner = candidate_idx[..., 0]
    primary_margin = primary_value[..., 0] - primary_value[..., 1]
    override = (primary_margin <= primary_margin_max) & (best_margin >= pair_margin_min)
    return torch.where(override, pair_winner, primary_winner)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--pair-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--x-tta", action="store_true")
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    base = build_train_dataset(cache_in_ram=False)
    _, val_indices = deterministic_split_indices(len(base))
    loader = DataLoader(Subset(base, val_indices), batch_size=128, num_workers=4, pin_memory=True)
    primary_models = [load_model(path, device) for path in args.checkpoints]
    pair_models = [load_pair_glyph_model(path, device) for path in args.pair_checkpoints]

    primary_probs = []
    color_probs = []
    pair_batches = []
    char_targets = []
    color_targets = []
    for images, char_target, color_target in loader:
        images = images.to(device, non_blocking=True)
        char_prob = color_prob = None
        shifts = (0, -4, 4) if args.x_tta else (0,)
        for shift in shifts:
            shifted = (
                images
                if shift == 0
                else VF.affine(
                    images, 0, [shift, 0], 1.0, [0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR, fill=[1.0, 1.0, 1.0],
                )
            )
            for model in primary_models:
                char_logits, color_logits = model(shifted)
                char_prob = F.softmax(char_logits, -1) if char_prob is None else char_prob + F.softmax(char_logits, -1)
                color_prob = F.softmax(color_logits, -1) if color_prob is None else color_prob + F.softmax(color_logits, -1)
        divisor = len(primary_models) * len(shifts)
        primary_probs.append((char_prob / divisor).cpu())
        color_probs.append((color_prob / divisor).cpu())

        crops = extract_glyph_crops(images).flatten(0, 1)
        group_logits = None
        for model in pair_models:
            _, current = model(crops)
            if group_logits is None:
                group_logits = current
            else:
                group_logits = [a + b for a, b in zip(group_logits, current)]
        batch_group_scores = []
        for chars, logits in zip(PAIR_GROUPS, group_logits):
            global_scores = torch.full(
                (len(crops), config.NUM_CHARS), -torch.inf, device=device
            )
            indices = [config.CHAR_TO_IDX[ch] for ch in chars]
            global_scores[:, indices] = logits / len(pair_models)
            batch_group_scores.append(
                global_scores.view(len(images), config.NUM_POSITIONS, config.NUM_CHARS)
            )
        pair_batches.append(torch.stack(batch_group_scores).cpu())
        char_targets.append(char_target)
        color_targets.append(color_target)

    primary_prob = torch.cat(primary_probs)
    color_prob = torch.cat(color_probs)
    pair_scores = torch.cat(pair_batches, dim=1)
    char_target = torch.cat(char_targets)
    color_target = torch.cat(color_targets)
    red_mask = color_target.bool()
    true_strings = [decode_prediction(char_target[i], color_target[i]) for i in range(len(char_target))]

    for top_k in args.top_k:
        best = (-1, None, None, None)
        for primary_i in range(1, 20):
            primary_margin = primary_i / 20
            for pair_i in range(0, 41):
                pair_margin = pair_i / 10
                char_pred = pair_rerank(primary_prob, pair_scores, top_k, primary_margin, pair_margin)
                red_correct = char_pred.eq(char_target)[red_mask].sum().item()
                if red_correct > best[0]:
                    best = (red_correct, primary_margin, pair_margin, char_pred)
        exact_best = (-1, None)
        for threshold in (0.20, 0.30, 0.40, 0.50):
            color_pred = color_prob[..., config.RED_INDEX].ge(threshold).long()
            exact = sum(
                decode_prediction(best[3][i], color_pred[i]) == true_strings[i]
                for i in range(len(char_target))
            )
            if exact > exact_best[0]:
                exact_best = (exact, threshold)
        print(
            f"top_k={top_k} pair exact={exact_best[0]}/{len(char_target)}={exact_best[0]/len(char_target):.5f} "
            f"primary_margin<={best[1]:.2f} pair_margin>={best[2]:.2f} "
            f"red_threshold={exact_best[1]:.2f} red_char_acc={best[0]/red_mask.sum().item():.6f}"
        )


if __name__ == "__main__":
    main()
