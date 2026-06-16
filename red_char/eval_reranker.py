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
from glyph import glyph_probabilities, load_glyph_model
from predict import load_model


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
    """Use the local model with the largest candidate margin for each position."""
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
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--glyph-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--use-glyph-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--x-tta", action="store_true", help="average original and +/-4px horizontal shifts")
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    base = build_train_dataset(cache_in_ram=False)
    _, val_indices = deterministic_split_indices(len(base))
    loader = DataLoader(
        Subset(base, val_indices),
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    primary_models = [load_model(path, device) for path in args.checkpoints]
    glyph_models = [
        load_glyph_model(path, device, use_ema=args.use_glyph_ema)
        for path in args.glyph_checkpoints
    ]

    primary_probs = []
    color_probs = []
    glyph_probs = []
    glyph_expert_probs = []
    char_targets = []
    color_targets = []
    for images, char_target, color_target in loader:
        images = images.to(device, non_blocking=True)
        primary = None
        color = None
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
                primary = current_char if primary is None else primary + current_char
                color = current_color if color is None else color + current_color
        divisor = len(primary_models) * len(shifts)
        primary_probs.append((primary / divisor).cpu())
        color_probs.append((color / divisor).cpu())
        expert_prob = torch.stack(
            [glyph_probabilities([model], images) for model in glyph_models], dim=0
        )
        glyph_expert_probs.append(expert_prob.cpu())
        glyph_probs.append(expert_prob.mean(dim=0).cpu())
        char_targets.append(char_target)
        color_targets.append(color_target)

    primary_prob = torch.cat(primary_probs)
    color_prob = torch.cat(color_probs)
    glyph_prob = torch.cat(glyph_probs)
    glyph_expert_prob = torch.cat(glyph_expert_probs, dim=1)
    char_target = torch.cat(char_targets)
    color_target = torch.cat(color_targets)
    red_mask = color_target.bool()
    true_strings = [
        decode_prediction(char_target[i], color_target[i]) for i in range(len(char_target))
    ]

    for top_k in args.top_k:
        best = (-1, None, None, None)
        best_true_color = (-1, None, None)
        for alpha_i in range(0, 41):
            alpha = alpha_i / 20
            char_pred = rerank(primary_prob, glyph_prob, top_k, alpha)
            true_color_exact = sum(
                decode_prediction(char_pred[i], color_target[i]) == true_strings[i]
                for i in range(len(char_target))
            )
            red_acc = char_pred.eq(char_target)[red_mask].float().mean().item()
            if true_color_exact > best_true_color[0]:
                best_true_color = (true_color_exact, alpha, red_acc)
            for threshold in (0.20, 0.30, 0.40, 0.50):
                color_pred = color_prob[..., config.RED_INDEX].ge(threshold).long()
                exact = sum(
                    decode_prediction(char_pred[i], color_pred[i]) == true_strings[i]
                    for i in range(len(char_target))
                )
                if exact > best[0]:
                    best = (exact, alpha, threshold, red_acc)
        print(
            f"top_k={top_k} exact={best[0]}/{len(char_target)}={best[0]/len(char_target):.5f} "
            f"alpha={best[1]:.2f} red_threshold={best[2]:.2f} red_char_acc={best[3]:.6f}; "
            f"true-color exact={best_true_color[0]}/{len(char_target)}="
            f"{best_true_color[0]/len(char_target):.5f} alpha={best_true_color[1]:.2f}"
        )

        selective_best = (-1, None, None, None)
        for primary_margin_i in range(1, 20):
            primary_margin_max = primary_margin_i / 20
            for glyph_margin_i in range(0, 20):
                glyph_margin_min = glyph_margin_i / 20
                char_pred = selective_rerank(
                    primary_prob, glyph_prob, top_k, primary_margin_max, glyph_margin_min
                )
                red_correct = char_pred.eq(char_target)[red_mask].sum().item()
                if red_correct > selective_best[0]:
                    selective_best = (
                        red_correct,
                        primary_margin_max,
                        glyph_margin_min,
                        char_pred,
                    )
        char_pred = selective_best[3]
        exact_best = (-1, None)
        for threshold in (0.20, 0.30, 0.40, 0.50):
            color_pred = color_prob[..., config.RED_INDEX].ge(threshold).long()
            exact = sum(
                decode_prediction(char_pred[i], color_pred[i]) == true_strings[i]
                for i in range(len(char_target))
            )
            if exact > exact_best[0]:
                exact_best = (exact, threshold)
        print(
            f"top_k={top_k} selective exact={exact_best[0]}/{len(char_target)}="
            f"{exact_best[0]/len(char_target):.5f} primary_margin<={selective_best[1]:.2f} "
            f"glyph_margin>={selective_best[2]:.2f} red_threshold={exact_best[1]:.2f} "
            f"red_char_acc={selective_best[0]/red_mask.sum().item():.6f}"
        )

        expert_best = (-1, None, None, None)
        for primary_margin_i in range(1, 20):
            primary_margin_max = primary_margin_i / 20
            for glyph_margin_i in range(0, 20):
                glyph_margin_min = glyph_margin_i / 20
                char_pred = expert_selective_rerank(
                    primary_prob,
                    glyph_expert_prob,
                    top_k,
                    primary_margin_max,
                    glyph_margin_min,
                )
                red_correct = char_pred.eq(char_target)[red_mask].sum().item()
                if red_correct > expert_best[0]:
                    expert_best = (
                        red_correct,
                        primary_margin_max,
                        glyph_margin_min,
                        char_pred,
                    )
        char_pred = expert_best[3]
        exact_best = (-1, None)
        for threshold in (0.20, 0.30, 0.40, 0.50):
            color_pred = color_prob[..., config.RED_INDEX].ge(threshold).long()
            exact = sum(
                decode_prediction(char_pred[i], color_pred[i]) == true_strings[i]
                for i in range(len(char_target))
            )
            if exact > exact_best[0]:
                exact_best = (exact, threshold)
        print(
            f"top_k={top_k} expert-selective exact={exact_best[0]}/{len(char_target)}="
            f"{exact_best[0]/len(char_target):.5f} primary_margin<={expert_best[1]:.2f} "
            f"glyph_margin>={expert_best[2]:.2f} red_threshold={exact_best[1]:.2f} "
            f"red_char_acc={expert_best[0]/red_mask.sum().item():.6f}"
        )


if __name__ == "__main__":
    main()
