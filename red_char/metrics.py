from __future__ import annotations

import torch

import config


@torch.no_grad()
def batch_metrics(char_logits: torch.Tensor, color_logits: torch.Tensor, char_targets: torch.Tensor, color_targets: torch.Tensor) -> dict[str, float]:
    char_pred = char_logits.argmax(dim=-1)
    color_pred = color_logits.argmax(dim=-1)
    char_correct = char_pred.eq(char_targets)
    color_correct = color_pred.eq(color_targets)
    position_joint = char_correct & color_correct
    exact = red_string_exact(char_pred, color_pred, char_targets, color_targets)
    return {
        "exact": exact,
        "char_acc": char_correct.float().mean().item(),
        "color_acc": color_correct.float().mean().item(),
        "joint_pos_acc": position_joint.float().mean().item(),
    }


def compute_loss(char_logits: torch.Tensor, color_logits: torch.Tensor, char_targets: torch.Tensor, color_targets: torch.Tensor, color_weight: float = config.COLOR_LOSS_WEIGHT) -> torch.Tensor:
    char_loss = torch.nn.functional.cross_entropy(
        char_logits.reshape(-1, config.NUM_CHARS),
        char_targets.reshape(-1),
    )
    color_loss = torch.nn.functional.cross_entropy(
        color_logits.reshape(-1, config.NUM_COLORS),
        color_targets.reshape(-1),
    )
    return char_loss + color_weight * color_loss


@torch.no_grad()
def red_string_exact(char_pred: torch.Tensor, color_pred: torch.Tensor, char_targets: torch.Tensor, color_targets: torch.Tensor) -> float:
    correct = 0
    batch_size = char_pred.size(0)
    for pred_chars, pred_colors, target_chars, target_colors in zip(char_pred.cpu(), color_pred.cpu(), char_targets.cpu(), color_targets.cpu()):
        pred = tuple(int(ch) for ch, color in zip(pred_chars.tolist(), pred_colors.tolist()) if color == config.RED_INDEX)
        target = tuple(int(ch) for ch, color in zip(target_chars.tolist(), target_colors.tolist()) if color == config.RED_INDEX)
        correct += int(pred == target)
    return correct / max(batch_size, 1)
