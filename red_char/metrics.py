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


def compute_loss(
    char_logits: torch.Tensor,
    color_logits: torch.Tensor,
    char_targets: torch.Tensor,
    color_targets: torch.Tensor,
    color_weight: float = config.COLOR_LOSS_WEIGHT,
    red_char_weight: float = 1.0,
) -> torch.Tensor:
    if red_char_weight <= 0:
        raise ValueError("red_char_weight must be positive")
    char_losses = torch.nn.functional.cross_entropy(
        char_logits.reshape(-1, config.NUM_CHARS),
        char_targets.reshape(-1),
        reduction="none",
    ).view_as(char_targets)
    if red_char_weight == 1.0:
        char_loss = char_losses.mean()
    else:
        char_weights = torch.ones_like(char_losses)
        char_weights = torch.where(
            color_targets.eq(config.RED_INDEX),
            char_weights * red_char_weight,
            char_weights,
        )
        char_loss = (char_losses * char_weights).sum() / char_weights.sum()
    color_loss = torch.nn.functional.cross_entropy(
        color_logits.reshape(-1, config.NUM_COLORS),
        color_targets.reshape(-1),
    )
    return char_loss + color_weight * color_loss


@torch.no_grad()
def red_string_exact(char_pred: torch.Tensor, color_pred: torch.Tensor, char_targets: torch.Tensor, color_targets: torch.Tensor) -> float:
    pred_codes = encode_red_sequences(char_pred, color_pred)
    target_codes = encode_red_sequences(char_targets, color_targets)
    return pred_codes.eq(target_codes).float().mean().item()


def encode_red_sequences(char_indices: torch.Tensor, color_indices: torch.Tensor) -> torch.Tensor:
    if char_indices.shape != color_indices.shape:
        raise ValueError("character and color tensors must have matching shapes")
    code = torch.zeros(char_indices.shape[:-1], dtype=torch.int64, device=char_indices.device)
    for position in range(char_indices.size(-1)):
        is_red = color_indices[..., position].eq(config.RED_INDEX)
        token = char_indices[..., position].to(torch.int64) + 1
        code = torch.where(is_red, code * (config.NUM_CHARS + 1) + token, code)
    return code
