from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import config
from dataset import build_train_dataset, decode_prediction, deterministic_split_indices
from glyph import GlyphDataset, extract_glyph_crops, load_glyph_model
from predict import load_model


@torch.no_grad()
def extract_reference_features(model, dataset, device) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    features = []
    targets = []
    for crops, target in loader:
        feature = F.normalize(model.features(crops.to(device, non_blocking=True)), dim=-1)
        features.append(feature.half().cpu())
        targets.append(target)
    return torch.cat(features), torch.cat(targets)


@torch.no_grad()
def class_nearest_similarity(
    model,
    references: torch.Tensor,
    reference_targets: torch.Tensor,
    images: torch.Tensor,
) -> torch.Tensor:
    crops = extract_glyph_crops(images).flatten(0, 1)
    queries = F.normalize(model.features(crops), dim=-1).half()
    result = torch.empty(len(queries), config.NUM_CHARS, device=queries.device)
    references = references.to(queries.device)
    reference_targets = reference_targets.to(queries.device)
    for char_idx in range(config.NUM_CHARS):
        class_reference = references[reference_targets == char_idx]
        result[:, char_idx] = (queries @ class_reference.T).amax(dim=-1)
    return result.view(images.shape[0], config.NUM_POSITIONS, config.NUM_CHARS)


def rerank(primary_prob: torch.Tensor, similarity: torch.Tensor, top_k: int, beta: float) -> torch.Tensor:
    candidate_prob, candidate_idx = primary_prob.topk(top_k, dim=-1)
    candidate_similarity = similarity.gather(-1, candidate_idx)
    score = candidate_prob.clamp_min(1e-8).log() + beta * candidate_similarity
    return candidate_idx.gather(-1, score.argmax(-1, keepdim=True)).squeeze(-1)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--glyph-checkpoint", type=Path, required=True)
    parser.add_argument("--use-glyph-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 3, 5])
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    base = build_train_dataset(cache_in_ram=True)
    train_indices, val_indices = deterministic_split_indices(len(base))
    glyph_model = load_glyph_model(args.glyph_checkpoint, device, use_ema=args.use_glyph_ema)
    print("extracting red-glyph reference features")
    references, reference_targets = extract_reference_features(
        glyph_model, GlyphDataset(base, train_indices, red_only=True, augment=False), device
    )
    print("reference features:", tuple(references.shape))

    primary_models = [load_model(path, device) for path in args.checkpoints]
    loader = DataLoader(
        Subset(base, val_indices), batch_size=128, shuffle=False, num_workers=4, pin_memory=True
    )
    primary_probs = []
    color_probs = []
    similarities = []
    char_targets = []
    color_targets = []
    for images, char_target, color_target in loader:
        images = images.to(device, non_blocking=True)
        char_prob = color_prob = None
        for model in primary_models:
            char_logits, color_logits = model(images)
            current_char = char_logits.softmax(-1)
            current_color = color_logits.softmax(-1)
            char_prob = current_char if char_prob is None else char_prob + current_char
            color_prob = current_color if color_prob is None else color_prob + current_color
        primary_probs.append((char_prob / len(primary_models)).cpu())
        color_probs.append((color_prob / len(primary_models)).cpu())
        similarities.append(
            class_nearest_similarity(
                glyph_model, references, reference_targets, images
            ).float().cpu()
        )
        char_targets.append(char_target)
        color_targets.append(color_target)

    primary_prob = torch.cat(primary_probs)
    color_prob = torch.cat(color_probs)
    similarity = torch.cat(similarities)
    char_target = torch.cat(char_targets)
    color_target = torch.cat(color_targets)
    red_mask = color_target.bool()
    true_strings = [
        decode_prediction(char_target[i], color_target[i]) for i in range(len(char_target))
    ]
    print(
        "nearest-class red-char accuracy:",
        similarity.argmax(-1).eq(char_target)[red_mask].float().mean().item(),
    )

    for top_k in args.top_k:
        best = (-1, None, None, None)
        for beta_i in range(0, 101):
            beta = beta_i / 2
            char_pred = rerank(primary_prob, similarity, top_k, beta)
            red_acc = char_pred.eq(char_target)[red_mask].float().mean().item()
            for threshold in (0.20, 0.30, 0.40, 0.50):
                color_pred = color_prob[..., config.RED_INDEX].ge(threshold).long()
                exact = sum(
                    decode_prediction(char_pred[i], color_pred[i]) == true_strings[i]
                    for i in range(len(char_target))
                )
                if exact > best[0]:
                    best = (exact, beta, threshold, red_acc)
        print(
            f"top_k={top_k} exact={best[0]}/{len(char_target)}={best[0]/len(char_target):.5f} "
            f"beta={best[1]:.2f} red_threshold={best[2]:.2f} red_char_acc={best[3]:.6f}"
        )


if __name__ == "__main__":
    main()
