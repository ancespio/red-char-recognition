from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config
from dataset import build_train_dataset, deterministic_split_indices, filename_hash, seed_everything
from ensemble import load_models
from metrics import encode_red_sequences


def generate_weight_vectors(model_count: int, units: int) -> list[tuple[float, ...]]:
    vectors = []
    for bars in itertools.combinations(range(units + model_count - 1), model_count - 1):
        points = (-1, *bars, units + model_count - 1)
        counts = tuple(points[index + 1] - points[index] - 1 for index in range(model_count))
        if sum(counts) == units:
            vectors.append(tuple(count / units for count in counts))
    return vectors


@torch.no_grad()
def cache_validation_logits(checkpoints: list[Path], device: torch.device):
    dataset = build_train_dataset(cache_in_ram=False)
    _, val_indices = deterministic_split_indices(len(dataset), seed=config.SPLIT_SEED)
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    models, payloads = load_models(checkpoints, device)
    train_indices, _ = deterministic_split_indices(len(dataset), seed=config.SPLIT_SEED)
    train_names = [dataset.samples[index].filename for index in train_indices]
    val_names = [dataset.samples[index].filename for index in val_indices]
    for checkpoint, payload in zip(checkpoints, payloads):
        if payload.get("train_hash") != filename_hash(train_names):
            raise ValueError(f"training split hash mismatch: {checkpoint}")
        if payload.get("val_hash") != filename_hash(val_names):
            raise ValueError(f"validation split hash mismatch: {checkpoint}")
    char_cache = [[] for _ in models]
    color_cache = [[] for _ in models]
    char_targets = []
    color_targets = []
    for images, batch_char_targets, batch_color_targets in tqdm(loader, desc="cache logits"):
        images = images.to(device, non_blocking=True)
        for index, model in enumerate(models):
            char_logits, color_logits = model(images)
            char_cache[index].append(char_logits.cpu())
            color_cache[index].append(color_logits.cpu())
        char_targets.append(batch_char_targets)
        color_targets.append(batch_color_targets)
    return (
        torch.stack([torch.cat(parts) for parts in char_cache]),
        torch.stack([torch.cat(parts) for parts in color_cache]),
        torch.cat(char_targets),
        torch.cat(color_targets),
    )


def candidate_predictions(logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    combined = torch.einsum("wm,mnpc->wnpc", weights, logits)
    return combined.argmax(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search separate character/color ensemble weights.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=config.LOG_DIR / "weighted_ensemble_search.csv")
    args = parser.parse_args()

    inverse = round(1.0 / args.step)
    if inverse <= 0 or abs(inverse * args.step - 1.0) > 1e-9:
        raise ValueError("--step must divide 1.0 exactly, e.g. 0.1, 0.05, 0.02")

    seed_everything()
    device = torch.device(config.DEVICE)
    char_logits, color_logits, char_targets, color_targets = cache_validation_logits(args.checkpoints, device)
    vectors = generate_weight_vectors(len(args.checkpoints), inverse)
    weight_tensor = torch.tensor(vectors, dtype=char_logits.dtype, device=device)
    char_preds = candidate_predictions(char_logits.to(device), weight_tensor)
    color_preds = candidate_predictions(color_logits.to(device), weight_tensor)
    char_targets = char_targets.to(device)
    color_targets = color_targets.to(device)
    target_codes = encode_red_sequences(char_targets, color_targets)

    char_accs = char_preds.eq(char_targets.unsqueeze(0)).float().mean(dim=(1, 2))
    color_accs = color_preds.eq(color_targets.unsqueeze(0)).float().mean(dim=(1, 2))
    char_acc_values = char_accs.cpu().tolist()
    color_acc_values = color_accs.cpu().tolist()
    rows = []
    for color_index, color_pred in enumerate(tqdm(color_preds, desc="pair weights")):
        color_batch = color_pred.unsqueeze(0).expand(char_preds.size(0), -1, -1)
        pred_codes = encode_red_sequences(char_preds, color_batch)
        exacts = pred_codes.eq(target_codes.unsqueeze(0)).float().mean(dim=1)
        joint_accs = (
            char_preds.eq(char_targets.unsqueeze(0))
            & color_batch.eq(color_targets.unsqueeze(0))
        ).float().mean(dim=(1, 2))
        exact_values = exacts.cpu().tolist()
        joint_acc_values = joint_accs.cpu().tolist()
        for char_index in range(char_preds.size(0)):
            rows.append(
                {
                    "exact": exact_values[char_index],
                    "char_acc": char_acc_values[char_index],
                    "color_acc": color_acc_values[color_index],
                    "joint_pos_acc": joint_acc_values[char_index],
                    "char_weights": "|".join(f"{weight:.6g}" for weight in vectors[char_index]),
                    "color_weights": "|".join(f"{weight:.6g}" for weight in vectors[color_index]),
                }
            )

    rows.sort(
        key=lambda row: (row["exact"], row["char_acc"], row["color_acc"], row["joint_pos_acc"]),
        reverse=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("Best weights:")
    print(rows[0])
    print("Search log:", args.output)


if __name__ == "__main__":
    main()
