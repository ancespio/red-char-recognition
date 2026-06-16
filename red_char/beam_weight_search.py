from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from tqdm import tqdm

import config
from dataset import seed_everything
from metrics import encode_red_sequences
from weighted_ensemble_search import cache_validation_logits, candidate_predictions, generate_weight_vectors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top-k coarse and local ensemble weight search.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--coarse-step", type=float, default=0.1)
    parser.add_argument("--fine-step", type=float, default=0.02)
    parser.add_argument("--radius", type=float, default=0.08)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, default=config.LOG_DIR / "beam_weight_search.csv")
    return parser


def step_to_units(step: float) -> int:
    units = round(1.0 / step)
    if units <= 0 or abs(units * step - 1.0) > 1e-9:
        raise ValueError("step must divide 1.0 exactly, e.g. 0.1, 0.05, 0.02")
    return units


def generate_local_weight_vectors(center: tuple[float, ...], step: float, radius: float) -> list[tuple[float, ...]]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    units = step_to_units(step)
    ranges = []
    for value in center:
        low = max(0, math.ceil((value - radius) * units - 1e-9))
        high = min(units, math.floor((value + radius) * units + 1e-9))
        ranges.append(range(low, high + 1))

    vectors: list[tuple[float, ...]] = []
    counts: list[int] = []

    def visit(index: int, remaining: int) -> None:
        if index == len(ranges) - 1:
            if remaining in ranges[index]:
                vector = tuple((*counts, remaining)[i] / units for i in range(len(center)))
                vectors.append(vector)
            return
        for count in ranges[index]:
            if count > remaining:
                break
            counts.append(count)
            visit(index + 1, remaining - count)
            counts.pop()

    visit(0, units)
    return sorted(set(vectors))


def row_score(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row["exact"]),
        float(row["char_acc"]),
        float(row["color_acc"]),
        float(row["joint_pos_acc"]),
    )


def keep_top_rows(rows: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return sorted(rows, key=row_score, reverse=True)[:top_k]


def format_weights(weights: tuple[float, ...]) -> str:
    return "|".join(f"{weight:.6g}" for weight in weights)


def search_top_rows(
    char_logits: torch.Tensor,
    color_logits: torch.Tensor,
    char_targets: torch.Tensor,
    color_targets: torch.Tensor,
    char_vectors: list[tuple[float, ...]],
    color_vectors: list[tuple[float, ...]],
    top_k: int,
    stage: str,
) -> list[dict[str, object]]:
    device = char_targets.device
    char_weights = torch.tensor(char_vectors, dtype=char_logits.dtype, device=device)
    color_weights = torch.tensor(color_vectors, dtype=color_logits.dtype, device=device)
    char_preds = candidate_predictions(char_logits, char_weights)
    color_preds = candidate_predictions(color_logits, color_weights)
    target_codes = encode_red_sequences(char_targets, color_targets)
    char_accs = char_preds.eq(char_targets.unsqueeze(0)).float().mean(dim=(1, 2)).cpu().tolist()
    color_accs = color_preds.eq(color_targets.unsqueeze(0)).float().mean(dim=(1, 2)).cpu().tolist()

    top_rows: list[dict[str, object]] = []
    for color_index, color_pred in enumerate(tqdm(color_preds, desc=f"{stage} pairs")):
        color_batch = color_pred.unsqueeze(0).expand(char_preds.size(0), -1, -1)
        pred_codes = encode_red_sequences(char_preds, color_batch)
        exacts = pred_codes.eq(target_codes.unsqueeze(0)).float().mean(dim=1).cpu().tolist()
        joint_accs = (
            char_preds.eq(char_targets.unsqueeze(0))
            & color_batch.eq(color_targets.unsqueeze(0))
        ).float().mean(dim=(1, 2)).cpu().tolist()
        batch_rows = [
            {
                "stage": stage,
                "exact": exacts[char_index],
                "char_acc": char_accs[char_index],
                "color_acc": color_accs[color_index],
                "joint_pos_acc": joint_accs[char_index],
                "char_weights": format_weights(char_vectors[char_index]),
                "color_weights": format_weights(color_vectors[color_index]),
            }
            for char_index in range(char_preds.size(0))
        ]
        top_rows = keep_top_rows(top_rows + batch_rows, top_k)
    return top_rows


def parse_weight_string(raw: str) -> tuple[float, ...]:
    return tuple(float(part) for part in raw.split("|"))


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    seed_everything()
    device = torch.device(config.DEVICE)
    char_logits, color_logits, char_targets, color_targets = cache_validation_logits(args.checkpoints, device)
    char_logits = char_logits.to(device)
    color_logits = color_logits.to(device)
    char_targets = char_targets.to(device)
    color_targets = color_targets.to(device)

    coarse_vectors = generate_weight_vectors(len(args.checkpoints), step_to_units(args.coarse_step))
    print(f"coarse vectors: {len(coarse_vectors)}")
    coarse_rows = search_top_rows(
        char_logits,
        color_logits,
        char_targets,
        color_targets,
        coarse_vectors,
        coarse_vectors,
        args.top_k,
        "coarse",
    )

    best = coarse_rows[0]
    fine_char_center = parse_weight_string(str(best["char_weights"]))
    fine_color_center = parse_weight_string(str(best["color_weights"]))
    fine_char_vectors = generate_local_weight_vectors(fine_char_center, args.fine_step, args.radius)
    fine_color_vectors = generate_local_weight_vectors(fine_color_center, args.fine_step, args.radius)
    print(f"fine char vectors: {len(fine_char_vectors)}")
    print(f"fine color vectors: {len(fine_color_vectors)}")
    fine_rows = search_top_rows(
        char_logits,
        color_logits,
        char_targets,
        color_targets,
        fine_char_vectors,
        fine_color_vectors,
        args.top_k,
        "fine",
    )

    rows = keep_top_rows(coarse_rows + fine_rows, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("Best weights:")
    print(rows[0])
    print("Top-k log:", args.output)


if __name__ == "__main__":
    main()
