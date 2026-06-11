from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config
from dataset import build_train_dataset, deterministic_split_indices, filename_hash, seed_everything
from metrics import batch_metrics, compute_loss
from model import RedCharNet


def load_model(checkpoint: Path, device: torch.device) -> tuple[RedCharNet, dict]:
    payload = torch.load(checkpoint, map_location=device)
    model = RedCharNet().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def evaluate(checkpoint: Path) -> dict[str, float]:
    seed_everything()
    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    dataset = build_train_dataset(cache_in_ram=False)
    train_indices, val_indices = deterministic_split_indices(len(dataset))
    train_names = [dataset.samples[idx].filename for idx in train_indices]
    val_names = [dataset.samples[idx].filename for idx in val_indices]
    model, payload = load_model(checkpoint, device)
    assert payload.get("train_hash") == filename_hash(train_names)
    assert payload.get("val_hash") == filename_hash(val_names)

    loader = DataLoader(Subset(dataset, val_indices), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    totals = {"exact": 0.0, "char_acc": 0.0, "color_acc": 0.0, "joint_pos_acc": 0.0}
    total_loss = 0.0
    total_items = 0
    errors = []
    for batch_offset, (images, char_targets, color_targets) in enumerate(tqdm(loader, desc="evaluate")):
        images = images.to(device, non_blocking=True)
        char_targets = char_targets.to(device, non_blocking=True)
        color_targets = color_targets.to(device, non_blocking=True)
        char_logits, color_logits = model(images)
        loss = compute_loss(char_logits, color_logits, char_targets, color_targets)
        metrics = batch_metrics(char_logits, color_logits, char_targets, color_targets)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        for key, value in metrics.items():
            totals[key] += value * batch_size

        char_pred = char_logits.argmax(dim=-1).cpu()
        color_pred = color_logits.argmax(dim=-1).cpu()
        char_cpu = char_targets.cpu()
        color_cpu = color_targets.cpu()
        for i in range(batch_size):
            pred_red = tuple(int(ch) for ch, color in zip(char_pred[i].tolist(), color_pred[i].tolist()) if color == config.RED_INDEX)
            target_red = tuple(int(ch) for ch, color in zip(char_cpu[i].tolist(), color_cpu[i].tolist()) if color == config.RED_INDEX)
            if pred_red == target_red:
                continue
            global_idx = val_indices[batch_offset * config.BATCH_SIZE + i]
            char_ok = torch.equal(char_pred[i], char_cpu[i])
            color_ok = torch.equal(color_pred[i], color_cpu[i])
            if char_ok:
                err_type = "color_only"
            elif color_ok:
                err_type = "char_only"
            else:
                err_type = "both"
            errors.append(
                {
                    "filename": dataset.samples[global_idx].filename,
                    "type": err_type,
                    "target_color": dataset.samples[global_idx].color,
                    "target_all_label": dataset.samples[global_idx].all_label,
                    "pred_char_idx": " ".join(map(str, char_pred[i].tolist())),
                    "pred_color_idx": " ".join(map(str, color_pred[i].tolist())),
                }
            )

    result = {"val_loss": total_loss / total_items}
    result.update({key: value / total_items for key, value in totals.items()})

    out_path = config.EDA_DIR / "val_errors.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["filename", "type", "target_color", "target_all_label", "pred_char_idx", "pred_color_idx"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(errors)
    print("metrics:", result)
    print("errors written:", out_path, len(errors))
    best_metrics = payload.get("metrics", {})
    if best_metrics:
        print("checkpoint metrics:", best_metrics)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_DIR / "best.pt")
    args = parser.parse_args()
    evaluate(args.checkpoint)


if __name__ == "__main__":
    main()
