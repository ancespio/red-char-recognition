from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import build_train_dataset, deterministic_split_indices, seed_everything
from glyph import GlyphDataset, PAIR_GROUPS, PairGlyphNet
from train import ModelEMA, build_scheduler


def pair_loss(
    full_logits: torch.Tensor,
    pair_logits: list[torch.Tensor],
    targets: torch.Tensor,
) -> torch.Tensor:
    losses = [0.25 * F.cross_entropy(full_logits, targets, label_smoothing=0.02)]
    for chars, logits in zip(PAIR_GROUPS, pair_logits):
        indices = torch.tensor([config.CHAR_TO_IDX[ch] for ch in chars], device=targets.device)
        mask = torch.isin(targets, indices)
        if mask.any():
            local_targets = (targets[mask, None] == indices[None, :]).long().argmax(-1)
            losses.append(F.cross_entropy(logits[mask], local_targets))
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model: PairGlyphNet, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    group_correct = group_total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        full_logits, pair_logits = model(images)
        correct += full_logits.argmax(-1).eq(targets).sum().item()
        total += targets.numel()
        for chars, logits in zip(PAIR_GROUPS, pair_logits):
            indices = torch.tensor([config.CHAR_TO_IDX[ch] for ch in chars], device=device)
            mask = torch.isin(targets, indices)
            if mask.any():
                local_targets = (targets[mask, None] == indices[None, :]).long().argmax(-1)
                group_correct += logits[mask].argmax(-1).eq(local_targets).sum().item()
                group_total += mask.sum().item()
    return correct / total, group_correct / group_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-glyph", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=505)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tag", type=str, default="_pair1")
    args = parser.parse_args()

    seed_everything(args.seed)
    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    payload = torch.load(args.pretrained_glyph, map_location="cpu")
    input_mode = payload.get("input_mode", "rgb")
    model = PairGlyphNet(input_mode=input_mode)
    base_state = payload.get("ema_state_dict") or payload["state_dict"]
    model.base.load_state_dict(base_state)
    model = model.to(device)

    base = build_train_dataset(cache_in_ram=True)
    train_indices, val_indices = deterministic_split_indices(len(base))
    train_loader = DataLoader(
        GlyphDataset(base, train_indices, red_only=True, augment=True),
        batch_size=512,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        GlyphDataset(base, val_indices, red_only=True, augment=False),
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, args.epochs, 1)
    ema = ModelEMA(model, 0.99)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best_pair_acc = -1.0
    log_path = config.LOG_DIR / f"pair_log{args.tag}.csv"
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = item_total = 0
        progress = tqdm(train_loader, desc="train-pair", leave=False)
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                full_logits, pair_logits = model(images)
                loss = pair_loss(full_logits, pair_logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            loss_total += loss.item() * targets.numel()
            item_total += targets.numel()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        full_acc, pair_acc = evaluate(ema.ema, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": loss_total / item_total, "full_acc": full_acc, "pair_acc": pair_acc}
        exists = log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys(), lineterminator="\n")
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        checkpoint = {
            "state_dict": model.state_dict(),
            "ema_state_dict": ema.ema.state_dict(),
            "epoch": epoch,
            "full_acc": full_acc,
            "pair_acc": pair_acc,
            "input_mode": input_mode,
            "pair_groups": PAIR_GROUPS,
        }
        torch.save(checkpoint, config.CHECKPOINT_DIR / f"last{args.tag}.pt")
        if pair_acc > best_pair_acc:
            best_pair_acc = pair_acc
            torch.save(checkpoint, config.CHECKPOINT_DIR / f"best{args.tag}.pt")
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} full_acc={full_acc:.5f} pair_acc={pair_acc:.5f}")
    print(f"training done; best pair accuracy={best_pair_acc:.5f}")


if __name__ == "__main__":
    main()
