from __future__ import annotations

import argparse
import csv

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import build_train_dataset, deterministic_split_indices, seed_everything
from glyph import GlyphDataset, GlyphNet
from train import ModelEMA, build_scheduler


CONFUSION_GROUPS = ("0OQ", "1ILT", "2Z7", "3S568E", "CG", "VY", "PF", "KX")


def glyph_loss(
    logits: torch.Tensor, targets: torch.Tensor, use_confusion_loss: bool = False
) -> torch.Tensor:
    """Full classification loss plus conditional losses inside hard glyph groups."""
    loss = F.cross_entropy(logits, targets, label_smoothing=0.02)
    if not use_confusion_loss:
        return loss
    group_losses = []
    for chars in CONFUSION_GROUPS:
        indices = torch.tensor([config.CHAR_TO_IDX[ch] for ch in chars], device=logits.device)
        mask = torch.isin(targets, indices)
        if not mask.any():
            continue
        local_targets = (targets[mask, None] == indices[None, :]).long().argmax(-1)
        group_losses.append(F.cross_entropy(logits[mask][:, indices], local_targets))
    if group_losses:
        loss = loss + torch.stack(group_losses).mean()
    return loss


def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=512,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and config.DEVICE == "cuda",
        persistent_workers=config.NUM_WORKERS > 0,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += logits.argmax(-1).eq(targets).sum().item()
        total += targets.numel()
    return total_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    epoch: int,
    val_acc: float,
    confusion_loss: bool,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "ema_state_dict": ema.ema.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "crop_width": model.crop_width,
            "hires": model.hires,
            "head_mode": model.head_mode,
            "model_version": ("gap" if model.head_mode == "gap" else "flat") + ("30x32" if model.hires else "15x16"),
            "confusion_loss": confusion_loss,
            "input_mode": model.input_mode,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--tag", type=str, default="_g1")
    parser.add_argument("--confusion-loss", action="store_true")
    parser.add_argument("--input-mode", choices=["rgb", "red"], default="rgb")
    parser.add_argument("--hires", action="store_true",
                        help="keep 30x32 feature map (one less pool) for finer stroke detail")
    parser.add_argument("--head-mode", choices=["flat", "gap"], default="flat",
                        help="gap = global avg pool head (tiny, no overfit); pairs well with --hires")
    parser.add_argument("--all-glyphs", action="store_true",
                        help="train on all 5 glyph slots (red+non-red) for ~5x data; "
                             "validation stays red-only (the deployment metric)")
    parser.add_argument("--red-line-aug", type=float, default=0.0,
                        help="probability of overlaying synthetic RED lines (robustness to residual red lines)")
    parser.add_argument("--crop-width", type=int, default=64,
                        help="glyph crop width; wider gives context to tell lines (span beyond glyph) from strokes")
    parser.add_argument("--boost-chars", type=str, default="",
                        help="characters to oversample in training, e.g. I1LVZT (vertical-stroke group)")
    parser.add_argument("--boost-factor", type=int, default=1, help="oversample multiplier for --boost-chars")
    parser.add_argument("--fold", type=int, default=None, help="K-fold OOF: hold this fold out")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cache-in-ram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    base = build_train_dataset(cache_in_ram=args.cache_in_ram)
    if args.fold is not None:
        from kfold import fold_split
        train_indices, val_indices = fold_split(len(base), args.fold, args.n_folds)
    else:
        train_indices, val_indices = deterministic_split_indices(len(base))
    train_ds = GlyphDataset(base, train_indices, red_only=not args.all_glyphs, augment=True,
                            red_line_p=args.red_line_aug, crop_width=args.crop_width,
                            boost_chars=args.boost_chars, boost_factor=args.boost_factor)
    val_ds = GlyphDataset(base, val_indices, red_only=True, augment=False, crop_width=args.crop_width)
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)

    model = GlyphNet(input_mode=args.input_mode, hires=args.hires, head_mode=args.head_mode,
                     crop_width=args.crop_width).to(device)
    # This position-level dataset has far fewer optimizer steps per epoch than
    # the full-image training. A faster EMA avoids carrying random initial
    # weights through most of a short reranker run.
    ema = ModelEMA(model, 0.99)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, args.epochs, config.WARMUP_EPOCHS)
    scaler = GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    use_amp = config.AMP and device.type == "cuda"
    best_acc = -1.0
    log_path = config.LOG_DIR / f"glyph_log{args.tag}.csv"
    if log_path.exists():
        log_path.unlink()

    print(
        f"device={device} train_glyphs={len(train_ds)} val_glyphs={len(val_ds)} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total = 0
        progress = tqdm(train_loader, desc="train", leave=False)
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = glyph_loss(logits, targets, use_confusion_loss=args.confusion_loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            total_loss += loss.item() * targets.numel()
            total += targets.numel()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_acc = evaluate(ema.ema, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
        }
        exists = log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys(), lineterminator="\n")
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        save_checkpoint(
            config.CHECKPOINT_DIR / f"last{args.tag}.pt",
            model,
            ema,
            epoch,
            val_acc,
            args.confusion_loss,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                config.CHECKPOINT_DIR / f"best{args.tag}.pt",
                model,
                ema,
                epoch,
                val_acc,
                args.confusion_loss,
            )
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.5f}"
        )
    print(f"training done; best red-glyph val accuracy={best_acc:.5f}")


if __name__ == "__main__":
    main()
