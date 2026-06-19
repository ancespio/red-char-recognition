from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import build_train_dataset, deterministic_split_indices, filename_hash, seed_everything
from glyph import GLYPH_CROP_WIDTH, GlyphDataset, GlyphNet
from kfold import N_FOLDS, fold_split
from model import count_parameters


def make_loader(dataset: GlyphDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": config.PIN_MEMORY and config.DEVICE == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def resolve_split_indices(n_items: int, fold: int | None, n_folds: int) -> tuple[list[int], list[int]]:
    if fold is None:
        return deterministic_split_indices(n_items, seed=config.SPLIT_SEED)
    return fold_split(n_items, fold=fold, n_folds=n_folds)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
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
    model: GlyphNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    val_acc: float,
    train_names: list[str],
    val_names: list[str],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "train_hash": filename_hash(train_names),
            "val_hash": filename_hash(val_names),
            "seed": args.seed,
            "run_name": args.run_name,
            "input_mode": args.input_mode,
            "hires": args.hires,
            "head_mode": args.head_mode,
            "crop_width": args.crop_width,
            "all_glyphs": args.all_glyphs,
            "augment": args.augment,
            "red_line_aug": args.red_line_aug,
            "fold": args.fold,
            "n_folds": args.n_folds,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--run-name", type=str, default="local_glyph_seed63")
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--input-mode", choices=["rgb", "red"], default="rgb")
    parser.add_argument("--hires", action="store_true")
    parser.add_argument("--head-mode", choices=["flat", "gap"], default="flat")
    parser.add_argument("--crop-width", type=int, default=GLYPH_CROP_WIDTH)
    parser.add_argument("--all-glyphs", action="store_true")
    parser.add_argument("--red-line-aug", type=float, default=0.0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-in-ram", action=argparse.BooleanOptionalAction, default=config.CACHE_IN_RAM)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    run_root = config.OUTPUT_DIR / "runs" / args.run_name
    checkpoint_dir = run_root / "checkpoints"
    log_path = run_root / "logs" / "glyph_log.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    base = build_train_dataset(cache_in_ram=args.cache_in_ram)
    train_indices, val_indices = resolve_split_indices(len(base), args.fold, args.n_folds)
    train_names = [base.samples[idx].filename for idx in train_indices]
    val_names = [base.samples[idx].filename for idx in val_indices]
    train_ds = GlyphDataset(
        base,
        train_indices,
        red_only=not args.all_glyphs,
        augment=args.augment,
        red_line_p=args.red_line_aug,
        crop_width=args.crop_width,
    )
    val_ds = GlyphDataset(base, val_indices, red_only=True, augment=False, crop_width=args.crop_width)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GlyphNet(input_mode=args.input_mode, hires=args.hires, head_mode=args.head_mode, crop_width=args.crop_width).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=config.ETA_MIN)
    scaler = GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    use_amp = config.AMP and device.type == "cuda"
    best_acc = -1.0
    start_epoch = 0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload.get("epoch", 0))
        best_acc = float(payload.get("val_acc", best_acc))
        best_path = checkpoint_dir / "best.pt"
        if best_path.exists():
            best_payload = torch.load(best_path, map_location="cpu")
            best_acc = max(best_acc, float(best_payload.get("val_acc", best_acc)))
        print(f"resumed checkpoint={args.resume} start_epoch={start_epoch} best_acc={best_acc:.5f}")
    elif log_path.exists():
        log_path.unlink()

    print(
        f"device={device} params={count_parameters(model):,} "
        f"train_glyphs={len(train_ds)} val_glyphs={len(val_ds)} "
        f"fold={args.fold if args.fold is not None else 'holdout'} n_folds={args.n_folds}"
    )
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        progress = tqdm(train_loader, desc="train", leave=False)
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, targets, label_smoothing=0.02)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM if hasattr(config, "GRAD_CLIP_NORM") else 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = targets.numel()
            total_loss += loss.item() * batch_size
            total += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
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
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, val_acc, train_names, val_names, args)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, val_acc, train_names, val_names, args)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.5f}"
        )
    print(f"training done; best red-glyph val accuracy={best_acc:.5f}")


if __name__ == "__main__":
    main()
