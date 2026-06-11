from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config
from dataset import build_train_dataset, deterministic_split_indices, filename_hash, seed_everything
from metrics import batch_metrics, compute_loss
from model import RedCharNet, count_parameters


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    exact: float
    char_acc: float
    color_acc: float
    joint_pos_acc: float
    lr: float


def make_loader(dataset, indices: list[int], batch_size: int, shuffle: bool, num_workers: int | None = None) -> DataLoader:
    workers = config.NUM_WORKERS if num_workers is None else num_workers
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": config.PIN_MEMORY and config.DEVICE == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(Subset(dataset, indices), **kwargs)


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    totals = {"exact": 0.0, "char_acc": 0.0, "color_acc": 0.0, "joint_pos_acc": 0.0}
    with torch.no_grad():
        for images, char_targets, color_targets in loader:
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
    return total_loss / total_items, {key: value / total_items for key, value in totals.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    steps_limit: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    progress = tqdm(loader, desc="train", leave=False)
    for step, (images, char_targets, color_targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        char_targets = char_targets.to(device, non_blocking=True)
        color_targets = color_targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            char_logits, color_logits = model(images)
            loss = compute_loss(char_logits, color_logits, char_targets, color_targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if steps_limit is not None and step >= steps_limit:
            break
    return total_loss / max(total_items, 1)


def save_checkpoint(path, model, optimizer, scheduler, epoch: int, metrics: dict, train_names: list[str], val_names: list[str]) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": {
                "charset": config.CHARSET,
                "seed": config.SEED,
                "val_ratio": config.VAL_RATIO,
                "batch_size": config.BATCH_SIZE,
                "epochs": config.EPOCHS,
                "lr": config.LR,
                "weight_decay": config.WEIGHT_DECAY,
                "color_loss_weight": config.COLOR_LOSS_WEIGHT,
            },
            "train_hash": filename_hash(train_names),
            "val_hash": filename_hash(val_names),
        },
        path,
    )


def append_log(path, row: EpochMetrics) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(row).keys()), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(row))


def run_training(args: argparse.Namespace) -> None:
    seed_everything()
    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    dataset = build_train_dataset(cache_in_ram=args.cache_in_ram)
    train_indices, val_indices = deterministic_split_indices(len(dataset))
    train_names = [dataset.samples[idx].filename for idx in train_indices]
    val_names = [dataset.samples[idx].filename for idx in val_indices]

    if args.overfit_sanity:
        overfit_indices = train_indices[:64]
        train_loader = make_loader(dataset, overfit_indices, batch_size=64, shuffle=True, num_workers=0)
        val_loader = make_loader(dataset, overfit_indices, batch_size=64, shuffle=False, num_workers=0)
        model = RedCharNet(dropout=0.0).to(device)
        epochs = 300
        steps_limit = 1
        scheduler = None
    else:
        train_loader = make_loader(dataset, train_indices, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = make_loader(dataset, val_indices, batch_size=config.BATCH_SIZE, shuffle=False)
        model = RedCharNet().to(device)
        epochs = args.epochs
        steps_limit = args.steps_per_epoch
        scheduler = None

    params = count_parameters(model)
    print(f"device={device} params={params}")
    assert 5_000_000 <= params <= 8_000_000

    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    if not args.overfit_sanity:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.ETA_MIN)
    scaler = GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    best_exact = -1.0
    use_amp = config.AMP and device.type == "cuda"
    log_path = config.LOG_DIR / ("overfit_log.csv" if args.overfit_sanity else "train_log.csv")
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, steps_limit=steps_limit)
        val_loss, metrics = evaluate_loader(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        row = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            exact=metrics["exact"],
            char_acc=metrics["char_acc"],
            color_acc=metrics["color_acc"],
            joint_pos_acc=metrics["joint_pos_acc"],
            lr=lr,
        )
        append_log(log_path, row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"exact={metrics['exact']:.4f} char={metrics['char_acc']:.4f} color={metrics['color_acc']:.4f}"
        )
        checkpoint_metrics = asdict(row)
        save_checkpoint(config.CHECKPOINT_DIR / "last.pt", model, optimizer, scheduler, epoch, checkpoint_metrics, train_names, val_names)
        if metrics["exact"] > best_exact:
            best_exact = metrics["exact"]
            save_checkpoint(config.CHECKPOINT_DIR / "best.pt", model, optimizer, scheduler, epoch, checkpoint_metrics, train_names, val_names)
        if args.overfit_sanity and train_loss < 0.01 and metrics["exact"] == 1.0:
            print("overfit sanity passed")
            return

    if args.overfit_sanity:
        raise SystemExit("overfit sanity failed: loss did not reach <0.01 and exact did not reach 100%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--overfit-sanity", action="store_true")
    parser.add_argument("--cache-in-ram", action=argparse.BooleanOptionalAction, default=config.CACHE_IN_RAM)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
