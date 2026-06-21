from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import TrainAugmentation, TransformSubset, build_train_dataset, deterministic_split_indices, filename_hash, seed_everything
from kfold import N_FOLDS, fold_split
from metrics import batch_metrics, compute_loss
from model import build_model, count_parameters


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


@dataclass(frozen=True)
class RunPaths:
    checkpoint_dir: Path
    log_path: Path

    def ensure(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            shadow = ema_state[key]
            if shadow.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)


def resolve_run_paths(run_name: str | None, output_root: Path = config.OUTPUT_DIR) -> RunPaths:
    if run_name:
        root = output_root / "runs" / run_name
        return RunPaths(root / "checkpoints", root / "logs" / "train_log.csv")
    return RunPaths(output_root / "checkpoints", output_root / "logs" / "train_log.csv")


def resolve_split_indices(n_items: int, fold: int | None, n_folds: int) -> tuple[list[int], list[int]]:
    if fold is None:
        return deterministic_split_indices(n_items, seed=config.SPLIT_SEED)
    return fold_split(n_items, fold=fold, n_folds=n_folds)


def make_loader(
    dataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int | None = None,
    transform=None,
) -> DataLoader:
    workers = config.NUM_WORKERS if num_workers is None else num_workers
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": config.PIN_MEMORY and config.DEVICE == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(TransformSubset(dataset, indices, transform=transform), **kwargs)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    red_char_weight: float = 1.0,
) -> tuple[float, dict[str, float]]:
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
            loss = compute_loss(
                char_logits,
                color_logits,
                char_targets,
                color_targets,
                red_char_weight=red_char_weight,
            )
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
    red_char_weight: float = 1.0,
    label_smoothing: float = config.LABEL_SMOOTHING,
    ema: ModelEMA | None = None,
    grad_clip: float = config.GRAD_CLIP_NORM,
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
            loss = compute_loss(
                char_logits,
                color_logits,
                char_targets,
                color_targets,
                red_char_weight=red_char_weight,
                label_smoothing=label_smoothing,
            )
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if steps_limit is not None and step >= steps_limit:
            break
    return total_loss / max(total_items, 1)


def build_scheduler(optimizer: AdamW, epochs: int, warmup_epochs: int, eta_min: float = config.ETA_MIN):
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    warmup = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def save_checkpoint(
    path,
    model,
    checkpoint_model,
    ema,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    train_names: list[str],
    val_names: list[str],
    training_seed: int,
    augment: bool,
    augment_preset: str,
    run_name: str | None,
    red_char_weight: float,
    model_size: str,
    fold: int | None,
    n_folds: int,
    lr: float,
    use_ema: bool,
    ema_decay: float,
    warmup_epochs: int,
    grad_clip: float,
    label_smoothing: float,
    red_line_aug: float,
) -> None:
    torch.save(
        {
            "state_dict": (checkpoint_model or model).state_dict(),
            "raw_state_dict": model.state_dict(),
            "ema_state_dict": ema.module.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": {
                "charset": config.CHARSET,
                "training_seed": training_seed,
                "split_seed": config.SPLIT_SEED,
                "val_ratio": config.VAL_RATIO,
                "batch_size": config.BATCH_SIZE,
                "epochs": config.EPOCHS,
                "lr": lr,
                "weight_decay": config.WEIGHT_DECAY,
                "color_loss_weight": config.COLOR_LOSS_WEIGHT,
                "red_char_weight": red_char_weight,
                "model_size": model_size,
                "augment": augment,
                "augment_preset": augment_preset,
                "run_name": run_name,
                "fold": fold,
                "n_folds": n_folds,
                "use_ema": use_ema,
                "ema_decay": ema_decay,
                "warmup_epochs": warmup_epochs,
                "grad_clip": grad_clip,
                "label_smoothing": label_smoothing,
                "red_line_aug": red_line_aug,
            },
            "train_hash": filename_hash(train_names),
            "val_hash": filename_hash(val_names),
        },
        path,
    )


def restore_training_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    ema: ModelEMA | None = None,
) -> tuple[int, float]:
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload.get("raw_state_dict") or payload["state_dict"])
    if ema is not None and payload.get("ema_state_dict") is not None:
        ema.module.load_state_dict(payload["ema_state_dict"])
    if "optimizer" in payload and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    metrics = payload.get("metrics", {})
    return int(payload.get("epoch", 0)), float(metrics.get("exact", -1.0))


def append_log(path, row: EpochMetrics) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(row).keys()), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(row))


def run_training(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    config.ensure_output_dirs()
    run_paths = resolve_run_paths(args.run_name)
    run_paths.ensure()
    device = torch.device(config.DEVICE)
    dataset = build_train_dataset(cache_in_ram=args.cache_in_ram)
    train_indices, val_indices = resolve_split_indices(len(dataset), args.fold, args.n_folds)
    train_names = [dataset.samples[idx].filename for idx in train_indices]
    val_names = [dataset.samples[idx].filename for idx in val_indices]

    effective_augment = args.augment and not args.overfit_sanity
    if args.overfit_sanity:
        overfit_indices = train_indices[:64]
        train_loader = make_loader(dataset, overfit_indices, batch_size=64, shuffle=True, num_workers=0)
        val_loader = make_loader(dataset, overfit_indices, batch_size=64, shuffle=False, num_workers=0)
        model = build_model(args.model_size, dropout=0.0).to(device)
        epochs = 300
        steps_limit = 1
        scheduler = None
    else:
        train_transform = TrainAugmentation.from_preset(args.augment_preset) if effective_augment else None
        if train_transform is not None:
            train_transform.red_line_p = args.red_line_aug
        train_loader = make_loader(
            dataset,
            train_indices,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=args.num_workers,
            transform=train_transform,
        )
        val_loader = make_loader(
            dataset,
            val_indices,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
        )
        model = build_model(args.model_size).to(device)
        epochs = args.epochs
        steps_limit = args.steps_per_epoch
        scheduler = None

    params = count_parameters(model)
    print(
        f"device={device} params={params} augment={effective_augment} "
        f"training_seed={args.seed} split_seed={config.SPLIT_SEED} "
        f"run_name={args.run_name or 'default'} red_char_weight={args.red_char_weight} "
        f"model_size={args.model_size} augment_preset={args.augment_preset} "
        f"fold={args.fold if args.fold is not None else 'holdout'} n_folds={args.n_folds}"
    )
    assert params > 5_000_000

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    use_ema = args.ema and not args.overfit_sanity
    ema = ModelEMA(model, args.ema_decay) if use_ema else None
    if not args.overfit_sanity:
        scheduler = build_scheduler(optimizer, epochs=epochs, warmup_epochs=args.warmup_epochs)
    scaler = GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    start_epoch = 0
    best_exact = -1.0
    if args.resume is not None:
        start_epoch, best_exact = restore_training_state(args.resume, model, optimizer, scheduler, device, ema=ema)
        best_checkpoint = run_paths.checkpoint_dir / "best.pt"
        if best_checkpoint.exists():
            best_payload = torch.load(best_checkpoint, map_location="cpu")
            best_exact = max(best_exact, float(best_payload.get("metrics", {}).get("exact", best_exact)))
        print(f"resumed checkpoint={args.resume} start_epoch={start_epoch} best_exact={best_exact:.4f}")
    use_amp = config.AMP and device.type == "cuda"
    log_path = (
        run_paths.log_path.with_name("overfit_log.csv")
        if args.overfit_sanity
        else run_paths.log_path
    )
    if log_path.exists() and args.resume is None:
        log_path.unlink()

    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            red_char_weight=args.red_char_weight,
            label_smoothing=0.0 if args.overfit_sanity else args.label_smoothing,
            ema=ema,
            grad_clip=0.0 if args.overfit_sanity else args.grad_clip,
            steps_limit=steps_limit,
        )
        eval_model = ema.module if ema is not None else model
        val_loss, metrics = evaluate_loader(eval_model, val_loader, device, red_char_weight=args.red_char_weight)
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
        checkpoint_metrics = {
            **asdict(row),
            "augment": effective_augment,
            "augment_preset": args.augment_preset,
            "red_char_weight": args.red_char_weight,
            "model_size": args.model_size,
            "use_ema": use_ema,
            "warmup_epochs": args.warmup_epochs,
            "grad_clip": args.grad_clip,
            "label_smoothing": 0.0 if args.overfit_sanity else args.label_smoothing,
            "red_line_aug": args.red_line_aug if effective_augment else 0.0,
        }
        save_checkpoint(
            run_paths.checkpoint_dir / "last.pt",
            model,
            model,
            ema,
            optimizer,
            scheduler,
            epoch,
            checkpoint_metrics,
            train_names,
            val_names,
            args.seed,
            effective_augment,
            args.augment_preset,
            args.run_name,
            args.red_char_weight,
            args.model_size,
            args.fold,
            args.n_folds,
            args.lr,
            use_ema,
            args.ema_decay,
            args.warmup_epochs,
            args.grad_clip,
            0.0 if args.overfit_sanity else args.label_smoothing,
            args.red_line_aug if effective_augment else 0.0,
        )
        if metrics["exact"] > best_exact:
            best_exact = metrics["exact"]
            save_checkpoint(
                run_paths.checkpoint_dir / "best.pt",
                model,
                eval_model,
                ema,
                optimizer,
                scheduler,
                epoch,
                checkpoint_metrics,
                train_names,
                val_names,
                args.seed,
                effective_augment,
                args.augment_preset,
                args.run_name,
                args.red_char_weight,
                args.model_size,
                args.fold,
                args.n_folds,
                args.lr,
                use_ema,
                args.ema_decay,
                args.warmup_epochs,
                args.grad_clip,
                0.0 if args.overfit_sanity else args.label_smoothing,
                args.red_line_aug if effective_augment else 0.0,
            )
        if args.overfit_sanity and train_loss < 0.01 and metrics["exact"] == 1.0:
            print("overfit sanity passed")
            return

    if args.overfit_sanity:
        raise SystemExit("overfit sanity failed: loss did not reach <0.01 and exact did not reach 100%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--overfit-sanity", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--red-char-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--warmup-epochs", type=int, default=config.WARMUP_EPOCHS)
    parser.add_argument("--grad-clip", type=float, default=config.GRAD_CLIP_NORM)
    parser.add_argument("--label-smoothing", type=float, default=config.LABEL_SMOOTHING)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=config.USE_EMA)
    parser.add_argument("--ema-decay", type=float, default=config.EMA_DECAY)
    parser.add_argument("--model-size", choices=list(config.MODEL_SIZES), default="base")
    parser.add_argument("--augment-preset", choices=list(config.AUGMENT_PRESETS), default="light")
    parser.add_argument("--red-line-aug", type=float, default=0.0)
    parser.add_argument("--cache-in-ram", action=argparse.BooleanOptionalAction, default=config.CACHE_IN_RAM)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


if __name__ == "__main__":
    run_training(parse_args())
