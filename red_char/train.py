from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config
from dataset import (
    build_split_datasets,
    build_train_dataset,
    deterministic_split_indices,
    filename_hash,
    seed_everything,
)
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


class ModelEMA:
    """Exponential moving average of model weights (params + BN buffers).

    The averaged weights are usually a smoother, better-generalising solution
    than the raw final weights, so we select/export ``best.pt`` from the EMA
    model. Costs one extra (gradient-free) copy per step.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.ema = deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.ema.state_dict()
        for key, value in model.state_dict().items():
            shadow = ema_state[key]
            if shadow.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int | None = None) -> DataLoader:
    workers = config.NUM_WORKERS if num_workers is None else num_workers
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": config.PIN_MEMORY and config.DEVICE == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


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
    ema: ModelEMA | None = None,
    grad_clip: float = config.GRAD_CLIP_NORM,
    label_smoothing: float = config.LABEL_SMOOTHING,
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
            loss = compute_loss(char_logits, color_logits, char_targets, color_targets, label_smoothing=label_smoothing)
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


def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, metrics, train_names, val_names, model_name) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "ema_state_dict": ema.ema.state_dict() if ema is not None else None,
            "model_name": model_name,
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
                "model": model_name,
                "use_ema": ema is not None,
                "use_augment": config.USE_AUGMENT,
                "label_smoothing": config.LABEL_SMOOTHING,
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


def build_scheduler(optimizer: AdamW, epochs: int, warmup_epochs: int):
    """Linear warmup followed by cosine annealing (stepped per epoch)."""
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.ETA_MIN)
    warmup = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=config.ETA_MIN)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def run_training(args: argparse.Namespace) -> None:
    if getattr(args, "heavy_aug", False):
        config.AUG_HEAVY = True
    if getattr(args, "denoised", False):
        config.TRAIN_IMAGES = config.DENOISED_TRAIN  # train on line-removed images
    seed_everything(args.seed)
    config.ensure_output_dirs()
    tag = args.tag
    device = torch.device(config.DEVICE)
    model_name = args.model or config.MODEL

    if args.overfit_sanity:
        # Pure memorisation check: no augment, no EMA, no dropout, no grad clip.
        dataset = build_train_dataset(cache_in_ram=args.cache_in_ram)
        train_indices, val_indices = deterministic_split_indices(len(dataset))
        train_names = [dataset.samples[i].filename for i in train_indices]
        val_names = [dataset.samples[i].filename for i in val_indices]
        overfit = Subset(dataset, train_indices[:64])
        train_loader = make_loader(overfit, batch_size=64, shuffle=True, num_workers=0)
        val_loader = make_loader(overfit, batch_size=64, shuffle=False, num_workers=0)
        model = build_model(model_name, dropout=0.0).to(device)
        epochs, steps_limit, warmup, use_ema, grad_clip = 300, 1, 0, False, 0.0
        label_smoothing = 0.0  # pure memorisation: CE must be able to reach 0
        augment_flag = False
    elif args.fold is not None:
        # K-fold training: train on all folds but `fold`, validate on `fold`.
        # The resulting checkpoint produces unbiased OOF predictions for `fold`.
        from torch.utils.data import Subset as _Subset
        from kfold import fold_split
        base = build_train_dataset(cache_in_ram=args.cache_in_ram)
        tr_idx, oof_idx = fold_split(len(base), args.fold, args.n_folds)
        train_names = [base.samples[i].filename for i in tr_idx]
        val_names = [base.samples[i].filename for i in oof_idx]
        if args.augment:
            from dataset import _AugmentedSubset
            from augment import TrainAugment
            train_ds = _AugmentedSubset(base, tr_idx, TrainAugment())
        else:
            train_ds = _Subset(base, tr_idx)
        val_ds = _Subset(base, oof_idx)
        train_loader = make_loader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        model = build_model(model_name).to(device)
        epochs = args.epochs
        steps_limit = args.steps_per_epoch
        warmup = config.WARMUP_EPOCHS
        use_ema = args.ema
        grad_clip = config.GRAD_CLIP_NORM
        label_smoothing = config.LABEL_SMOOTHING
        augment_flag = args.augment
        print(f"K-fold: fold={args.fold}/{args.n_folds} train={len(tr_idx)} oof={len(oof_idx)}")
    else:
        dn_dir = config.DENOISED_TRAIN if getattr(args, "concat_denoised", False) else None
        train_ds, val_ds, train_names, val_names, _ = build_split_datasets(
            cache_in_ram=args.cache_in_ram, augment=args.augment, denoised_dir=dn_dir,
            red_line_p=getattr(args, "red_line_aug", 0.0),
        )
        train_loader = make_loader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        model = build_model(model_name).to(device)
        epochs = args.epochs
        steps_limit = args.steps_per_epoch
        warmup = config.WARMUP_EPOCHS
        use_ema = args.ema
        grad_clip = config.GRAD_CLIP_NORM
        label_smoothing = config.LABEL_SMOOTHING
        augment_flag = args.augment

    params = count_parameters(model)
    print(f"device={device} model={model_name} params={params:,} augment={augment_flag} ema={use_ema} epochs={epochs}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = None if args.overfit_sanity else build_scheduler(optimizer, epochs, warmup)
    ema = ModelEMA(model, config.EMA_DECAY) if use_ema else None
    scaler = GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    use_amp = config.AMP and device.type == "cuda"
    best_exact = -1.0
    log_path = config.LOG_DIR / ("overfit_log.csv" if args.overfit_sanity else f"train_log{tag}.csv")
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp,
            ema=ema, grad_clip=grad_clip, label_smoothing=label_smoothing, steps_limit=steps_limit,
        )
        eval_model = ema.ema if ema is not None else model
        val_loss, metrics = evaluate_loader(eval_model, val_loader, device)
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
        save_checkpoint(config.CHECKPOINT_DIR / f"last{tag}.pt", model, ema, optimizer, scheduler, epoch, checkpoint_metrics, train_names, val_names, model_name)
        if metrics["exact"] > best_exact:
            best_exact = metrics["exact"]
            save_checkpoint(config.CHECKPOINT_DIR / f"best{tag}.pt", model, ema, optimizer, scheduler, epoch, checkpoint_metrics, train_names, val_names, model_name)
        if args.overfit_sanity and train_loss < 0.01 and metrics["exact"] == 1.0:
            print("overfit sanity passed")
            return

    if args.overfit_sanity:
        raise SystemExit("overfit sanity failed: loss did not reach <0.01 and exact did not reach 100%")
    print(f"training done; best val exact-match = {best_exact:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--overfit-sanity", action="store_true")
    parser.add_argument("--model", type=str, default=None, choices=["v1", "v2", "v2hi", "v2hi6", "v3"], help="override config.MODEL")
    parser.add_argument("--lr", type=float, default=config.LR, help="peak LR (lower for pretrained v3)")
    parser.add_argument("--cache-in-ram", action=argparse.BooleanOptionalAction, default=config.CACHE_IN_RAM)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=config.USE_AUGMENT)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=config.USE_EMA)
    parser.add_argument("--heavy-aug", action="store_true", help="enable stronger colour-safe augmentation (fights overfit)")
    parser.add_argument("--denoised", action="store_true", help="train on the line-removed (U-Net denoised) images")
    parser.add_argument("--concat-denoised", action="store_true", help="6ch input: original + denoised (use with --model v2hi6)")
    parser.add_argument("--red-line-aug", type=float, default=0.0, help="prob of overlaying synthetic red lines (robustness)")
    parser.add_argument("--seed", type=int, default=config.SEED, help="global seed for init/shuffle/aug; val split stays fixed")
    parser.add_argument("--tag", type=str, default="", help="suffix for checkpoint/log filenames, e.g. _seed1")
    parser.add_argument("--fold", type=int, default=None, help="K-fold OOF: train on all folds but this one, validate on it")
    parser.add_argument("--n-folds", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
