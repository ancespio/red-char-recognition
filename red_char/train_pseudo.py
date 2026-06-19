from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

import config
from dataset import (
    RedCharDataset,
    Sample,
    TrainAugmentation,
    build_test_dataset,
    build_train_dataset,
    deterministic_split_indices,
    seed_everything,
)
from ensemble import average_model_logits, load_models
from model import build_model, count_parameters
from train import EpochMetrics, append_log, evaluate_loader, restore_training_state, save_checkpoint, train_one_epoch


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform=None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if self.transform is None:
            return item
        values = list(item)
        values[0] = self.transform(values[0])
        return tuple(values)


def make_direct_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": config.PIN_MEMORY and config.DEVICE == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def select_confident_pseudo_samples(
    filenames: list[str],
    char_prob: torch.Tensor,
    color_prob: torch.Tensor,
    char_threshold: float,
    color_threshold: float,
) -> list[Sample]:
    char_conf, char_idx = char_prob.max(dim=-1)
    color_conf, color_idx = color_prob.max(dim=-1)
    keep = (char_conf.min(dim=-1).values >= char_threshold) & (color_conf.min(dim=-1).values >= color_threshold)
    samples: list[Sample] = []
    for row_idx, filename in enumerate(filenames):
        if not bool(keep[row_idx]):
            continue
        all_label = "".join(config.IDX_TO_CHAR[int(idx)] for idx in char_idx[row_idx].tolist())
        color = "".join("r" if int(idx) == config.RED_INDEX else "u" for idx in color_idx[row_idx].tolist())
        samples.append(Sample(filename, color, all_label))
    return samples


@torch.no_grad()
def generate_pseudo_labels(
    teacher_checkpoints: list[Path],
    device: torch.device,
    char_threshold: float,
    color_threshold: float,
    char_weights: list[float] | None = None,
    color_weights: list[float] | None = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,
) -> list[Sample]:
    dataset = build_test_dataset(cache_in_ram=False)
    loader = make_direct_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    models, _ = load_models(teacher_checkpoints, device)
    pseudo_samples: list[Sample] = []
    print(
        f"teacher_models={len(models)} char_threshold={char_threshold} "
        f"color_threshold={color_threshold}"
    )
    for images, filenames in tqdm(loader, desc="pseudo-label"):
        images = images.to(device, non_blocking=True)
        char_logits, color_logits = average_model_logits(
            models,
            images,
            char_weights=char_weights,
            color_weights=color_weights,
        )
        pseudo_samples.extend(
            select_confident_pseudo_samples(
                list(filenames),
                F.softmax(char_logits, dim=-1).cpu(),
                F.softmax(color_logits, dim=-1).cpu(),
                char_threshold=char_threshold,
                color_threshold=color_threshold,
            )
        )
    print(f"kept_pseudo={len(pseudo_samples)}/{len(dataset)}")
    return pseudo_samples


def run_training(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    run_root = config.OUTPUT_DIR / "runs" / args.run_name
    checkpoint_dir = run_root / "checkpoints"
    log_path = run_root / "logs" / "train_log.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    base = build_train_dataset(cache_in_ram=args.cache_in_ram)
    train_indices, val_indices = deterministic_split_indices(len(base), seed=config.SPLIT_SEED)
    train_names = [base.samples[idx].filename for idx in train_indices]
    val_names = [base.samples[idx].filename for idx in val_indices]

    pseudo_samples = generate_pseudo_labels(
        args.teacher_checkpoints,
        device,
        char_threshold=args.char_threshold,
        color_threshold=args.color_threshold,
        char_weights=args.char_weights,
        color_weights=args.color_weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    pseudo_base = RedCharDataset(pseudo_samples, config.TEST_IMAGES, is_test=False, cache_in_ram=args.cache_in_ram)
    train_transform = TrainAugmentation.from_preset(args.augment_preset) if args.augment else None
    train_ds = ConcatDataset(
        [
            TransformDataset(Subset(base, train_indices), train_transform),
            TransformDataset(pseudo_base, train_transform),
        ]
    )
    val_ds = Subset(base, val_indices)
    train_loader = make_direct_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_direct_loader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model_size).to(device)
    print(
        f"device={device} model_size={args.model_size} params={count_parameters(model):,} "
        f"true_train={len(train_indices)} pseudo_train={len(pseudo_samples)} val={len(val_indices)}"
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=config.ETA_MIN)
    scaler = GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    use_amp = config.AMP and device.type == "cuda"
    best_exact = -1.0
    start_epoch = 0
    if args.resume:
        start_epoch, best_exact = restore_training_state(args.resume, model, optimizer, scheduler, device)
        best_path = checkpoint_dir / "best.pt"
        if best_path.exists():
            payload = torch.load(best_path, map_location="cpu")
            best_exact = max(best_exact, float(payload.get("metrics", {}).get("exact", -1.0)))
        print(f"resumed_from={args.resume} start_epoch={start_epoch} best_exact={best_exact:.4f}")
    elif args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(payload["state_dict"])
        print(f"initialized_from={args.init_checkpoint}")
    elif log_path.exists():
        log_path.unlink()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            red_char_weight=args.red_char_weight,
        )
        val_loss, metrics = evaluate_loader(model, val_loader, device, red_char_weight=args.red_char_weight)
        lr = optimizer.param_groups[0]["lr"]
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
        checkpoint_metrics = {
            **asdict(row),
            "model_size": args.model_size,
            "augment": args.augment,
            "augment_preset": args.augment_preset,
            "red_char_weight": args.red_char_weight,
            "pseudo_count": len(pseudo_samples),
            "char_threshold": args.char_threshold,
            "color_threshold": args.color_threshold,
        }
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            checkpoint_metrics,
            train_names,
            val_names,
            args.seed,
            args.augment,
            args.augment_preset,
            args.run_name,
            args.red_char_weight,
            args.model_size,
        )
        if metrics["exact"] > best_exact:
            best_exact = metrics["exact"]
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                checkpoint_metrics,
                train_names,
                val_names,
                args.seed,
                args.augment,
                args.augment_preset,
                args.run_name,
                args.red_char_weight,
                args.model_size,
            )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"exact={metrics['exact']:.4f} char={metrics['char_acc']:.4f} color={metrics['color_acc']:.4f}"
        )
    print(f"pseudo training done; best exact={best_exact:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--char-weights", type=float, nargs="+")
    parser.add_argument("--color-weights", type=float, nargs="+")
    parser.add_argument("--char-threshold", type=float, default=0.92)
    parser.add_argument("--color-threshold", type=float, default=0.90)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--run-name", type=str, default="local_pseudo_seed70")
    parser.add_argument("--model-size", choices=list(config.MODEL_SIZES), default="v2hi")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment-preset", choices=list(config.AUGMENT_PRESETS), default="light")
    parser.add_argument("--red-char-weight", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--cache-in-ram", action=argparse.BooleanOptionalAction, default=config.CACHE_IN_RAM)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    return parser


if __name__ == "__main__":
    run_training(build_parser().parse_args())
