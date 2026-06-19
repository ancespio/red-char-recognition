"""Pseudo-label self-training (transductive adaptation to the test set).

A strong teacher ensemble labels the 5000 test images; only predictions that
are confident at *every* position (char and colour) are kept and added to the
training pool. This adapts the student to the test distribution, which is the
most promising lever for the ~0.8% val->leaderboard gap.

The deterministic 2500-image validation split is kept pure (true labels, no
pseudo, no augment), so the reported exact-match stays honest and directly
comparable to the v2/v2hi runs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Subset

import config
from augment import TrainAugment
from dataset import (
    RedCharDataset,
    Sample,
    _AugmentedSubset,
    build_train_dataset,
    deterministic_split_indices,
    load_submission_sample,
    seed_everything,
)
from model import build_model, count_parameters
from predict import load_model
from train import (
    EpochMetrics,
    ModelEMA,
    append_log,
    build_scheduler,
    evaluate_loader,
    make_loader,
    save_checkpoint,
    train_one_epoch,
)


@torch.no_grad()
def generate_pseudo_labels(teacher_ckpts, device, char_th: float, color_th: float, use_ema: bool):
    sample = load_submission_sample()
    samples = [Sample(row.id) for row in sample.itertuples(index=False)]
    ds = RedCharDataset(samples, config.TEST_IMAGES, is_test=True, cache_in_ram=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4,
                        pin_memory=device.type == "cuda")
    models = [load_model(c, device, use_ema=use_ema) for c in teacher_ckpts]
    print(f"teacher ensemble: {len(models)} models; thresholds char>={char_th} color>={color_th}")

    pseudo: list[Sample] = []
    for images, filenames in loader:
        images = images.to(device, non_blocking=True)
        char_prob = color_prob = None
        for m in models:
            cl, kl = m(images)
            cp, kp = F.softmax(cl, -1), F.softmax(kl, -1)
            char_prob = cp if char_prob is None else char_prob + cp
            color_prob = kp if color_prob is None else color_prob + kp
        char_prob /= len(models)
        color_prob /= len(models)
        char_conf, char_idx = char_prob.max(-1)   # [B,5]
        color_conf, color_idx = color_prob.max(-1)
        keep = (char_conf.min(-1).values >= char_th) & (color_conf.min(-1).values >= color_th)
        for i in range(images.size(0)):
            if not bool(keep[i]):
                continue
            all_label = "".join(config.IDX_TO_CHAR[int(j)] for j in char_idx[i].tolist())
            color = "".join("r" if int(j) == config.RED_INDEX else "u" for j in color_idx[i].tolist())
            pseudo.append(Sample(filenames[i], color, all_label))
    print(f"kept {len(pseudo)}/{len(ds)} confident pseudo-labelled test images")
    return pseudo


def run(args: argparse.Namespace) -> None:
    if args.heavy_aug:
        config.AUG_HEAVY = True
    if getattr(args, "full_data", False):
        config.VAL_RATIO = 0.01
    seed_everything(args.seed)
    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    tag = args.tag

    # --- build pure val split + true-labelled train split -------------------
    base = build_train_dataset(cache_in_ram=args.cache_in_ram)
    train_idx, val_idx = deterministic_split_indices(len(base), config.VAL_RATIO)
    train_names = [base.samples[i].filename for i in train_idx]
    val_names = [base.samples[i].filename for i in val_idx]
    true_train = _AugmentedSubset(base, train_idx, TrainAugment())
    val_ds = Subset(base, val_idx)

    # --- pseudo-labelled test images ---------------------------------------
    pseudo_samples = generate_pseudo_labels(
        args.teacher, device, args.char_th, args.color_th, use_ema=True
    )
    pseudo_base = RedCharDataset(pseudo_samples, config.TEST_IMAGES, is_test=False,
                                 cache_in_ram=args.cache_in_ram)
    pseudo_train = _AugmentedSubset(pseudo_base, list(range(len(pseudo_base))), TrainAugment())

    train_ds = ConcatDataset([true_train, pseudo_train])
    print(f"combined train pool: {len(true_train)} true + {len(pseudo_train)} pseudo = {len(train_ds)}")
    train_loader = make_loader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- standard training loop (mirrors train.run_training) ----------------
    model_name = args.model
    model = build_model(model_name).to(device)
    print(f"student model={model_name} params={count_parameters(model):,} epochs={args.epochs} seed={args.seed}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, args.epochs, config.WARMUP_EPOCHS)
    ema = ModelEMA(model, config.EMA_DECAY)
    use_amp = config.AMP and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    best_exact = -1.0
    log_path = config.LOG_DIR / f"train_log{tag}.csv"
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp,
            ema=ema, grad_clip=config.GRAD_CLIP_NORM, label_smoothing=config.LABEL_SMOOTHING,
        )
        val_loss, metrics = evaluate_loader(ema.ema, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = EpochMetrics(epoch, train_loss, val_loss, metrics["exact"], metrics["char_acc"],
                           metrics["color_acc"], metrics["joint_pos_acc"], lr)
        append_log(log_path, row)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
              f"exact={metrics['exact']:.4f} char={metrics['char_acc']:.4f} color={metrics['color_acc']:.4f}")
        cm = asdict(row)
        save_checkpoint(config.CHECKPOINT_DIR / f"last{tag}.pt", model, ema, optimizer, scheduler, epoch, cm, train_names, val_names, model_name)
        if metrics["exact"] > best_exact:
            best_exact = metrics["exact"]
            save_checkpoint(config.CHECKPOINT_DIR / f"best{tag}.pt", model, ema, optimizer, scheduler, epoch, cm, train_names, val_names, model_name)
    print(f"pseudo self-training done; best val exact-match = {best_exact:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=Path, nargs="+", required=True, help="teacher ensemble checkpoints")
    p.add_argument("--model", type=str, default="v2hi", choices=["v1", "v2", "v2hi", "v3"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--tag", type=str, default="_pseudo")
    p.add_argument("--heavy-aug", action="store_true", help="stronger colour-safe augmentation")
    p.add_argument("--full-data", action="store_true", help="near-full-data retrain (VAL_RATIO->0.01)")
    p.add_argument("--char-th", type=float, default=0.997)
    p.add_argument("--color-th", type=float, default=0.997)
    p.add_argument("--cache-in-ram", action=argparse.BooleanOptionalAction, default=config.CACHE_IN_RAM)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
