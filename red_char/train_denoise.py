"""Train the line-removal U-Net on on-the-fly synthetic (lined->clean) pairs.

Every few hundred steps, dumps a preview of the U-Net applied to REAL captchas
(outputs/eda/denoise_preview.png) — this is the make-or-break check: if real
interference lines vanish while characters survive, the approach works.
"""
from __future__ import annotations

import argparse
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast

import config
from dataset import build_train_dataset, deterministic_split_indices, seed_everything
from denoise import LineRemoverUNet
from synth import make_pair


def synth_batch(bs: int) -> tuple[torch.Tensor, torch.Tensor]:
    lined = np.empty((bs, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH), np.float32)
    clean = np.empty_like(lined)
    for i in range(bs):
        lined[i], clean[i] = make_pair()
    return torch.from_numpy(lined), torch.from_numpy(clean)


@torch.no_grad()
def dump_preview(model, real_imgs, device, path):
    model.eval()
    x = real_imgs.to(device)
    y = model(x).cpu()
    rows = []
    for i in range(real_imgs.shape[0]):
        a = (real_imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        b = (y[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rows.append(np.concatenate([a, b], axis=1))
    sheet = np.concatenate(rows, axis=0)
    Image.fromarray(sheet).resize((sheet.shape[1] * 2, sheet.shape[0] * 2), Image.NEAREST).save(path)
    model.train()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--preview-every", type=int, default=500)
    args = p.parse_args()

    seed_everything(args.seed)
    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    model = LineRemoverUNet(base=args.base).to(device)
    ema = deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    l1 = nn.L1Loss()

    base = build_train_dataset(cache_in_ram=False)
    ti, _ = deterministic_split_indices(len(base))
    real_preview = torch.stack([base[i][0] for i in ti[:8]])

    model.train()
    for step in range(1, args.steps + 1):
        lined, clean = synth_batch(args.batch_size)
        lined, clean = lined.to(device), clean.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(lined)
            loss = l1(out, clean)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            for pe, me in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(me, alpha=0.001)
            for pe, me in zip(ema.buffers(), model.buffers()):
                pe.copy_(me)
        if step % 200 == 0:
            print(f"step={step}/{args.steps} l1={loss.item():.4f} lr={opt.param_groups[0]['lr']:.2e}")
        if step % args.preview_every == 0 or step == args.steps:
            dump_preview(ema, real_preview, device, config.EDA_DIR / f"denoise_preview{args.tag}.png")

    torch.save({"state_dict": model.state_dict(), "ema_state_dict": ema.state_dict(), "base": args.base},
               config.CHECKPOINT_DIR / f"denoiser{args.tag}.pt")
    print(f"saved denoiser -> {config.CHECKPOINT_DIR / f'denoiser{args.tag}.pt'}")
    print(f"preview -> {config.EDA_DIR / f'denoise_preview{args.tag}.png'}")


if __name__ == "__main__":
    main()
