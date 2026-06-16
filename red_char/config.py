from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_ROOT = PROJECT_ROOT / "红色字符识别"
TRAIN_IMAGES = DATA_ROOT / "train" / "images"
TEST_IMAGES = DATA_ROOT / "test" / "images"
TRAIN_LABELS = DATA_ROOT / "train" / "labels.csv"
SUBMISSION_SAMPLE = DATA_ROOT / "submission_sample.csv"

OUTPUT_DIR = ROOT / "outputs"
DENOISED_TRAIN = OUTPUT_DIR / "denoised" / "train" / "images"
DENOISED_TEST = OUTPUT_DIR / "denoised" / "test" / "images"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
EDA_DIR = OUTPUT_DIR / "eda"

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {ch: idx for idx, ch in enumerate(CHARSET)}
IDX_TO_CHAR = {idx: ch for idx, ch in enumerate(CHARSET)}

SEED = 42
VAL_RATIO = 0.05
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 60
NUM_POSITIONS = 5
NUM_CHARS = len(CHARSET)
NUM_COLORS = 2
RED_INDEX = 1
NON_RED_INDEX = 0

BATCH_SIZE = 256
EPOCHS = 40
LR = 1.5e-3
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-5
COLOR_LOSS_WEIGHT = 1.0
NUM_WORKERS = 4
CACHE_IN_RAM = True
PIN_MEMORY = True
PERSISTENT_WORKERS = NUM_WORKERS > 0
AMP = True

# --- Model selection ---------------------------------------------------------
# "v2": residual + Squeeze-Excite + CoordConv backbone (default, stronger).
# "v1": original plain 4-block CNN (kept for a fair baseline comparison).
MODEL = "v2"

# --- Optimisation extras -----------------------------------------------------
WARMUP_EPOCHS = 2          # linear LR warmup before cosine annealing
GRAD_CLIP_NORM = 5.0       # max grad-norm; <=0 disables clipping
LABEL_SMOOTHING = 0.05     # applied to the character head only (0.0 = off)

# --- Exponential Moving Average of weights -----------------------------------
# EMA reliably improves a multi-task classifier's val/exact metric for free.
# best.pt is selected on the EMA model's exact-match.
USE_EMA = True
EMA_DECAY = 0.999

# --- Data augmentation (training split only; never on val/test) --------------
# IMPORTANT: only geometric + light additive noise. NEVER hue/saturation/
# channel-swap/grayscale, because the red colour IS the supervision signal.
USE_AUGMENT = True
AUG_TRANSLATE = 0.06       # fraction of width/height for random translation
AUG_SCALE = (0.90, 1.10)   # random scale range
AUG_DEGREES = 4.0          # random rotation (degrees)
AUG_NOISE_STD = 0.02       # std of additive Gaussian noise (on [0,1] pixels)

# --- Heavy augmentation (opt-in, to fight the ~1% train/val overfit gap) -----
# Clean-train exact-match is ~0.9997 while val is ~0.990 -> high variance, so
# stronger colour-PRESERVING augmentation (bigger affine + perspective + blur +
# more noise) is the right lever. Still NO hue/saturation/channel ops. Enabled
# per-run via train.py/train_pseudo.py --heavy-aug (sets config.AUG_HEAVY=True).
AUG_HEAVY = False
AUG_H_TRANSLATE = 0.09
AUG_H_SCALE = (0.84, 1.16)
AUG_H_DEGREES = 8.0
AUG_H_NOISE_STD = 0.04
AUG_H_PERSPECTIVE = 0.22   # torchvision perspective distortion_scale
AUG_H_PERSPECTIVE_P = 0.5
AUG_H_BLUR_P = 0.30        # prob of a light Gaussian blur (sigma 0.1-1.1)

# --- TF32 (Ampere+ GPUs such as the A100) ------------------------------------
ALLOW_TF32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def enable_perf_flags() -> None:
    """Enable cuDNN benchmark + TF32 matmul/conv on supported GPUs."""
    torch.backends.cudnn.benchmark = True
    if ALLOW_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, EDA_DIR):
        path.mkdir(parents=True, exist_ok=True)
