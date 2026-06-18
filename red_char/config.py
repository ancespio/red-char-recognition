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
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
EDA_DIR = OUTPUT_DIR / "eda"

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {ch: idx for idx, ch in enumerate(CHARSET)}
IDX_TO_CHAR = {idx: ch for idx, ch in enumerate(CHARSET)}

SEED = 42
SPLIT_SEED = 42
VAL_RATIO = 0.05
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 60
NUM_POSITIONS = 5
NUM_CHARS = len(CHARSET)
NUM_COLORS = 2
RED_INDEX = 1
NON_RED_INDEX = 0

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-5
COLOR_LOSS_WEIGHT = 1.0
NUM_WORKERS = 4
CACHE_IN_RAM = True
PIN_MEMORY = True
PERSISTENT_WORKERS = NUM_WORKERS > 0
AMP = True

AUGMENT_DEGREES = 3.0
AUGMENT_TRANSLATE = (0.05, 0.05)
AUGMENT_NOISE_STD = 0.01
AUGMENT_PRESETS = {
    "light": {
        "degrees": AUGMENT_DEGREES,
        "translate": AUGMENT_TRANSLATE,
        "noise_std": AUGMENT_NOISE_STD,
        "erase_scale": None,
    },
    "medium": {
        "degrees": 5.0,
        "translate": (0.08, 0.08),
        "noise_std": 0.02,
        "erase_scale": None,
    },
    "strong": {
        "degrees": 8.0,
        "translate": (0.10, 0.10),
        "noise_std": 0.03,
        "erase_scale": (0.02, 0.05),
    },
}
MODEL_SIZES = ("base", "wide", "k5", "resblock", "deep3")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, EDA_DIR):
        path.mkdir(parents=True, exist_ok=True)
