from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

import config


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config.enable_perf_flags()


def encode_labels(color: str, all_label: str) -> tuple[torch.Tensor, torch.Tensor]:
    if len(color) != config.NUM_POSITIONS or any(ch not in "ru" for ch in color):
        raise ValueError(f"invalid color label: {color!r}")
    if len(all_label) != config.NUM_POSITIONS or any(ch not in config.CHAR_TO_IDX for ch in all_label):
        raise ValueError(f"invalid char label: {all_label!r}")
    char_target = torch.tensor([config.CHAR_TO_IDX[ch] for ch in all_label], dtype=torch.long)
    color_target = torch.tensor([config.RED_INDEX if ch == "r" else config.NON_RED_INDEX for ch in color], dtype=torch.long)
    return char_target, color_target


def decode_prediction(char_indices: Iterable[int], color_indices: Iterable[int]) -> str:
    chars = [config.IDX_TO_CHAR[int(idx)] for idx in char_indices]
    colors = [int(idx) for idx in color_indices]
    if len(chars) != config.NUM_POSITIONS or len(colors) != config.NUM_POSITIONS:
        raise ValueError("prediction must contain exactly five positions")
    return "".join(ch for ch, color in zip(chars, colors) if color == config.RED_INDEX)


def target_to_red_string(color: str, all_label: str) -> str:
    char_target, color_target = encode_labels(color, all_label)
    return decode_prediction(char_target.tolist(), color_target.tolist())


def load_train_frame() -> pd.DataFrame:
    df = pd.read_csv(config.TRAIN_LABELS, dtype=str, keep_default_na=False)
    expected = ["filename", "color", "all_label"]
    if list(df.columns) != expected:
        raise ValueError(f"labels.csv columns must be {expected}, got {list(df.columns)}")
    return df


def load_submission_sample() -> pd.DataFrame:
    return pd.read_csv(config.SUBMISSION_SAMPLE, dtype=str, keep_default_na=False)


def deterministic_split_indices(n_items: int, val_ratio: float = config.VAL_RATIO, seed: int = config.SEED) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_items, generator=generator).tolist()
    val_size = int(round(n_items * val_ratio))
    val_indices = sorted(perm[:val_size])
    train_indices = sorted(perm[val_size:])
    return train_indices, val_indices


def filename_hash(filenames: Iterable[str]) -> str:
    payload = "\n".join(filenames).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


@dataclass(frozen=True)
class Sample:
    filename: str
    color: str | None = None
    all_label: str | None = None


class RedCharDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_dir: Path,
        is_test: bool = False,
        cache_in_ram: bool = config.CACHE_IN_RAM,
        denoised_dir: Path | None = None,
    ) -> None:
        self.samples = samples
        self.image_dir = image_dir
        self.is_test = is_test
        self.cache_in_ram = cache_in_ram
        # When set, each item is a 6-channel concat of [original | denoised],
        # so the model keeps original colour/fidelity AND a line-removed cue.
        self.denoised_dir = denoised_dir
        self._cache: list[torch.Tensor] | None = None
        if cache_in_ram:
            self._cache = [self._load_image(sample.filename) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_one(self, image_dir: Path, filename: str) -> torch.Tensor:
        path = image_dir / filename
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT):
                raise ValueError(f"{path} size must be 200x60, got {img.size}")
            return _image_to_tensor(img)

    def _load_image(self, filename: str) -> torch.Tensor:
        orig = self._load_one(self.image_dir, filename)
        if self.denoised_dir is None:
            return orig
        dn = self._load_one(self.denoised_dir, filename)
        return torch.cat([orig, dn], dim=0)  # [6, H, W]

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._cache[index] if self._cache is not None else self._load_image(sample.filename)
        if self.is_test:
            return image, sample.filename
        if sample.color is None or sample.all_label is None:
            raise ValueError("training sample is missing labels")
        char_target, color_target = encode_labels(sample.color, sample.all_label)
        return image, char_target, color_target


def build_train_dataset(cache_in_ram: bool = config.CACHE_IN_RAM,
                        denoised_dir: Path | None = None) -> RedCharDataset:
    df = load_train_frame()
    samples = [Sample(row.filename, row.color, row.all_label) for row in df.itertuples(index=False)]
    return RedCharDataset(samples, config.TRAIN_IMAGES, is_test=False, cache_in_ram=cache_in_ram,
                          denoised_dir=denoised_dir)


def build_test_dataset(cache_in_ram: bool = False) -> RedCharDataset:
    sample_df = load_submission_sample()
    samples = [Sample(row.id) for row in sample_df.itertuples(index=False)]
    return RedCharDataset(samples, config.TEST_IMAGES, is_test=True, cache_in_ram=cache_in_ram)


class _AugmentedSubset(Dataset):
    """View over a base dataset that applies a transform to the image only.

    Kept separate from ``RedCharDataset`` so the base dataset's RAM cache holds
    clean (un-augmented) tensors that are reused every epoch, while each
    ``__getitem__`` returns a freshly warped copy. Used for the training split
    only; the validation split reads the base dataset directly (no transform).
    """

    def __init__(self, base: RedCharDataset, indices: list[int], transform) -> None:
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, char_target, color_target = self.base[self.indices[index]]
        return self.transform(image), char_target, color_target


class SynthDataset(Dataset):
    """On-the-fly procedural synthetic captchas (labels exact by construction).

    Mixed into recognition training to add controlled coverage of hard
    confusions / occlusions. Style is matched to the real captcha in synth.py.
    """

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        import synth
        img, chars, colors = synth.make_labeled()
        return (torch.from_numpy(img),
                torch.tensor(chars, dtype=torch.long),
                torch.tensor(colors, dtype=torch.long))


def build_split_datasets(
    cache_in_ram: bool = config.CACHE_IN_RAM,
    augment: bool = config.USE_AUGMENT,
    denoised_dir: Path | None = None,
    red_line_p: float = 0.0,
) -> tuple[Dataset, Dataset, list[str], list[str], RedCharDataset]:
    """Deterministic train/val split with optional train-only augmentation.

    The split itself is unchanged from ``deterministic_split_indices`` so the
    filename hashes recorded in checkpoints stay comparable across runs.
    """
    base = build_train_dataset(cache_in_ram=cache_in_ram, denoised_dir=denoised_dir)
    train_indices, val_indices = deterministic_split_indices(len(base), config.VAL_RATIO)
    train_names = [base.samples[idx].filename for idx in train_indices]
    val_names = [base.samples[idx].filename for idx in val_indices]
    if augment:
        from augment import TrainAugment

        train_ds: Dataset = _AugmentedSubset(base, train_indices, TrainAugment(red_line_p=red_line_p))
    else:
        train_ds = Subset(base, train_indices)
    val_ds: Dataset = Subset(base, val_indices)
    return train_ds, val_ds, train_names, val_names, base


def _loader_kwargs(shuffle: bool) -> dict:
    kwargs = {
        "batch_size": config.BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY and config.DEVICE == "cuda",
    }
    if config.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = config.PERSISTENT_WORKERS
    return kwargs


def build_dataloaders(
    cache_in_ram: bool = config.CACHE_IN_RAM,
    augment: bool = config.USE_AUGMENT,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    train_ds, val_ds, train_names, val_names, _ = build_split_datasets(cache_in_ram, augment)
    train_loader = DataLoader(train_ds, **_loader_kwargs(shuffle=True))
    val_loader = DataLoader(val_ds, **_loader_kwargs(shuffle=False))
    return train_loader, val_loader, train_names, val_names


def _self_test() -> None:
    cases = [
        ("ruuur", "DPVKD", "DD"),
        ("uuuuu", "AB12C", ""),
        ("rrrrr", "AB12C", "AB12C"),
        ("rurur", "0A1B2", "012"),
        ("uurrr", "E25A7", "5A7"),
    ]
    for color, all_label, expected in cases:
        assert target_to_red_string(color, all_label) == expected

    # Split determinism (and hash stability) must hold regardless of augment.
    train_loader, _, train_names, val_names = build_dataloaders(cache_in_ram=False, augment=True)
    assert len(train_names) == 47500
    assert len(val_names) == 2500
    _, _, train_names_2, val_names_2 = build_dataloaders(cache_in_ram=False, augment=False)
    assert filename_hash(train_names) == filename_hash(train_names_2)
    assert filename_hash(val_names) == filename_hash(val_names_2)

    images, char_targets, color_targets = next(iter(train_loader))
    assert images.shape == (config.BATCH_SIZE, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    assert images.dtype == torch.float32
    assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0
    assert char_targets.shape == (config.BATCH_SIZE, config.NUM_POSITIONS)
    assert color_targets.shape == (config.BATCH_SIZE, config.NUM_POSITIONS)

    for _ in train_loader:
        pass
    print("dataset self-test passed")


if __name__ == "__main__":
    seed_everything()
    _self_test()
