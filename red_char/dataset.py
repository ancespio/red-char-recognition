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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomAffine

import config


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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
    ) -> None:
        self.samples = samples
        self.image_dir = image_dir
        self.is_test = is_test
        self.cache_in_ram = cache_in_ram
        self._cache: list[torch.Tensor] | None = None
        if cache_in_ram:
            self._cache = [self._load_image(sample.filename) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, filename: str) -> torch.Tensor:
        path = self.image_dir / filename
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT):
                raise ValueError(f"{path} size must be 200x60, got {img.size}")
            return _image_to_tensor(img)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._cache[index] if self._cache is not None else self._load_image(sample.filename)
        if self.is_test:
            return image, sample.filename
        if sample.color is None or sample.all_label is None:
            raise ValueError("training sample is missing labels")
        char_target, color_target = encode_labels(sample.color, sample.all_label)
        return image, char_target, color_target


class TrainAugmentation:
    def __init__(
        self,
        degrees: float = config.AUGMENT_DEGREES,
        translate: tuple[float, float] = config.AUGMENT_TRANSLATE,
        noise_std: float = config.AUGMENT_NOISE_STD,
        erase_scale: tuple[float, float] | None = None,
        erase_prob: float = 0.25,
    ) -> None:
        self.affine = RandomAffine(
            degrees=degrees,
            translate=translate,
            interpolation=InterpolationMode.BILINEAR,
            fill=1.0,
        )
        self.noise_std = noise_std
        self.erase_scale = erase_scale
        self.erase_prob = erase_prob

    @classmethod
    def from_preset(cls, preset: str) -> "TrainAugmentation":
        try:
            values = config.AUGMENT_PRESETS[preset]
        except KeyError as exc:
            raise ValueError(f"unknown augment preset: {preset}") from exc
        return cls(**values)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        augmented = self.affine(image)
        if self.noise_std > 0:
            augmented = augmented + torch.randn_like(augmented) * self.noise_std
        augmented = augmented.clamp_(0.0, 1.0)
        if self.erase_scale is not None and torch.rand(()) < self.erase_prob:
            augmented = self._erase(augmented)
        return augmented

    def _erase(self, image: torch.Tensor) -> torch.Tensor:
        channels, height, width = image.shape
        min_scale, max_scale = self.erase_scale
        area = height * width
        erase_area = int(area * float(torch.empty(()).uniform_(min_scale, max_scale)))
        erase_h = max(1, min(height, int(erase_area ** 0.5)))
        erase_w = max(1, min(width, erase_area // erase_h))
        top = int(torch.randint(0, height - erase_h + 1, ()).item())
        left = int(torch.randint(0, width - erase_w + 1, ()).item())
        fill_value = float(torch.randint(0, 2, ()).item())
        erased = image.clone()
        erased[:, top : top + erase_h, left : left + erase_w] = fill_value
        return erased


class TransformSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], transform=None) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        item = self.dataset[self.indices[index]]
        if self.transform is None:
            return item
        values = list(item)
        values[0] = self.transform(values[0])
        return tuple(values)


def build_train_dataset(cache_in_ram: bool = config.CACHE_IN_RAM) -> RedCharDataset:
    df = load_train_frame()
    samples = [Sample(row.filename, row.color, row.all_label) for row in df.itertuples(index=False)]
    return RedCharDataset(samples, config.TRAIN_IMAGES, is_test=False, cache_in_ram=cache_in_ram)


def build_test_dataset(cache_in_ram: bool = False) -> RedCharDataset:
    sample_df = load_submission_sample()
    samples = [Sample(row.id) for row in sample_df.itertuples(index=False)]
    return RedCharDataset(samples, config.TEST_IMAGES, is_test=True, cache_in_ram=cache_in_ram)


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


def build_dataloaders(cache_in_ram: bool = config.CACHE_IN_RAM, augment: bool = False) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    dataset = build_train_dataset(cache_in_ram=cache_in_ram)
    train_indices, val_indices = deterministic_split_indices(len(dataset))
    train_transform = TrainAugmentation.from_preset("light") if augment else None
    train_loader = DataLoader(TransformSubset(dataset, train_indices, train_transform), **_loader_kwargs(shuffle=True))
    val_loader = DataLoader(TransformSubset(dataset, val_indices), **_loader_kwargs(shuffle=False))
    train_names = [dataset.samples[idx].filename for idx in train_indices]
    val_names = [dataset.samples[idx].filename for idx in val_indices]
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

    train_loader, _, train_names, val_names = build_dataloaders(cache_in_ram=False)
    assert len(train_names) == 47500
    assert len(val_names) == 2500
    _, _, train_names_2, val_names_2 = build_dataloaders(cache_in_ram=False)
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
