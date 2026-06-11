from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

import config


def _assert_labels(df: pd.DataFrame) -> None:
    assert len(df) == 50000
    assert list(df.columns) == ["filename", "color", "all_label"]
    assert df["filename"].str.fullmatch(r"\d{5}\.png").all()
    assert df["color"].str.fullmatch(r"[ru]{5}").all()
    assert df["all_label"].str.fullmatch(f"[{config.CHARSET}]{{5}}").all()


def _assert_images(df: pd.DataFrame) -> None:
    sample = df.sample(500, random_state=config.SEED)
    for filename in sample["filename"]:
        path = config.TRAIN_IMAGES / filename
        with Image.open(path) as img:
            assert img.size == (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
            assert img.mode == "RGB"


def _save_sample_grid(df: pd.DataFrame) -> None:
    config.ensure_output_dirs()
    sample = df.sample(16, random_state=config.SEED).reset_index(drop=True)
    fig, axes = plt.subplots(4, 4, figsize=(12, 5))
    for ax, row in zip(axes.ravel(), sample.itertuples(index=False)):
        with Image.open(config.TRAIN_IMAGES / row.filename) as img:
            ax.imshow(img)
        red = "".join(ch for c, ch in zip(row.color, row.all_label) if c == "r")
        ax.set_title(f"{row.filename} -> {red}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.EDA_DIR / "samples.png", dpi=160)
    plt.close(fig)


def _print_submission_empty_line_bytes() -> None:
    with config.SUBMISSION_SAMPLE.open("rb") as fh:
        for raw in fh:
            stripped = raw.rstrip(b"\r\n")
            if stripped.endswith(b","):
                print("empty label raw bytes:", stripped)
                return
    print("empty label raw bytes: NOT_FOUND")


def main() -> None:
    df = pd.read_csv(config.TRAIN_LABELS, dtype=str, keep_default_na=False)
    _assert_labels(df)
    _assert_images(df)

    color_counts = Counter("".join(df["color"].tolist()))
    red_by_position = [(df["color"].str[i] == "r").mean() for i in range(config.NUM_POSITIONS)]
    red_count_dist = df["color"].map(lambda value: value.count("r")).value_counts().sort_index()
    char_counts = Counter("".join(df["all_label"].tolist()))

    print("cuda_available:", torch.cuda.is_available())
    print("color_counts:", dict(color_counts))
    print("red_ratio_by_position:", [round(x, 4) for x in red_by_position])
    print("red_count_distribution:", red_count_dist.to_dict())
    print("char_frequency_minmax:", min(char_counts.values()), max(char_counts.values()))
    print("char_classes:", len(char_counts))
    _print_submission_empty_line_bytes()
    _save_sample_grid(df)
    print("eda passed:", config.EDA_DIR / "samples.png")


if __name__ == "__main__":
    main()
