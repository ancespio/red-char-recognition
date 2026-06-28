from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import RedCharDataset, Sample, decode_prediction, load_submission_sample
from ensemble import average_model_logits, load_models
from eval_reranker import average_primary_logits


@torch.no_grad()
def predict_labels(
    models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    char_weights: list[float] | None = None,
    color_weights: list[float] | None = None,
    red_threshold: float | None = None,
    tta: bool = False,
    x_tta: bool = False,
) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    labels: list[str] = []
    for images, filenames in tqdm(loader, desc="predict"):
        images = images.to(device, non_blocking=True)
        if x_tta:
            char_logits, color_logits = average_primary_logits(
                models,
                images,
                char_weights=char_weights,
                color_weights=color_weights,
                x_tta=True,
            )
        else:
            char_logits, color_logits = average_model_logits(
                models,
                images,
                char_weights=char_weights,
                color_weights=color_weights,
                tta=tta,
            )
        char_pred = char_logits.argmax(dim=-1).cpu().tolist()
        if red_threshold is None:
            color_pred = color_logits.argmax(dim=-1).cpu().tolist()
        else:
            color_pred = F.softmax(color_logits, dim=-1)[..., config.RED_INDEX].ge(red_threshold).long().cpu().tolist()
        for filename, chars, colors in zip(filenames, char_pred, color_pred):
            ids.append(filename)
            labels.append(decode_prediction(chars, colors))
    return ids, labels


def write_submission(ids: list[str], labels: list[str], output_path: Path) -> None:
    df = pd.DataFrame({"id": ids, "label": labels})
    assert not df["label"].isna().any()
    df.to_csv(output_path, index=False, lineterminator="\n")


def validate_submission(output_path: Path) -> None:
    sample = load_submission_sample()
    df = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    assert list(df.columns) == ["id", "label"]
    assert len(df) == 5000
    assert df["id"].tolist() == sample["id"].tolist()
    assert df["label"].map(lambda x: set(x).issubset(set(config.CHARSET)) and len(x) <= 5).all()
    raw = output_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert b"nan" not in raw.lower()
    assert b'""' not in raw
    lines = raw.split(b"\n")
    assert len(lines) == 5001 + 1 and lines[0] == b"id,label" and lines[-1] == b""
    empty_lines = [line for line in lines[1:-1] if line.endswith(b",")]
    if empty_lines:
        print("empty label example bytes:", empty_lines[0])
    dist = Counter(df["label"].map(len))
    print("prediction length distribution:", dict(sorted(dist.items())))
    empty_count = dist.get(0, 0)
    five_count = dist.get(5, 0)
    if empty_count < 50 or empty_count > 300:
        print("warning: suspicious empty-label count; sample expectation is about 150")
    if five_count == 0 or five_count > 30:
        print("warning: suspicious length-5 count; sample expectation is about 4")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--checkpoints", type=Path, nargs="+")
    parser.add_argument("--char-weights", type=float, nargs="+")
    parser.add_argument("--color-weights", type=float, nargs="+")
    parser.add_argument("--red-threshold", type=float)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--x-tta", action="store_true")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "submission.csv")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    sample = load_submission_sample()
    samples = [Sample(row.id) for row in sample.itertuples(index=False)]
    dataset = RedCharDataset(samples, config.TEST_IMAGES, is_test=True, cache_in_ram=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    checkpoints = args.checkpoints or [args.checkpoint or config.CHECKPOINT_DIR / "best.pt"]
    models, _ = load_models(checkpoints, device)
    ids, labels = predict_labels(
        models,
        loader,
        device,
        char_weights=args.char_weights,
        color_weights=args.color_weights,
        red_threshold=args.red_threshold,
        tta=args.tta,
        x_tta=args.x_tta,
    )
    write_submission(ids, labels, args.output)
    validate_submission(args.output)
    print("submission written:", args.output)


if __name__ == "__main__":
    main()
