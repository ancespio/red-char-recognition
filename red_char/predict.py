from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import RedCharDataset, Sample, decode_prediction, load_submission_sample
from model import RedCharNet


def load_model(checkpoint: Path, device: torch.device) -> RedCharNet:
    model = RedCharNet().to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_labels(model: RedCharNet, loader: DataLoader, device: torch.device) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    labels: list[str] = []
    for images, filenames in tqdm(loader, desc="predict"):
        images = images.to(device, non_blocking=True)
        char_logits, color_logits = model(images)
        char_pred = char_logits.argmax(dim=-1).cpu().tolist()
        color_pred = color_logits.argmax(dim=-1).cpu().tolist()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "submission.csv")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    sample = load_submission_sample()
    samples = [Sample(row.id) for row in sample.itertuples(index=False)]
    dataset = RedCharDataset(samples, config.TEST_IMAGES, is_test=True, cache_in_ram=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = load_model(args.checkpoint, device)
    ids, labels = predict_labels(model, loader, device)
    write_submission(ids, labels, args.output)
    validate_submission(args.output)
    print("submission written:", args.output)


if __name__ == "__main__":
    main()
