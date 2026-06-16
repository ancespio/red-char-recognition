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
from model import build_model


def load_model(checkpoint: Path, device: torch.device, use_ema: bool = True):
    payload = torch.load(checkpoint, map_location=device)
    model = build_model(payload.get("model_name", config.MODEL)).to(device)
    state = payload.get("ema_state_dict") if use_ema else None
    if state is None:
        state = payload["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_labels(models: list, loader: DataLoader, device: torch.device,
                   red_threshold: float = 0.5) -> tuple[list[str], list[str]]:
    """Ensemble prediction: average softmax probabilities across models.

    Averaging probabilities (not logits) over an ensemble of independently
    trained checkpoints is the safe, reliable accuracy boost for this task.
    Geometric test-time augmentation is intentionally avoided because any
    horizontal shift would misalign the per-position character heads.

    ``red_threshold`` is the minimum averaged P(red) for a position to count as
    red. The colour head has a small systematic miss-red (r->u) bias on
    ambiguous glyphs, so a threshold slightly below 0.5 recovers missed reds;
    0.40 was the start of a wide, stable plateau on the validation split.
    """
    ids: list[str] = []
    labels: list[str] = []
    for images, filenames in tqdm(loader, desc="predict"):
        images = images.to(device, non_blocking=True)
        char_prob = None
        color_prob = None
        for model in models:
            char_logits, color_logits = model(images)
            cp = F.softmax(char_logits, dim=-1)
            kp = F.softmax(color_logits, dim=-1)
            char_prob = cp if char_prob is None else char_prob + cp
            color_prob = kp if color_prob is None else color_prob + kp
        color_prob = color_prob / len(models)
        char_pred = char_prob.argmax(dim=-1).cpu().tolist()
        is_red = color_prob[..., config.RED_INDEX] >= red_threshold
        color_pred = torch.where(is_red, config.RED_INDEX, config.NON_RED_INDEX).cpu().tolist()
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
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=[config.CHECKPOINT_DIR / "best.pt"],
        help="one or more checkpoints to ensemble (softmax averaged)",
    )
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "submission.csv")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True,
                        help="load EMA weights when present in the checkpoint")
    parser.add_argument("--red-threshold", type=float, default=0.5,
                        help="min averaged P(red) to call a position red (0.40 recovers missed reds)")
    args = parser.parse_args()

    config.ensure_output_dirs()
    device = torch.device(config.DEVICE)
    sample = load_submission_sample()
    samples = [Sample(row.id) for row in sample.itertuples(index=False)]
    dataset = RedCharDataset(samples, config.TEST_IMAGES, is_test=True, cache_in_ram=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    models = [load_model(ckpt, device, use_ema=args.use_ema) for ckpt in args.checkpoints]
    print(f"ensembling {len(models)} checkpoint(s); use_ema={args.use_ema}")
    ids, labels = predict_labels(models, loader, device, red_threshold=args.red_threshold)
    write_submission(ids, labels, args.output)
    validate_submission(args.output)
    print("submission written:", args.output)


if __name__ == "__main__":
    main()
