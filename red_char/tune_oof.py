from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

import config
from dataset import decode_prediction
from eval_reranker import selective_rerank


@dataclass(frozen=True)
class TuneResult:
    exact: int
    total: int
    primary_margin_max: float
    glyph_margin_min: float
    red_threshold: float
    char_pred: torch.Tensor
    color_pred: torch.Tensor

    @property
    def accuracy(self) -> float:
        return self.exact / self.total


def color_pred_at(color_prob: torch.Tensor, threshold: float) -> torch.Tensor:
    return color_prob[..., config.RED_INDEX].ge(threshold).long()


def exact_count(
    char_pred: torch.Tensor,
    color_pred: torch.Tensor,
    char_target: torch.Tensor,
    color_target: torch.Tensor,
) -> int:
    true_red = color_target == config.RED_INDEX
    pred_red = color_pred == config.RED_INDEX
    position_ok = torch.where(true_red, pred_red & (char_pred == char_target), ~pred_red)
    return int(position_ok.all(dim=1).sum().item())


def string_exact_count(
    char_pred: torch.Tensor,
    color_pred: torch.Tensor,
    char_target: torch.Tensor,
    color_target: torch.Tensor,
) -> int:
    total = 0
    for idx in range(char_pred.shape[0]):
        pred = decode_prediction(char_pred[idx].tolist(), color_pred[idx].tolist())
        target = decode_prediction(char_target[idx].tolist(), color_target[idx].tolist())
        total += int(pred == target)
    return total


def tune_selective(
    primary_prob: torch.Tensor,
    glyph_prob: torch.Tensor,
    color_prob: torch.Tensor,
    char_target: torch.Tensor,
    color_target: torch.Tensor,
    top_k: int,
    primary_margins: list[float],
    glyph_margins: list[float],
    red_thresholds: list[float],
) -> TuneResult:
    best: TuneResult | None = None
    total = int(char_target.shape[0])
    for primary_margin in primary_margins:
        for glyph_margin in glyph_margins:
            char_pred = selective_rerank(primary_prob, glyph_prob, top_k, primary_margin, glyph_margin)
            for red_threshold in red_thresholds:
                color_pred = color_pred_at(color_prob, red_threshold)
                exact = exact_count(char_pred, color_pred, char_target, color_target)
                if best is None or exact > best.exact:
                    best = TuneResult(
                        exact=exact,
                        total=total,
                        primary_margin_max=primary_margin,
                        glyph_margin_min=glyph_margin,
                        red_threshold=red_threshold,
                        char_pred=char_pred,
                        color_pred=color_pred,
                    )
    if best is None:
        raise ValueError("empty tuning grid")
    return best


def _grid(max_value: float, step: float) -> list[float]:
    count = int(round(max_value / step))
    return [round(idx * step, 10) for idx in range(count + 1)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, default=config.OUTPUT_DIR / "oof" / "oof.pt")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--primary-margin-max", type=float, default=1.0)
    parser.add_argument("--glyph-margin-max", type=float, default=1.0)
    parser.add_argument("--margin-step", type=float, default=0.05)
    parser.add_argument("--red-thresholds", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.50])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = torch.load(args.oof, map_location="cpu")
    primary = data["primary_char"]
    color = data["color"]
    char_target = data["char_target"]
    color_target = data["color_target"]
    total = int(char_target.shape[0])

    base_char = primary.argmax(dim=-1)
    base_best = max(
        (
            exact_count(base_char, color_pred_at(color, threshold), char_target, color_target),
            threshold,
        )
        for threshold in args.red_thresholds
    )
    print(f"baseline primary: exact={base_best[0]}/{total}={base_best[0] / total:.5f} red_threshold={base_best[1]:.2f}")

    if "glyph" not in data:
        verified = string_exact_count(base_char, color_pred_at(color, base_best[1]), char_target, color_target)
        print(f"baseline string-verified: exact={verified}/{total}={verified / total:.5f}")
        return

    result = tune_selective(
        primary,
        data["glyph"],
        color,
        char_target,
        color_target,
        top_k=args.top_k,
        primary_margins=_grid(args.primary_margin_max, args.margin_step),
        glyph_margins=_grid(args.glyph_margin_max, args.margin_step),
        red_thresholds=args.red_thresholds,
    )
    verified = string_exact_count(result.char_pred, result.color_pred, char_target, color_target)
    print(
        "selective rerank: "
        f"exact={result.exact}/{result.total}={result.accuracy:.5f} "
        f"string_verified={verified}/{result.total}={verified / result.total:.5f} "
        f"primary_margin<={result.primary_margin_max:.2f} "
        f"glyph_margin>={result.glyph_margin_min:.2f} "
        f"red_threshold={result.red_threshold:.2f}"
    )
    print(f"gain over baseline: +{result.exact - base_best[0]}/{total}")


if __name__ == "__main__":
    main()
