from __future__ import annotations

import unittest
from pathlib import Path

import torch

import config
from oof_predict import build_parser as build_oof_parser, format_fold_paths
from tune_oof import color_pred_at, exact_count, tune_selective


class OofToolTests(unittest.TestCase):
    def test_format_fold_paths_expands_fold_placeholder(self) -> None:
        paths = format_fold_paths(["outputs/runs/oof_f{fold}_s1/checkpoints/best.pt"], fold=3)

        self.assertEqual(paths, [Path("outputs/runs/oof_f3_s1/checkpoints/best.pt")])

    def test_oof_parser_accepts_patterns_and_output(self) -> None:
        args = build_oof_parser().parse_args(
            [
                "--primary-patterns",
                "outputs/runs/oof_f{fold}_s1/checkpoints/best.pt",
                "--glyph-patterns",
                "outputs/runs/glyph_f{fold}/checkpoints/best.pt",
                "--n-folds",
                "5",
                "--x-tta",
                "--out",
                "outputs/oof/oof.pt",
            ]
        )

        self.assertEqual(args.primary_patterns, ["outputs/runs/oof_f{fold}_s1/checkpoints/best.pt"])
        self.assertEqual(args.glyph_patterns, ["outputs/runs/glyph_f{fold}/checkpoints/best.pt"])
        self.assertEqual(args.n_folds, 5)
        self.assertTrue(args.x_tta)
        self.assertEqual(args.out, Path("outputs/oof/oof.pt"))

    def test_exact_count_respects_red_sequence_predictions(self) -> None:
        char_target = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        color_target = torch.tensor(
            [
                [config.RED_INDEX, config.NON_RED_INDEX, config.RED_INDEX, config.NON_RED_INDEX, config.NON_RED_INDEX],
                [config.NON_RED_INDEX, config.RED_INDEX, config.NON_RED_INDEX, config.NON_RED_INDEX, config.NON_RED_INDEX],
            ]
        )
        char_pred = char_target.clone()
        color_pred = color_target.clone()
        color_pred[1, 0] = config.RED_INDEX

        self.assertEqual(exact_count(char_pred, color_pred, char_target, color_target), 1)

    def test_tune_selective_finds_glyph_override_thresholds(self) -> None:
        primary = torch.full((1, config.NUM_POSITIONS, config.NUM_CHARS), 0.001)
        glyph = torch.full_like(primary, 0.001)
        color = torch.zeros(1, config.NUM_POSITIONS, config.NUM_COLORS)
        char_target = torch.zeros(1, config.NUM_POSITIONS, dtype=torch.long)
        color_target = torch.zeros(1, config.NUM_POSITIONS, dtype=torch.long)
        color_target[0, 0] = config.RED_INDEX
        color[0, :, config.NON_RED_INDEX] = 0.9
        color[0, :, config.RED_INDEX] = 0.1
        color[0, 0, config.RED_INDEX] = 0.9
        color[0, 0, config.NON_RED_INDEX] = 0.1

        primary[0, 0, 1] = 0.51
        primary[0, 0, 0] = 0.49
        glyph[0, 0, 0] = 0.95
        glyph[0, 0, 1] = 0.05

        result = tune_selective(
            primary,
            glyph,
            color,
            char_target,
            color_target,
            top_k=2,
            primary_margins=[0.05],
            glyph_margins=[0.50],
            red_thresholds=[0.50],
        )

        self.assertEqual(result.exact, 1)
        self.assertEqual(result.primary_margin_max, 0.05)
        self.assertEqual(result.glyph_margin_min, 0.50)
        self.assertTrue(torch.equal(color_pred_at(color, 0.50), color_target))


if __name__ == "__main__":
    unittest.main()
