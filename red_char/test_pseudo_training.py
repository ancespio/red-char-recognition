from __future__ import annotations

import unittest
from pathlib import Path

import torch

import config
from train_pseudo import build_parser, select_confident_pseudo_samples


class PseudoTrainingTests(unittest.TestCase):
    def test_select_confident_pseudo_samples_keeps_only_all_position_confident_rows(self) -> None:
        filenames = ["keep.png", "drop.png"]
        char_prob = torch.full((2, config.NUM_POSITIONS, config.NUM_CHARS), 0.001)
        color_prob = torch.full((2, config.NUM_POSITIONS, config.NUM_COLORS), 0.001)
        for pos, ch in enumerate("ABCDE"):
            char_prob[0, pos, config.CHAR_TO_IDX[ch]] = 0.98
            color_prob[0, pos, config.RED_INDEX if pos % 2 == 0 else config.NON_RED_INDEX] = 0.97
        char_prob[1, :, config.CHAR_TO_IDX["Z"]] = 0.99
        color_prob[1, :, config.RED_INDEX] = 0.99
        char_prob[1, 3, config.CHAR_TO_IDX["Z"]] = 0.02
        char_prob = char_prob / char_prob.sum(dim=-1, keepdim=True)
        color_prob = color_prob / color_prob.sum(dim=-1, keepdim=True)

        samples = select_confident_pseudo_samples(filenames, char_prob, color_prob, char_threshold=0.90, color_threshold=0.90)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].filename, "keep.png")
        self.assertEqual(samples[0].all_label, "ABCDE")
        self.assertEqual(samples[0].color, "rurur")

    def test_train_pseudo_parser_accepts_local_options(self) -> None:
        args = build_parser().parse_args(
            [
                "--teacher-checkpoints",
                "a.pt",
                "b.pt",
                "--char-weights",
                "0.25",
                "0.75",
                "--color-weights",
                "0.4",
                "0.6",
                "--char-threshold",
                "0.92",
                "--color-threshold",
                "0.90",
                "--epochs",
                "3",
                "--seed",
                "70",
                "--run-name",
                "pseudo_seed70",
                "--model-size",
                "v2hi",
                "--resume",
                "runs/pseudo/checkpoints/last.pt",
                "--init-checkpoint",
                "runs/v2hi/checkpoints/best.pt",
                "--lr",
                "0.0002",
                "--num-workers",
                "0",
            ]
        )

        self.assertEqual(args.teacher_checkpoints, [Path("a.pt"), Path("b.pt")])
        self.assertEqual(args.char_weights, [0.25, 0.75])
        self.assertEqual(args.color_weights, [0.4, 0.6])
        self.assertEqual(args.char_threshold, 0.92)
        self.assertEqual(args.color_threshold, 0.90)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.seed, 70)
        self.assertEqual(args.run_name, "pseudo_seed70")
        self.assertEqual(args.model_size, "v2hi")
        self.assertEqual(args.resume, Path("runs/pseudo/checkpoints/last.pt"))
        self.assertEqual(args.init_checkpoint, Path("runs/v2hi/checkpoints/best.pt"))
        self.assertEqual(args.lr, 0.0002)
        self.assertEqual(args.num_workers, 0)


if __name__ == "__main__":
    unittest.main()
