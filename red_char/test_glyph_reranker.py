from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import config
from model import count_parameters


class GlyphRerankerTests(unittest.TestCase):
    def test_extract_glyph_crops_returns_one_crop_per_position(self) -> None:
        from glyph import GLYPH_CROP_WIDTH, extract_glyph_crops

        images = torch.rand(2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

        crops = extract_glyph_crops(images)

        self.assertEqual(crops.shape, (2, config.NUM_POSITIONS, 3, config.IMAGE_HEIGHT, GLYPH_CROP_WIDTH))

    def test_extract_single_glyph_crop_matches_batched_crop(self) -> None:
        from glyph import extract_glyph_crop, extract_glyph_crops

        image = torch.rand(3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

        single = extract_glyph_crop(image, position=2, crop_width=72)
        batched = extract_glyph_crops(image, crop_width=72)[0, 2]

        self.assertTrue(torch.equal(single, batched))

    def test_glyph_model_preserves_classifier_shape(self) -> None:
        from glyph import GLYPH_CROP_WIDTH, GlyphNet

        model = GlyphNet(crop_width=GLYPH_CROP_WIDTH)
        crops = torch.rand(4, 3, config.IMAGE_HEIGHT, GLYPH_CROP_WIDTH)

        logits = model(crops)

        self.assertEqual(logits.shape, (4, config.NUM_CHARS))
        self.assertGreater(count_parameters(model), 100_000)

    def test_load_glyph_model_uses_checkpoint_options(self) -> None:
        from glyph import GlyphNet, load_glyph_model

        model = GlyphNet(input_mode="red", hires=True, crop_width=72, head_mode="gap")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "glyph.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_mode": "red",
                    "hires": True,
                    "crop_width": 72,
                    "head_mode": "gap",
                },
                checkpoint,
            )

            loaded = load_glyph_model(checkpoint, torch.device("cpu"), use_ema=True)

        self.assertEqual(loaded.input_mode, "red")
        self.assertTrue(loaded.hires)
        self.assertEqual(loaded.crop_width, 72)
        self.assertEqual(loaded.head_mode, "gap")

    def test_rerank_combines_primary_and_glyph_probabilities(self) -> None:
        from eval_reranker import rerank

        primary = torch.full((1, config.NUM_POSITIONS, config.NUM_CHARS), 0.001)
        glyph = torch.full_like(primary, 0.001)
        primary[..., config.CHAR_TO_IDX["A"]] = 0.60
        primary[..., config.CHAR_TO_IDX["B"]] = 0.40
        glyph[..., config.CHAR_TO_IDX["A"]] = 0.10
        glyph[..., config.CHAR_TO_IDX["B"]] = 0.90

        pred = rerank(primary, glyph, top_k=2, alpha=1.0)

        self.assertTrue(pred.eq(config.CHAR_TO_IDX["B"]).all())

    def test_selective_rerank_only_overrides_low_margin_primary(self) -> None:
        from eval_reranker import selective_rerank

        primary = torch.full((1, config.NUM_POSITIONS, config.NUM_CHARS), 0.001)
        glyph = torch.full_like(primary, 0.001)
        primary[..., config.CHAR_TO_IDX["A"]] = 0.55
        primary[..., config.CHAR_TO_IDX["B"]] = 0.45
        glyph[..., config.CHAR_TO_IDX["A"]] = 0.20
        glyph[..., config.CHAR_TO_IDX["B"]] = 0.80

        override = selective_rerank(primary, glyph, top_k=2, primary_margin_max=0.11, glyph_margin_min=0.50)
        keep = selective_rerank(primary, glyph, top_k=2, primary_margin_max=0.05, glyph_margin_min=0.50)

        self.assertTrue(override.eq(config.CHAR_TO_IDX["B"]).all())
        self.assertTrue(keep.eq(config.CHAR_TO_IDX["A"]).all())

    def test_reranker_eval_parser_accepts_primary_weights(self) -> None:
        from eval_reranker import build_parser

        args = build_parser().parse_args(
            [
                "--checkpoints",
                "a.pt",
                "b.pt",
                "--glyph-checkpoints",
                "g.pt",
                "--char-weights",
                "0.2",
                "0.8",
                "--color-weights",
                "0.1",
                "0.9",
            ]
        )

        self.assertEqual(args.char_weights, [0.2, 0.8])
        self.assertEqual(args.color_weights, [0.1, 0.9])

    def test_train_glyph_parser_accepts_local_options(self) -> None:
        from train_glyph import build_parser

        args = build_parser().parse_args(
            [
                "--epochs",
                "3",
                "--seed",
                "63",
                "--run-name",
                "glyph_seed63",
                "--input-mode",
                "red",
                "--hires",
                "--head-mode",
                "gap",
                "--crop-width",
                "72",
                "--num-workers",
                "0",
                "--no-augment",
                "--resume",
                "runs/glyph/checkpoints/last.pt",
            ]
        )

        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.seed, 63)
        self.assertEqual(args.run_name, "glyph_seed63")
        self.assertEqual(args.input_mode, "red")
        self.assertTrue(args.hires)
        self.assertEqual(args.head_mode, "gap")
        self.assertEqual(args.crop_width, 72)
        self.assertEqual(args.num_workers, 0)
        self.assertFalse(args.augment)
        self.assertEqual(args.resume, Path("runs/glyph/checkpoints/last.pt"))

    def test_predict_reranker_parser_accepts_submission_options(self) -> None:
        from predict_reranker import build_parser

        args = build_parser().parse_args(
            [
                "--checkpoints",
                "a.pt",
                "b.pt",
                "--glyph-checkpoints",
                "g.pt",
                "--char-weights",
                "0.2",
                "0.8",
                "--color-weights",
                "0.1",
                "0.9",
                "--selective",
                "--primary-margin-max",
                "0.2",
                "--glyph-margin-min",
                "0.5",
                "--output",
                "submission.csv",
            ]
        )

        self.assertEqual(args.char_weights, [0.2, 0.8])
        self.assertEqual(args.color_weights, [0.1, 0.9])
        self.assertTrue(args.selective)
        self.assertEqual(str(args.output), "submission.csv")


if __name__ == "__main__":
    unittest.main()
