from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import config
from beam_weight_search import build_parser as build_beam_parser, generate_local_weight_vectors, keep_top_rows
from dataset import deterministic_split_indices
from ensemble import average_model_logits, load_models, normalize_weights
from evaluate import build_parser as build_evaluate_parser
from metrics import compute_loss, encode_red_sequences
from model import build_model, count_parameters
from predict import build_parser as build_predict_parser
from train import build_parser as build_train_parser, resolve_run_paths
from tta import translate_images, tta_views
from weighted_ensemble_search import generate_weight_vectors


class _FixedModel(torch.nn.Module):
    def __init__(self, char_value: float, color_value: float) -> None:
        super().__init__()
        self.char_value = char_value
        self.color_value = color_value

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = images.size(0)
        char = torch.full((batch, config.NUM_POSITIONS, config.NUM_CHARS), self.char_value)
        color = torch.full((batch, config.NUM_POSITIONS, config.NUM_COLORS), self.color_value)
        return char, color


class _MeanModel(torch.nn.Module):
    def __init__(self, char_scale: float, color_scale: float) -> None:
        super().__init__()
        self.char_scale = char_scale
        self.color_scale = color_scale

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = images.mean(dim=(1, 2, 3))
        char = values[:, None, None].expand(-1, config.NUM_POSITIONS, config.NUM_CHARS) * self.char_scale
        color = values[:, None, None].expand(-1, config.NUM_POSITIONS, config.NUM_COLORS) * self.color_scale
        return char, color


class EnsembleTests(unittest.TestCase):
    def test_beam_parser_accepts_local_search_options(self) -> None:
        args = build_beam_parser().parse_args(
            [
                "--checkpoints",
                "a.pt",
                "b.pt",
                "--coarse-step",
                "0.1",
                "--fine-step",
                "0.02",
                "--radius",
                "0.08",
                "--top-k",
                "7",
            ]
        )

        self.assertEqual(len(args.checkpoints), 2)
        self.assertEqual(args.coarse_step, 0.1)
        self.assertEqual(args.fine_step, 0.02)
        self.assertEqual(args.radius, 0.08)
        self.assertEqual(args.top_k, 7)

    def test_local_weight_vectors_stay_normalized_and_near_center(self) -> None:
        center = (0.2, 0.3, 0.5)

        vectors = generate_local_weight_vectors(center=center, step=0.1, radius=0.1)

        self.assertIn(center, vectors)
        self.assertNotIn((1.0, 0.0, 0.0), vectors)
        self.assertTrue(all(abs(sum(vector) - 1.0) < 1e-9 for vector in vectors))
        self.assertTrue(all(max(abs(a - b) for a, b in zip(vector, center)) <= 0.1 + 1e-9 for vector in vectors))

    def test_keep_top_rows_keeps_highest_scoring_rows(self) -> None:
        rows = [
            {"exact": 0.8, "char_acc": 0.9, "color_acc": 1.0, "joint_pos_acc": 0.9},
            {"exact": 0.9, "char_acc": 0.8, "color_acc": 1.0, "joint_pos_acc": 0.8},
            {"exact": 0.9, "char_acc": 0.95, "color_acc": 0.9, "joint_pos_acc": 0.85},
        ]

        top_rows = keep_top_rows(rows, top_k=2)

        self.assertEqual(top_rows[0]["char_acc"], 0.95)
        self.assertEqual(top_rows[1]["exact"], 0.9)
        self.assertEqual(len(top_rows), 2)

    def test_evaluate_and_predict_parsers_accept_tta_flag(self) -> None:
        self.assertTrue(build_evaluate_parser().parse_args(["--tta"]).tta)
        self.assertTrue(build_predict_parser().parse_args(["--tta"]).tta)

    def test_train_parser_accepts_red_char_weight(self) -> None:
        args = build_train_parser().parse_args(["--red-char-weight", "2.5"])

        self.assertEqual(args.red_char_weight, 2.5)

    def test_train_parser_accepts_wide_model_size(self) -> None:
        args = build_train_parser().parse_args(["--model-size", "wide"])

        self.assertEqual(args.model_size, "wide")

    def test_train_parser_accepts_stage2_model_sizes(self) -> None:
        for model_size in ("base", "wide", "k5", "resblock", "deep3"):
            with self.subTest(model_size=model_size):
                args = build_train_parser().parse_args(["--model-size", model_size])

                self.assertEqual(args.model_size, model_size)

    def test_train_parser_accepts_augment_presets(self) -> None:
        for preset in ("light", "medium", "strong"):
            with self.subTest(preset=preset):
                args = build_train_parser().parse_args(["--augment-preset", preset])

                self.assertEqual(args.augment_preset, preset)

    def test_train_parser_accepts_num_workers_override(self) -> None:
        args = build_train_parser().parse_args(["--num-workers", "0"])

        self.assertEqual(args.num_workers, 0)

    def test_wide_model_preserves_output_shapes_and_adds_capacity(self) -> None:
        images = torch.randn(2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

        base = build_model("base")
        wide = build_model("wide")
        char_logits, color_logits = wide(images)

        self.assertEqual(char_logits.shape, (2, config.NUM_POSITIONS, config.NUM_CHARS))
        self.assertEqual(color_logits.shape, (2, config.NUM_POSITIONS, config.NUM_COLORS))
        self.assertGreater(count_parameters(wide), count_parameters(base))

    def test_stage2_model_sizes_preserve_output_shapes(self) -> None:
        images = torch.randn(2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

        for model_size in ("k5", "resblock", "deep3"):
            with self.subTest(model_size=model_size):
                model = build_model(model_size)
                char_logits, color_logits = model(images)

                self.assertEqual(char_logits.shape, (2, config.NUM_POSITIONS, config.NUM_CHARS))
                self.assertEqual(color_logits.shape, (2, config.NUM_POSITIONS, config.NUM_COLORS))
                self.assertGreater(count_parameters(model), 5_000_000)

    def test_load_models_uses_checkpoint_model_size(self) -> None:
        wide = build_model("wide")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "wide.pt"
            torch.save(
                {
                    "state_dict": wide.state_dict(),
                    "config": {"model_size": "wide"},
                },
                checkpoint,
            )

            models, payloads = load_models([checkpoint], torch.device("cpu"))

        self.assertEqual(count_parameters(models[0]), count_parameters(wide))
        self.assertEqual(payloads[0]["config"]["model_size"], "wide")

    def test_load_models_uses_stage2_checkpoint_model_size(self) -> None:
        model = build_model("resblock")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "resblock.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {"model_size": "resblock"},
                },
                checkpoint,
            )

            models, payloads = load_models([checkpoint], torch.device("cpu"))

        self.assertEqual(count_parameters(models[0]), count_parameters(model))
        self.assertEqual(payloads[0]["config"]["model_size"], "resblock")

    def test_red_char_weight_one_matches_legacy_loss(self) -> None:
        char_logits = torch.randn(2, config.NUM_POSITIONS, config.NUM_CHARS)
        color_logits = torch.randn(2, config.NUM_POSITIONS, config.NUM_COLORS)
        char_targets = torch.randint(0, config.NUM_CHARS, (2, config.NUM_POSITIONS))
        color_targets = torch.randint(0, config.NUM_COLORS, (2, config.NUM_POSITIONS))

        legacy_loss = compute_loss(char_logits, color_logits, char_targets, color_targets)
        weighted_loss = compute_loss(
            char_logits,
            color_logits,
            char_targets,
            color_targets,
            red_char_weight=1.0,
        )

        self.assertTrue(torch.allclose(legacy_loss, weighted_loss))

    def test_red_char_weight_emphasizes_red_position_character_loss(self) -> None:
        char_logits = torch.zeros(1, config.NUM_POSITIONS, config.NUM_CHARS)
        color_logits = torch.zeros(1, config.NUM_POSITIONS, config.NUM_COLORS)
        char_targets = torch.zeros(1, config.NUM_POSITIONS, dtype=torch.long)
        color_targets = torch.tensor([[config.RED_INDEX, 0, 0, 0, 0]])
        char_logits[0, 0, 0] = -8.0
        char_logits[0, 0, 1] = 8.0

        base_loss = compute_loss(
            char_logits,
            color_logits,
            char_targets,
            color_targets,
            color_weight=0.0,
            red_char_weight=1.0,
        )
        red_weighted_loss = compute_loss(
            char_logits,
            color_logits,
            char_targets,
            color_targets,
            color_weight=0.0,
            red_char_weight=3.0,
        )

        self.assertGreater(float(red_weighted_loss), float(base_loss))

    def test_translation_uses_white_fill_without_wrapping(self) -> None:
        images = torch.tensor([[[[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]]])

        translated = translate_images(images, dx=1, dy=0)

        expected = torch.tensor([[[[1.0, 0.0, 0.1, 0.2], [1.0, 0.4, 0.5, 0.6]]]])
        self.assertTrue(torch.equal(translated, expected))

    def test_tta_views_use_configured_five_offsets(self) -> None:
        images = torch.zeros(1, 1, 3, 4)

        views = tta_views(images)

        self.assertEqual(len(views), 5)
        self.assertTrue(torch.equal(views[0], images))
        self.assertTrue(torch.all(views[1][..., -2:] == 1.0))
        self.assertTrue(torch.all(views[2][..., :2] == 1.0))
        self.assertTrue(torch.all(views[3][..., -1:, :] == 1.0))
        self.assertTrue(torch.all(views[4][..., :1, :] == 1.0))

    def test_training_seed_does_not_change_fixed_validation_split(self) -> None:
        _, val_a = deterministic_split_indices(50000, seed=config.SPLIT_SEED)
        torch.manual_seed(43)
        _, val_b = deterministic_split_indices(50000, seed=config.SPLIT_SEED)

        self.assertEqual(val_a, val_b)

    def test_average_model_logits_averages_before_prediction(self) -> None:
        images = torch.zeros(2, 3, 60, 200)
        models = [_FixedModel(1.0, 2.0), _FixedModel(3.0, 4.0)]

        char_logits, color_logits = average_model_logits(models, images)

        self.assertTrue(torch.all(char_logits == 2.0))
        self.assertTrue(torch.all(color_logits == 3.0))

    def test_weighted_model_logits_use_separate_char_and_color_weights(self) -> None:
        images = torch.zeros(1, 3, 60, 200)
        models = [_FixedModel(1.0, 2.0), _FixedModel(3.0, 4.0)]

        char_logits, color_logits = average_model_logits(
            models,
            images,
            char_weights=[3.0, 1.0],
            color_weights=[1.0, 3.0],
        )

        self.assertTrue(torch.all(char_logits == 1.5))
        self.assertTrue(torch.all(color_logits == 3.5))

    def test_tta_averages_views_before_model_weighting(self) -> None:
        images = torch.zeros(1, 1, 3, 4)
        models = [_MeanModel(1.0, 2.0), _MeanModel(3.0, 4.0)]

        char_logits, color_logits = average_model_logits(
            models,
            images,
            char_weights=[3.0, 1.0],
            color_weights=[1.0, 3.0],
            tta=True,
        )

        self.assertTrue(torch.allclose(char_logits, torch.full_like(char_logits, 0.5)))
        self.assertTrue(torch.allclose(color_logits, torch.full_like(color_logits, 7.0 / 6.0)))

    def test_normalize_weights_rejects_wrong_length_and_zero_sum(self) -> None:
        with self.assertRaises(ValueError):
            normalize_weights([1.0], model_count=2, label="char")
        with self.assertRaises(ValueError):
            normalize_weights([0.0, 0.0], model_count=2, label="color")

    def test_weight_grid_contains_all_three_model_compositions(self) -> None:
        vectors = generate_weight_vectors(model_count=3, units=2)

        self.assertEqual(len(vectors), 6)
        self.assertIn((1.0, 0.0, 0.0), vectors)
        self.assertIn((0.5, 0.5, 0.0), vectors)
        self.assertTrue(all(abs(sum(vector) - 1.0) < 1e-9 for vector in vectors))

    def test_red_sequence_encoding_distinguishes_order_and_empty(self) -> None:
        chars = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        colors = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]])

        codes = encode_red_sequences(chars, colors)

        self.assertNotEqual(int(codes[0]), int(codes[1]))
        self.assertEqual(int(codes[2]), 0)

    def test_named_run_uses_isolated_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)

            paths = resolve_run_paths("seed43", output_root=output_root)

            self.assertEqual(paths.checkpoint_dir, output_root / "runs" / "seed43" / "checkpoints")
            self.assertEqual(paths.log_path, output_root / "runs" / "seed43" / "logs" / "train_log.csv")

    def test_default_run_keeps_legacy_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)

            paths = resolve_run_paths(None, output_root=output_root)

            self.assertEqual(paths.checkpoint_dir, output_root / "checkpoints")
            self.assertEqual(paths.log_path, output_root / "logs" / "train_log.csv")


if __name__ == "__main__":
    unittest.main()
