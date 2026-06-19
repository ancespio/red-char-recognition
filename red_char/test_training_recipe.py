from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.optim import AdamW

import config
from metrics import compute_loss
import train


class TrainingRecipeTests(unittest.TestCase):
    def test_char_label_smoothing_preserves_red_char_weighting(self) -> None:
        self.assertIn("label_smoothing", compute_loss.__code__.co_varnames)
        char_logits = torch.tensor(
            [
                [[3.0, 0.0] + [-4.0] * (config.NUM_CHARS - 2)],
                [[0.0, 3.0] + [-4.0] * (config.NUM_CHARS - 2)],
            ]
        )
        color_logits = torch.tensor([[[3.0, 0.0]], [[0.0, 3.0]]])
        char_targets = torch.tensor([[0], [1]])
        color_targets = torch.tensor([[config.NON_RED_INDEX], [config.RED_INDEX]])

        actual = compute_loss(
            char_logits,
            color_logits,
            char_targets,
            color_targets,
            red_char_weight=3.0,
            label_smoothing=0.1,
        )

        char_losses = torch.nn.functional.cross_entropy(
            char_logits.reshape(-1, config.NUM_CHARS),
            char_targets.reshape(-1),
            reduction="none",
            label_smoothing=0.1,
        ).view_as(char_targets)
        weights = torch.tensor([[1.0], [3.0]])
        expected_char = (char_losses * weights).sum() / weights.sum()
        expected_color = torch.nn.functional.cross_entropy(
            color_logits.reshape(-1, config.NUM_COLORS),
            color_targets.reshape(-1),
        )

        self.assertTrue(torch.allclose(actual, expected_char + expected_color))

    def test_build_scheduler_uses_linear_warmup_before_cosine(self) -> None:
        self.assertTrue(hasattr(train, "build_scheduler"))
        model = nn.Linear(2, 1)
        optimizer = AdamW(model.parameters(), lr=0.1)

        scheduler = train.build_scheduler(optimizer, epochs=5, warmup_epochs=2, eta_min=0.001)

        first_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        second_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        third_lr = optimizer.param_groups[0]["lr"]

        self.assertLess(first_lr, 0.1)
        self.assertGreater(second_lr, first_lr)
        self.assertAlmostEqual(third_lr, 0.1)

    def test_model_ema_updates_float_state_and_copies_integer_buffers(self) -> None:
        self.assertTrue(hasattr(train, "ModelEMA"))
        model = nn.BatchNorm1d(2)
        ema = train.ModelEMA(model, decay=0.5)

        with torch.no_grad():
            model.weight.fill_(3.0)
            model.running_mean.fill_(4.0)
            model.num_batches_tracked.fill_(7)
        ema.update(model)

        self.assertTrue(torch.allclose(ema.module.weight, torch.full_like(ema.module.weight, 2.0)))
        self.assertTrue(torch.allclose(ema.module.running_mean, torch.full_like(ema.module.running_mean, 2.0)))
        self.assertEqual(int(ema.module.num_batches_tracked), 7)


if __name__ == "__main__":
    unittest.main()
