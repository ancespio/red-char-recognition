from __future__ import annotations

import unittest

import torch
from torch.utils.data import Dataset

from dataset import TrainAugmentation, TransformSubset


class _SingleSampleDataset(Dataset):
    def __init__(self) -> None:
        self.image = torch.linspace(0, 1, 3 * 60 * 200).reshape(3, 60, 200)
        self.char_target = torch.tensor([1, 2, 3, 4, 5])
        self.color_target = torch.tensor([0, 1, 0, 1, 0])

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return self.image, self.char_target, self.color_target


class AugmentationTests(unittest.TestCase):
    def test_train_augmentation_preserves_shape_dtype_and_range(self) -> None:
        torch.manual_seed(42)
        image = torch.rand(3, 60, 200)

        augmented = TrainAugmentation()(image)

        self.assertEqual(augmented.shape, image.shape)
        self.assertEqual(augmented.dtype, image.dtype)
        self.assertGreaterEqual(float(augmented.min()), 0.0)
        self.assertLessEqual(float(augmented.max()), 1.0)
        self.assertFalse(torch.equal(augmented, image))

    def test_transform_subset_changes_only_image_and_keeps_targets(self) -> None:
        base = _SingleSampleDataset()
        subset = TransformSubset(base, [0], transform=lambda image: torch.zeros_like(image))

        image, char_target, color_target = subset[0]

        self.assertTrue(torch.equal(image, torch.zeros_like(image)))
        self.assertTrue(torch.equal(char_target, base.char_target))
        self.assertTrue(torch.equal(color_target, base.color_target))

    def test_transform_subset_without_transform_returns_unmodified_image(self) -> None:
        base = _SingleSampleDataset()
        subset = TransformSubset(base, [0], transform=None)

        image, _, _ = subset[0]

        self.assertTrue(torch.equal(image, base.image))


if __name__ == "__main__":
    unittest.main()
