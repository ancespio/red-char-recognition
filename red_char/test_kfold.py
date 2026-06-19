from __future__ import annotations

import unittest

from kfold import fold_split, kfold_assignment
from train import build_parser as build_train_parser
from train_glyph import build_parser as build_glyph_parser


class KFoldTests(unittest.TestCase):
    def test_kfold_assignment_is_deterministic_balanced_partition(self) -> None:
        first = kfold_assignment(103, n_folds=5, seed=1234)
        second = kfold_assignment(103, n_folds=5, seed=1234)

        self.assertEqual(first, second)
        self.assertEqual(sorted(set(first)), [0, 1, 2, 3, 4])
        sizes = [first.count(fold) for fold in range(5)]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_fold_split_partitions_indices(self) -> None:
        seen: set[int] = set()
        for fold in range(5):
            train_idx, val_idx = fold_split(103, fold=fold, n_folds=5, seed=1234)

            self.assertFalse(set(train_idx) & set(val_idx))
            self.assertEqual(len(train_idx) + len(val_idx), 103)
            seen.update(val_idx)

        self.assertEqual(seen, set(range(103)))

    def test_training_parsers_accept_fold_options(self) -> None:
        train_args = build_train_parser().parse_args(["--fold", "2", "--n-folds", "5"])
        glyph_args = build_glyph_parser().parse_args(["--fold", "2", "--n-folds", "5"])

        self.assertEqual(train_args.fold, 2)
        self.assertEqual(train_args.n_folds, 5)
        self.assertEqual(glyph_args.fold, 2)
        self.assertEqual(glyph_args.n_folds, 5)


if __name__ == "__main__":
    unittest.main()
