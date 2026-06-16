"""Deterministic K-fold split over the full 50000 training images.

Used to build out-of-fold (OOF) predictions: for each fold k, a model trains
on the other K-1 folds and predicts fold k. Concatenating all folds yields an
unbiased prediction for every training image, so the reranker can be tuned /
error-analysed on 50000 images instead of the tiny 2500 holdout.

The permutation is seeded independently of training seeds (like
``deterministic_split_indices``), so the fold assignment is fixed and shared by
every primary/glyph model and every evaluation script.
"""
from __future__ import annotations

import torch

import config

KFOLD_SEED = 1234  # fixed, independent of config.SEED and any training seed
N_FOLDS = 5


def kfold_assignment(n_items: int, n_folds: int = N_FOLDS, seed: int = KFOLD_SEED) -> list[int]:
    """Return a length-n_items list mapping each index -> its fold id [0, n_folds)."""
    perm = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed)).tolist()
    fold_of = [0] * n_items
    for rank, idx in enumerate(perm):
        fold_of[idx] = rank % n_folds
    return fold_of


def fold_split(n_items: int, fold: int, n_folds: int = N_FOLDS, seed: int = KFOLD_SEED) -> tuple[list[int], list[int]]:
    """Indices that TRAIN this fold's model (all other folds) and its OOF indices (this fold)."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0,{n_folds}), got {fold}")
    fold_of = kfold_assignment(n_items, n_folds, seed)
    train_idx = [i for i in range(n_items) if fold_of[i] != fold]
    oof_idx = [i for i in range(n_items) if fold_of[i] == fold]
    return sorted(train_idx), sorted(oof_idx)


if __name__ == "__main__":
    # sanity: folds partition the dataset exactly and are reproducible
    n = 50000
    fa = kfold_assignment(n)
    sizes = [fa.count(k) for k in range(N_FOLDS)]
    assert sum(sizes) == n
    seen: set[int] = set()
    for k in range(N_FOLDS):
        tr, oof = fold_split(n, k)
        assert len(tr) + len(oof) == n
        assert not (set(tr) & set(oof))
        seen |= set(oof)
    assert seen == set(range(n))
    assert kfold_assignment(n) == fa  # deterministic
    print(f"kfold OK: {N_FOLDS} folds, sizes={sizes}, seed={KFOLD_SEED}")
