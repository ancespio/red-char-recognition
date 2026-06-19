from __future__ import annotations

import torch


KFOLD_SEED = 1234
N_FOLDS = 5


def kfold_assignment(n_items: int, n_folds: int = N_FOLDS, seed: int = KFOLD_SEED) -> list[int]:
    if n_items < 1:
        raise ValueError(f"n_items must be positive, got {n_items}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if n_folds > n_items:
        raise ValueError(f"n_folds cannot exceed n_items, got n_folds={n_folds} n_items={n_items}")

    perm = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed)).tolist()
    fold_of = [0] * n_items
    for rank, idx in enumerate(perm):
        fold_of[idx] = rank % n_folds
    return fold_of


def fold_split(n_items: int, fold: int, n_folds: int = N_FOLDS, seed: int = KFOLD_SEED) -> tuple[list[int], list[int]]:
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0,{n_folds}), got {fold}")

    fold_of = kfold_assignment(n_items=n_items, n_folds=n_folds, seed=seed)
    train_idx = [idx for idx, idx_fold in enumerate(fold_of) if idx_fold != fold]
    val_idx = [idx for idx, idx_fold in enumerate(fold_of) if idx_fold == fold]
    return train_idx, val_idx


if __name__ == "__main__":
    assignment = kfold_assignment(50000)
    sizes = [assignment.count(fold) for fold in range(N_FOLDS)]
    seen: set[int] = set()
    for fold in range(N_FOLDS):
        train_idx, val_idx = fold_split(50000, fold)
        assert not (set(train_idx) & set(val_idx))
        assert len(train_idx) + len(val_idx) == 50000
        seen.update(val_idx)
    assert seen == set(range(50000))
    print(f"kfold OK: folds={N_FOLDS} sizes={sizes} seed={KFOLD_SEED}")
