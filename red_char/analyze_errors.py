"""Detailed validation-error analysis for an ensemble.

Dumps every mis-predicted val image with per-position confidences and
categorises the failures (char vs colour, which confusion pairs, which
position, and how confident the model was when wrong -> is it fixable or a
genuinely ambiguous glyph?).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import config
from dataset import build_train_dataset, deterministic_split_indices, decode_prediction
from predict import load_model


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dump", type=Path, default=config.EDA_DIR / "val_errors_ens.csv")
    args = p.parse_args()

    device = torch.device(config.DEVICE)
    base = build_train_dataset(cache_in_ram=False)
    _, val_idx = deterministic_split_indices(len(base))
    val_names = [base.samples[i].filename for i in val_idx]
    val_ds = Subset(base, val_idx)
    loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    models = [load_model(c, device, use_ema=args.use_ema) for c in args.checkpoints]
    print(f"ensemble of {len(models)} on {len(val_ds)} val images")

    CH = config.CHARSET
    char_pairs = Counter()   # true_char -> pred_char (char mistakes)
    pos_err = Counter()
    type_cnt = Counter()
    color_pairs = Counter()  # (true r/u -> pred r/u)
    err_rows = []
    n = exact = 0
    cursor = 0
    for images, char_t, color_t in loader:
        images = images.to(device)
        cp = kp = None
        for m in models:
            cl, kl = m(images)
            a, b = F.softmax(cl, -1), F.softmax(kl, -1)
            cp = a if cp is None else cp + a
            kp = b if kp is None else kp + b
        cp /= len(models); kp /= len(models)
        cconf, cidx = cp.max(-1)
        kconf, kidx = kp.max(-1)
        cidx = cidx.cpu(); kidx = kidx.cpu(); cconf = cconf.cpu(); kconf = kconf.cpu()
        bs = images.size(0)
        for i in range(bs):
            fname = val_names[cursor + i]
            true_s = decode_prediction(char_t[i].tolist(), color_t[i].tolist())
            pred_s = decode_prediction(cidx[i].tolist(), kidx[i].tolist())
            n += 1
            if pred_s == true_s:
                exact += 1
                continue
            char_wrong = color_wrong = False
            details = []
            for pos in range(config.NUM_POSITIONS):
                tc, pc = int(char_t[i][pos]), int(cidx[i][pos])
                tk, pk = int(color_t[i][pos]), int(kidx[i][pos])
                if tc != pc:
                    char_wrong = True
                    char_pairs[(CH[tc], CH[pc])] += 1
                    pos_err[pos] += 1
                    details.append(f"p{pos}:char {CH[tc]}->{CH[pc]}@{cconf[i][pos]:.2f}")
                if tk != pk:
                    color_wrong = True
                    tr = "r" if tk == config.RED_INDEX else "u"
                    pr = "r" if pk == config.RED_INDEX else "u"
                    color_pairs[(tr, pr)] += 1
                    details.append(f"p{pos}:col {tr}->{pr}@{kconf[i][pos]:.2f}")
            etype = "both" if char_wrong and color_wrong else ("char" if char_wrong else "color")
            type_cnt[etype] += 1
            err_rows.append((fname, etype, true_s, pred_s, " ".join(details)))
        cursor += bs

    print(f"\nval exact={exact/n:.4f}  errors={len(err_rows)}/{n}")
    print("error types:", dict(type_cnt))
    print("char confusion (true->pred):", char_pairs.most_common(20))
    print("color confusion:", dict(color_pairs))
    print("char-error position counts:", dict(sorted(pos_err.items())))

    # confidence breakdown of the WRONG char prediction
    args.dump.parent.mkdir(parents=True, exist_ok=True)
    with args.dump.open("w", encoding="utf-8") as f:
        f.write("filename,type,true,pred,details\n")
        for r in err_rows:
            f.write(",".join(r) + "\n")
    print("dumped:", args.dump)


if __name__ == "__main__":
    main()
