"""Tune the reranker on the full-50000 OOF predictions (unbiased, large sample).

Reports baseline (primary), selective-rerank exact-match, the best thresholds,
and a confusion breakdown of the residual errors. Because this uses 50000 OOF
images instead of the 2500 holdout, the numbers are statistically meaningful
(1 image = 0.002%, vs 0.04% on val) and the thresholds don't overfit a tiny set.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

import config
from dataset import decode_prediction
from eval_reranker import selective_rerank

CH = config.CHARSET


def exact_count(char_pred, color_pred, char_t, color_t) -> int:
    """Vectorised exact-match count.

    A position is OK iff: (true red -> predicted red AND char matches) or
    (true non-red -> predicted non-red). Sample correct iff all 5 OK. This
    equals decode-string equality except for pathological red-slot misalignments
    that coincidentally concatenate equal (negligible); verify_exact() below
    re-checks the chosen config with the true string compare.
    """
    red_t = color_t == config.RED_INDEX
    red_p = color_pred == config.RED_INDEX
    pos_ok = torch.where(red_t, red_p & (char_pred == char_t), ~red_p)
    return int(pos_ok.all(dim=1).sum())


def verify_exact(char_pred, color_pred, char_t, color_t) -> int:
    ok = 0
    for i in range(char_pred.shape[0]):
        if decode_prediction(char_pred[i].tolist(), color_pred[i].tolist()) == \
           decode_prediction(char_t[i].tolist(), color_t[i].tolist()):
            ok += 1
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oof", type=Path, default=config.OUTPUT_DIR / "oof" / "oof.pt")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--dump-errors", type=Path, default=config.EDA_DIR / "oof_errors.csv")
    args = p.parse_args()

    d = torch.load(args.oof)
    primary, glyph, color = d["primary_char"], d["glyph"], d["color"]
    char_t, color_t = d["char_target"], d["color_target"]
    n = primary.shape[0]
    red_mask = color_t.bool()
    print(f"OOF on {n} images (x_tta={d.get('x_tta')}, primary_tags={d.get('primary_tags')})")

    # red decision threshold tuned once (color comes from primary only)
    def color_pred_at(tau):
        return color[..., config.RED_INDEX].ge(tau).long()

    # baseline: primary argmax char + best red threshold
    base_char = primary.argmax(-1)
    base_best = (-1, None)
    for tau in (0.20, 0.30, 0.40, 0.50):
        e = exact_count(base_char, color_pred_at(tau), char_t, color_t)
        if e > base_best[0]:
            base_best = (e, tau)
    print(f"baseline primary: exact={base_best[0]}/{n}={base_best[0]/n:.5f} (red_tau={base_best[1]})")

    # selective rerank: grid search primary_margin_max x glyph_margin_min x red_tau
    best = (-1, None, None, None, None)
    for pm_i in range(1, 21):
        pm = pm_i / 20
        for gm_i in range(0, 21):
            gm = gm_i / 20
            char_pred = selective_rerank(primary, glyph, args.top_k, pm, gm)
            for tau in (0.20, 0.30, 0.40, 0.50):
                e = exact_count(char_pred, color_pred_at(tau), char_t, color_t)
                if e > best[0]:
                    best = (e, pm, gm, tau, char_pred)
    e, pm, gm, tau, char_pred = best
    verify = verify_exact(char_pred, color_pred_at(tau), char_t, color_t)
    print(f"selective rerank: exact={e}/{n}={e/n:.5f} (string-verified={verify}/{n}={verify/n:.5f}) "
          f"primary_margin<={pm:.2f} glyph_margin>={gm:.2f} red_tau={tau:.2f}")
    print(f"  gain over baseline: +{(e-base_best[0])}/{n} = +{(e-base_best[0])/n*100:.3f}%")

    # residual error analysis (char confusions on red positions)
    color_pred = color_pred_at(tau)
    pairs = Counter()
    rows = []
    for i in range(n):
        ts = decode_prediction(char_t[i].tolist(), color_t[i].tolist())
        ps = decode_prediction(char_pred[i].tolist(), color_pred[i].tolist())
        if ps == ts:
            continue
        det = []
        for pos in range(config.NUM_POSITIONS):
            if char_t[i, pos] != char_pred[i, pos] and color_t[i, pos] == config.RED_INDEX:
                pairs[(CH[char_t[i, pos]], CH[char_pred[i, pos]])] += 1
                det.append(f"p{pos}:{CH[char_t[i,pos]]}->{CH[char_pred[i,pos]]}")
            if color_t[i, pos] != color_pred[i, pos]:
                det.append(f"p{pos}:col{'ru'[color_t[i,pos]]}->{'ru'[color_pred[i,pos]]}")
        rows.append((i, ts, ps, " ".join(det)))
    print(f"residual errors: {len(rows)}/{n}")
    print("top char confusions (true->pred):", pairs.most_common(25))
    args.dump_errors.parent.mkdir(parents=True, exist_ok=True)
    with args.dump_errors.open("w", encoding="utf-8") as f:
        f.write("index,true,pred,details\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    print("dumped:", args.dump_errors)


if __name__ == "__main__":
    main()
