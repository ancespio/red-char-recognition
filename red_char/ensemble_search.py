from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import config
from evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every non-empty checkpoint combination.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=config.LOG_DIR / "ensemble_search.csv")
    args = parser.parse_args()

    rows = []
    for size in range(1, len(args.checkpoints) + 1):
        for combo in itertools.combinations(args.checkpoints, size):
            print("\nEvaluating:", " + ".join(str(path) for path in combo))
            metrics = evaluate(list(combo), export_errors=False)
            rows.append(
                {
                    "size": size,
                    "checkpoints": "|".join(str(path) for path in combo),
                    **metrics,
                }
            )

    rows.sort(key=lambda row: (row["exact"], row["char_acc"], -row["val_loss"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("\nBest combination:")
    print(rows[0])
    print("Search log:", args.output)


if __name__ == "__main__":
    main()
