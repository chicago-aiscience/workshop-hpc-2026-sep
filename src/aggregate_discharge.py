#!/usr/bin/env python3
"""Aggregate the per-shard prediction CSVs into one tidy discharge table.

The prediction stage runs as a job array, so each task leaves a
`predictions_XXX.csv` in the output dir. This stage concatenates them, sorts by
reach and date, and writes a single `discharge.csv` for scoring and plotting
(the analysis analogue of gathering a fan-out's outputs).

    python src/aggregate_discharge.py --pred-dir runs/predictions \
        --out runs/discharge.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for aggregation."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="dir with predictions_*.csv from the array")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> None:
    """Concatenate every prediction shard into a single sorted discharge table."""
    args = parse_args()
    logger = configure_logging("aggregate_discharge")
    for name, value in vars(args).items(): logger.info("%s = %s", name, value)

    indexes = sorted(Path(args.pred_dir).glob("predictions_*.csv"))
    if not indexes:
        raise SystemExit(f"[aggregate] no predictions_*.csv found in {args.pred_dir}")

    df = pd.concat((pd.read_csv(s) for s in indexes), ignore_index=True)
    df = df.sort_values(["reach_id", "date"]).reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    logger.info(f"[aggregate] {len(indexes)} indexes -> {len(df)} rows, "
          f"{df.reach_id.nunique()} reaches -> {args.out}")


if __name__ == "__main__":
    main()
