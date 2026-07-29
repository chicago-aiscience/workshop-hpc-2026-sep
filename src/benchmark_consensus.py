#!/usr/bin/env python3
"""Score the learned consensus against the gauge, versus the baselines.

Reads the aggregated discharge table and, on rows that have an observed gauge
value, computes standard hydrologic skill scores (NSE, KGE, RMSE, percent bias)
for three estimators:

  * `learned`    -- the trained MLP's predicted_q,
  * `naive_mean` -- the plain mean of the available FLPE algorithms,
  * `consensus`  -- Confluence's own consensus_q,

overall and per basin. This is where the workshop shows whether the model earned
its keep. Writes a tidy `scored.csv` (one row per scope x estimator).

    python src/benchmark_consensus.py --discharge runs/discharge.csv \
        --out runs/scored.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import configure_logging

FLPE_COLS = ["q_metroman", "q_momma", "q_neobam", "q_sic4dvar"]

logger = configure_logging("benchmark_consensus")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for scoring."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--discharge", required=True, help="aggregated discharge.csv")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def skill(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    """Compute NSE, KGE, RMSE, and percent bias on paired obs/sim discharge.

    Pairs with a missing or non-positive value on either side are dropped first.

    Args:
        obs: Observed gauge discharge.
        sim: Estimated discharge.

    Returns:
        Dict with `n`, `nse`, `kge`, `rmse`, `pbias` (NaN-safe: metrics are NaN if
        fewer than three valid pairs remain).
    """
    # Keep only paired rows valid on BOTH sides: finite and positive discharge.
    # `o`/`s` are the aligned observed/simulated values used by every metric below.
    m = np.isfinite(obs) & np.isfinite(sim) & (obs > 0) & (sim > 0)
    o, s = obs[m], sim[m]

    # Too few pairs to say anything meaningful (correlation/variance are unstable);
    # report the count and NaN out the metrics rather than emit noise.
    if len(o) < 3:
        return {"n": int(len(o)), "nse": np.nan, "kge": np.nan, "rmse": np.nan, "pbias": np.nan}

    # RMSE: root-mean-square error, in m^3/s. Typical error magnitude; 0 is perfect,
    # lower is better. Squaring penalizes large misses most.
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))

    # NSE (Nash-Sutcliffe): 1 - (error variance / observed variance). 1 = perfect,
    # 0 = no better than always predicting the observed mean, <0 = worse than the mean.
    nse = float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))

    # KGE (Kling-Gupta) decomposes skill into three parts, each ideally 1:
    r = float(np.corrcoef(o, s)[0, 1])                                #   correlation: timing/shape agreement
    alpha = float(s.std() / o.std()) if o.std() > 0 else np.nan       #   variability ratio: spread match
    beta = float(s.mean() / o.mean())                                 #   bias ratio: mean match
    # KGE = 1 - Euclidean distance of (r, alpha, beta) from the ideal (1, 1, 1).
    # 1 = perfect; higher is better.
    kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    # Percent bias: net over/under-estimation as a % of total observed flow.
    # 0 = unbiased; positive = over-predicts, negative = under-predicts.
    pbias = float(100 * (s.sum() - o.sum()) / o.sum())

    # `n` = number of valid pairs the scores were computed on (context for the rest).
    return {"n": int(len(o)), "nse": nse, "kge": kge, "rmse": rmse, "pbias": pbias}


def estimators(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the three discharge estimators as aligned arrays."""
    return {
        "learned": df["predicted_q"].to_numpy(dtype=float),
        "naive_mean": df[FLPE_COLS].mean(axis=1, skipna=True).to_numpy(dtype=float),
        "consensus": df["q_consensus"].to_numpy(dtype=float),
    }


def common_mask(obs: np.ndarray, est: dict[str, np.ndarray]) -> np.ndarray:
    """Boolean mask of rows valid (finite, positive) for the gauge AND every estimator.

    Restricting all estimators to these shared rows keeps the comparison
    apples-to-apples rather than scoring each method on its own easiest subset.
    """
    mask = np.isfinite(obs) & (obs > 0)
    for sim in est.values():
        mask &= np.isfinite(sim) & (sim > 0)
    return mask


def score_estimators(obs: np.ndarray, est: dict[str, np.ndarray],
                     basin: np.ndarray, common: np.ndarray) -> pd.DataFrame:
    """Score every estimator overall and per basin -> one row per (scope, estimator).

    Args:
        obs: Observed gauge discharge.
        est: Estimator name -> discharge array (all aligned with `obs`).
        basin: Per-row basin id (used to build the per-basin scopes).
        common: Shared valid-row mask from `common_mask`.

    Returns:
        DataFrame with columns `scope, estimator, n, nse, kge, rmse, pbias`.
    """
    scopes = [("overall", common)]
    scopes += [(f"basin_{b}", common & (basin == b)) for b in sorted(np.unique(basin))]
    rows = [
        {"scope": scope, "estimator": name, **skill(obs[mask], sim[mask])}
        for scope, mask in scopes
        for name, sim in est.items()
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Load the discharge table, score every estimator, and write the scoreboard."""
    args = parse_args()
    for name, value in vars(args).items(): logger.info("%s = %s", name, value)

    df = pd.read_csv(args.discharge)
    df = df[df["gauge_q"].notna()].reset_index(drop=True)   # keep only gauged (scorable) rows
    obs = df["gauge_q"].to_numpy(dtype=float)
    est = estimators(df)

    common = common_mask(obs, est)
    logger.info(f"[benchmark] {int(common.sum())} rows scorable by all estimators "
                f"(of {len(df)} gauged)")

    out = score_estimators(obs, est, df["basin"].to_numpy(), common)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    logger.info(f"[benchmark] scored {len(df)} gauged rows -> {args.out}")
    with pd.option_context("display.width", 120):
        logger.info(out[out.scope == "overall"].to_string(index=False))


if __name__ == "__main__":
    main()
