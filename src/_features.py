#!/usr/bin/env python3
"""Shared feature engineering for the learned-consensus pipeline.

Kept in one place so `train_consensus.py`, `predict_discharge.py`, and
`benchmark_consensus.py` build the model's inputs identically -- if training and
prediction disagreed on how a feature is computed, the model would silently see
the wrong thing at inference. The manifest CSV stores *raw* discharge values
(readable); this module turns a manifest DataFrame into the numeric matrix the
MLP consumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Raw discharge columns used as model inputs (all in m^3/s, log-transformed
# below because river discharge spans several orders of magnitude). The four
# FLPE algorithms + Confluence's consensus + the ML-prior flow stats.
DISCHARGE_COLS: list[str] = [
    "q_metroman", "q_momma", "q_neobam", "q_sic4dvar",
    "q_consensus", "prior_mean_q", "prior_monthly_q",
]
# The FLPE algorithm subset, used for row-median imputation and the naive-mean
# baseline in benchmark_consensus.py.
FLPE_COLS: list[str] = ["q_metroman", "q_momma", "q_neobam", "q_sic4dvar"]

TARGET_COL: str = "gauge_q"


def make_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Turn manifest rows into the MLP's raw (un-standardized) feature matrix.

    Each discharge column is log1p-transformed (discharge is heavy-tailed). Missing
    algorithm values are imputed row-wise with the median of that row's available
    log-discharges -- a per-row fill introduces no train/test leakage and needs no
    saved statistics. Month is encoded as sin/cos so December and January are
    adjacent.

    Args:
        df: Manifest rows with the `DISCHARGE_COLS` and a `month` column.

    Returns:
        A tuple `(X, names)`: `X` is a float32 array of shape `[n_rows, n_feats]`
        and `names` are the feature column labels in order.
    """
    # 1. Log-transform each discharge column. clip(0, None) floors negatives/fills
    #    at 0 (discharge can't be negative); log1p maps 0 -> 0 and compresses the
    #    heavy tail so a 10x change is a constant step. Missing values stay NaN.
    logs = {c: np.log1p(np.clip(df[c].to_numpy(dtype=float), 0, None)) for c in DISCHARGE_COLS}

    # 2. Stack the per-column arrays into one [n_feats, n_rows] grid so we can take
    #    statistics down each column of the ORIGINAL table (i.e. across features,
    #    per row) with axis=0.
    stacked = np.vstack([logs[c] for c in DISCHARGE_COLS])          # [n_feats, n_rows]

    # 3. Per-row median fill value: for each row, the median of its available
    #    (finite) log-discharges. `has_any` marks rows with at least one valid
    #    value; rows with none stay 0 (and nanmedian is computed only over the
    #    non-empty rows to avoid an all-NaN-slice warning).
    row_median = np.zeros(stacked.shape[1], dtype=float)
    has_any = np.isfinite(stacked).any(axis=0)
    row_median[has_any] = np.nanmedian(stacked[:, has_any], axis=0)

    # 4. Impute: rebuild each discharge column, keeping real values and swapping any
    #    NaN for that row's median fill. Per-row fill leaks no cross-row info and
    #    needs no saved statistics.
    cols = []
    for c in DISCHARGE_COLS:
        v = logs[c]
        cols.append(np.where(np.isfinite(v), v, row_median))

    # 5. Encode month cyclically as (sin, cos) on the unit circle, so Dec (12) and
    #    Jan (1) land next to each other instead of 11 apart.
    month = df["month"].to_numpy(dtype=float)
    cols.append(np.sin(2 * np.pi * month / 12.0))
    cols.append(np.cos(2 * np.pi * month / 12.0))

    # 6. Assemble the final matrix: name the columns in order, then column-stack the
    #    per-feature arrays into an [n_rows, n_feats] float32 matrix for the model.
    names = list(DISCHARGE_COLS) + ["month_sin", "month_cos"]
    X = np.column_stack(cols).astype(np.float32)
    return X, names


def make_target(df: pd.DataFrame) -> np.ndarray:
    """Return the log1p training target (log gauge discharge), as float32."""
    return np.log1p(np.clip(df[TARGET_COL].to_numpy(dtype=float), 0, None)).astype(np.float32)
