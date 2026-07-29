#!/usr/bin/env python3
"""Build the training/inference manifest for the learned-consensus pipeline.

Turns the SWOT Confluence products into one tidy CSV, one row per
(reach, SWOT overpass): the FLPE algorithm discharges (metroman, momma, neobam,
sic4dvar) and Confluence's consensus at that overpass, the reach's ML-prior flow
stats, and -- in `--mode train` -- the observed gauge discharge on that date from
the SWOT Validation Set (SVS), which is the supervised target.

Everything joins on reach_id (v16). The SoS gives per-reach overpass *times*
(seconds since 2000-01-01); the SVS gives *daily* gauge discharge (days since
2023-01-01); we match each overpass to the gauge value on the same calendar day.

    python src/build_manifest.py --sos eu_SOS_mini.nc --svs svs_mini.nc \
        --priors priors_mini.csv --basins 232270 232290 \
        --mode train --out manifest_train.csv
"""
import argparse
import csv
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import netCDF4 as nc

from utils import configure_logging

MISSING = -999999999999.0
SOS_EPOCH = dt.datetime(2000, 1, 1)          # SoS reaches/time units
SVS_EPOCH = dt.date(2023, 1, 1)              # SVS time units (days since)
# FLPE discharge: manifest column -> (SoS group path, variable).
FLPE = {
    "q_metroman": ("metroman", "allq"),
    "q_momma": ("momma", "Q"),
    "q_neobam": ("neobam/q", "q"),
    "q_sic4dvar": ("sic4dvar", "Q_da"),
}
COLUMNS = ["reach_id", "basin", "date", "month",
           "q_metroman", "q_momma", "q_neobam", "q_sic4dvar",
           "q_consensus", "prior_mean_q", "prior_monthly_q", "gauge_q"]


def flat(x: object) -> np.ndarray:
    """Return a netCDF value (possibly a VLEN element) as a 1-D float array."""
    return np.asarray(x, dtype=float).ravel()


def clean(v: float) -> float:
    """Map fill/invalid discharge values to NaN; keep valid positive flows."""
    return v if (np.isfinite(v) and v != MISSING and v > 0) else np.nan


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manifest construction."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sos", required=True, help="SoS results netCDF (mini or full)")
    ap.add_argument("--svs", required=True, help="SWOT Validation Set netCDF (gauge target)")
    ap.add_argument("--priors", default=None, help="priors_mini.csv (ML-prior features); optional")
    ap.add_argument("--basins", nargs="+", type=str, required=True,
                    help="SWORD basin prefixes to include (any level), e.g. 2322 2326")
    ap.add_argument("--mode", choices=["train", "infer"], required=True,
                    help="train: keep gauged rows with >=2 algorithms; infer: keep all reaches")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load_svs(path: Path) -> dict[int, dict[dt.date, list[float]]]:
    """Load SVS gauge discharge as reach_id -> {date -> [daily Q]}.

    Args:
        path: The SVS netCDF (mini or full).

    Returns:
        Nested dict mapping each reach_id (v16) to a map from calendar date to the
        list of observed discharges recorded there (a list because a reach can
        carry more than one gauge station; averaged at lookup time).
    """
    ds = nc.Dataset(path)
    ds.set_auto_mask(False)

    reach = ds["reach_id_v16"][:].astype(np.int64)
    days = ds["time"][:]
    Q = ds["Q"][:]
    dates = [SVS_EPOCH + dt.timedelta(days=int(round(d))) for d in days]

    # Reshape data so it is keyed by reach identifier
    out: dict[int, dict[dt.date, list[float]]] = {}
    for s in range(len(reach)):
        rid = int(reach[s])
        if rid <= 0:
            continue
        per = out.setdefault(rid, {})
        row = Q[s]
        for k, q in enumerate(row):
            qc = clean(float(q))
            if not np.isnan(qc):
                per.setdefault(dates[k], []).append(qc)

    ds.close()
    return out


def load_priors(path: Path | None) -> dict[int, tuple[float, list[float]]]:
    """Load per-reach ML-prior stats as reach_id -> (mean_q, [monthly_q x12])."""
    if not path or not Path(path).exists():
        return {}

    out: dict[int, tuple[float, list[float]]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rid = int(row["reach_id"])
            monthly = [float(row[f"prior_monthly_q_{m:02d}"]) for m in range(1, 13)]
            out[rid] = (float(row["prior_mean_q"]), monthly)
    return out


def load_results(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load the SoS results needed to build the manifest.

    Args:
        path: The SoS results netCDF (mini or full).

    Returns:
        A 4-tuple of:
          * reach_id: per-reach identifiers (v16), shape (n_reach,).
          * times: per-reach overpass times (seconds since 2000-01-01); a VLEN
            array, one variable-length series per reach.
          * cons: per-reach Confluence consensus discharge, aligned index-for-index
            with `times`.
          * algo_series: FLPE-algorithm discharges keyed by manifest column name
            (see `FLPE`), each a per-reach VLEN series aligned with `times`.
    """
    ds = nc.Dataset(path)
    ds.set_auto_mask(False)

    reach_id = ds["reaches"]["reach_id"][:].astype(np.int64)
    times = ds["reaches"]["time"][:]
    cons = ds["consensus"]["consensus_q"][:]

    algo_series = {}
    for col, (grp, var) in FLPE.items():
        node = ds
        for part in grp.split("/"):
            node = node[part]
        algo_series[col] = node[var][:]
    ds.close()

    return reach_id, times, cons, algo_series

def match_data(
    reach_id: np.ndarray,
    basins: list[str],
    algo_series: dict[str, np.ndarray],
    times: np.ndarray,
    svs: dict[int, dict[dt.date, list[float]]],
    priors: dict[int, tuple[float, list[float]]],
    consensus: np.ndarray,
    mode: str,
) -> list[list]:
    """Join the SoS/SVS/priors into one manifest row per (reach, overpass).

    For each reach whose id starts with one of the requested `basins`, walk its
    overpass times and emit a row combining the FLPE algorithm discharges, the
    Confluence consensus, the ML-prior stats, and -- when available -- the gauge
    discharge observed on the same calendar day. Fill-value overpass times are
    skipped, and in ``train`` mode rows are dropped unless they have a gauge value
    and at least two valid algorithm discharges.

    Args:
        reach_id: Per-reach identifiers (v16), as returned by `load_results`.
        basins: SWORD basin prefixes to include; a reach is kept if its id starts
            with any prefix.
        algo_series: FLPE-algorithm VLEN series keyed by manifest column name.
        times: Per-reach overpass times (seconds since 2000-01-01), VLEN.
        svs: Gauge discharge lookup, reach_id -> {date -> [daily Q]}.
        priors: ML-prior stats lookup, reach_id -> (mean_q, [monthly_q x12]).
        consensus: Per-reach Confluence consensus discharge, VLEN.
        mode: ``"train"`` (keep gauged rows with >=2 algorithms) or ``"infer"``
            (keep all reaches).

    Returns:
        Manifest rows, each a list of values ordered to match `COLUMNS`.
    """
    rows: list[list] = []
    for i, rid in enumerate(reach_id):
        rid = int(rid)
        matched = next((p for p in basins if str(rid).startswith(p)), None)
        if matched is None:
            continue

        basin = int(matched)  # group rows by the matched basin prefix
        time_flat = flat(times[i])
        series = {col: flat(s[i]) for col, s in algo_series.items()} # pre-flatten each reach's VLEN series once (aligned index-for-index with t)
        consensus_series = flat(consensus[i])
        prior = priors.get(rid)
        gauge = svs.get(rid, {})

        for time in range(len(time_flat)):
            time_unit = float(time_flat[time])
            if not np.isfinite(time_unit) or time_unit == MISSING or time_unit <= 0:
                continue  # skip fill-value overpass times

            date = (SOS_EPOCH + dt.timedelta(seconds=time_unit)).date()
            month = date.month
            algos = {col: (clean(float(s[time])) if time < len(s) else np.nan)
                        for col, s in series.items()}

            n_algo = sum(not np.isnan(v) for v in algos.values())

            q_cons = clean(float(consensus_series[time])) if time < len(consensus_series) else np.nan
            g_vals = gauge.get(date, [])
            gauge_q = float(np.mean(g_vals)) if g_vals else np.nan

            p_mean = prior[0] if prior else np.nan
            p_month = prior[1][month - 1] if prior else np.nan

            if mode == "train" and (np.isnan(gauge_q) or n_algo < 2):
                continue

            rows.append([rid, basin, date.isoformat(), month,
                        algos["q_metroman"], algos["q_momma"], algos["q_neobam"],
                        algos["q_sic4dvar"], q_cons, p_mean, p_month, gauge_q])

    return rows

def write_output(out_path: str | Path, rows: list[list]) -> int:
    """Write the manifest rows to CSV and return the distinct reach count.

    Args:
        out_path: Destination CSV path; parent directories are created as needed.
        rows: Manifest rows from `match_data`, ordered to match `COLUMNS`.

    Returns:
        The number of distinct reach_ids present in `rows`.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    n_reach = len({r[0] for r in rows})
    return n_reach

def main() -> None:
    """Read the SoS/SVS/priors, join per (reach, overpass), and write the manifest."""
    # Command line arguments + logging
    args = parse_args()
    logger = configure_logging("build_manifest")
    for name, value in vars(args).items(): logger.info("%s = %s", name, value)

    # Load input data
    basins = list(args.basins)
    svs = load_svs(Path(args.svs))
    priors = load_priors(Path(args.priors) if args.priors else None)
    reach_id, times, consensus, algo_series = load_results(Path(args.sos))

    # Match up input source data
    rows = match_data(reach_id, basins, algo_series, times, svs, priors, consensus, args.mode)

    # Write intermediate file with formatted and aligned input data
    n_reach = write_output(args.out, rows)
    logger.info(f"[manifest] mode={args.mode} rows={len(rows)} reaches={n_reach} -> {args.out}")


if __name__ == "__main__":
    main()
